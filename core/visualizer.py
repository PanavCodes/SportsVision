import cv2
import numpy as np
import threading

import config


class CourtVisionVisualizer:
    """
    Renders professional broadcast-quality graphics over video frames.
    Features:
    - Team-colored ellipses at player feet (not rectangles)
    - Track ID labels with team-colored backgrounds
    - Ball possession triangle indicator
    - Ball control percentage HUD
    - Pass/Interception live stats
    - Player speed overlay
    - Frame counter display
    """
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_duplex = cv2.FONT_HERSHEY_DUPLEX
        self.team_colors_bgr = {
            1: config.TEAM_1_COLOR[::-1],  # BGR for OpenCV
            2: config.TEAM_2_COLOR[::-1],
            0: (128, 128, 128)             # Grey for referee/unclassified
        }
        # HUD state accumulators
        self.team1_possession_frames = 0
        self.team2_possession_frames = 0
        self.total_team1_passes = 0
        self.total_team2_passes = 0
        self.total_team1_interceptions = 0
        self.total_team2_interceptions = 0
        self.global_frame_counter = 0
        self.total_frames = 0
        self.draw_lock = threading.Lock()
        
        try:
            from sports.common.ball import BallAnnotator
            self.ball_annotator = BallAnnotator(radius=8, buffer_size=15, thickness=-1)
        except ImportError:
            self.ball_annotator = None
    
    def update_colors(self, team1_bgr, team2_bgr):
        """Update team colors dynamically after K-Means discovery."""
        self.team_colors_bgr[1] = team1_bgr
        self.team_colors_bgr[2] = team2_bgr
    
    def set_total_frames(self, total):
        self.total_frames = total
    
    def _get_center(self, bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
    
    def _get_width(self, bbox):
        return int(bbox[2] - bbox[0])
    
    def _draw_ellipse(self, frame, bbox, color, track_id=None, jersey_number=None):
        """Draw an anti-aliased ellipse at player's feet with optional ID label."""
        y2 = int(bbox[3])
        x_center = int((bbox[0] + bbox[2]) / 2)
        width = self._get_width(bbox)
        
        # Ellipse under the player
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width * 0.5), int(0.18 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )
        
        # Track ID and Jersey Number label
        if track_id is not None:
            label = str(track_id)
            if jersey_number is not None:
                label = f"#{jersey_number}"
                
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.5, 1)
            rect_w = max(tw + 10, 30)
            rect_h = 18
            rx1 = x_center - rect_w // 2
            ry1 = y2 + 5
            rx2 = x_center + rect_w // 2
            ry2 = ry1 + rect_h
            
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, cv2.FILLED)
            # Black text on light colors, white text on dark colors
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cv2.putText(frame, label, (rx1 + 5, ry2 - 4), self.font, 0.45, text_color, 1, cv2.LINE_AA)
        
        return frame
    
    def _draw_possession_triangle(self, frame, bbox):
        """Draw a red filled triangle above the player holding the ball."""
        y_top = int(bbox[1])
        x_center = int((bbox[0] + bbox[2]) / 2)
        
        triangle = np.array([
            [x_center, y_top - 5],
            [x_center - 10, y_top - 25],
            [x_center + 10, y_top - 25]
        ])
        cv2.drawContours(frame, [triangle], 0, (0, 0, 255), cv2.FILLED)
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)
        return frame
    
    def _draw_ball_control_hud(self, frame):
        """Draw ball possession percentage HUD in bottom-left corner."""
        h, w = frame.shape[:2]
        
        total = self.team1_possession_frames + self.team2_possession_frames
        t1_pct = (self.team1_possession_frames / total * 100) if total > 0 else 0
        t2_pct = (self.team2_possession_frames / total * 100) if total > 0 else 0
        
        # Box dimensions
        box_w = int(w * 0.22)
        box_h = 70
        bx1 = 10
        by1 = h - box_h - 10
        bx2 = bx1 + box_w
        by2 = h - 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (40, 30, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 187, 220), 1)
        
        # Header
        cv2.putText(frame, "Ball Control", (bx1 + 8, by1 + 18),
                    self.font_duplex, 0.45, (0, 187, 220), 1, cv2.LINE_AA)
        
        # Team percentages
        cv2.putText(frame, f"Team 1: {t1_pct:.1f}%", (bx1 + 8, by1 + 38),
                    self.font_duplex, 0.4, self.team_colors_bgr[1], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Team 2: {t2_pct:.1f}%", (bx1 + 8, by1 + 56),
                    self.font_duplex, 0.4, self.team_colors_bgr[2], 1, cv2.LINE_AA)
        
        return frame
    
    def _draw_pass_stats_hud(self, frame):
        """Draw pass/interception stats HUD next to ball control."""
        h, w = frame.shape[:2]
        
        box_w = int(w * 0.26)
        box_h = 70
        bx1 = int(w * 0.22) + 25
        by1 = h - box_h - 10
        bx2 = bx1 + box_w
        by2 = h - 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (40, 30, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 187, 220), 1)
        
        # Header
        cv2.putText(frame, "Pass(P) & Intercept(I)", (bx1 + 8, by1 + 18),
                    self.font_duplex, 0.38, (0, 187, 220), 1, cv2.LINE_AA)
        
        # Stats
        cv2.putText(frame, f"Team 1: P {self.total_team1_passes} | I {self.total_team1_interceptions}",
                    (bx1 + 8, by1 + 38), self.font_duplex, 0.38, self.team_colors_bgr[1], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Team 2: P {self.total_team2_passes} | I {self.total_team2_interceptions}",
                    (bx1 + 8, by1 + 56), self.font_duplex, 0.38, self.team_colors_bgr[2], 1, cv2.LINE_AA)
        
        return frame
    
    def _draw_frame_counter(self, frame):
        """Draw frame counter in top-right corner."""
        h, w = frame.shape[:2]
        
        if self.total_frames > 0:
            text = f"Frame {self.global_frame_counter}/{self.total_frames}"
        else:
            text = f"Frame {self.global_frame_counter}"
        
        (tw, th), _ = cv2.getTextSize(text, self.font, 0.45, 1)
        
        # Semi-transparent background
        rx1 = w - tw - 20
        ry1 = 8
        rx2 = w - 5
        ry2 = ry1 + th + 12
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, text, (rx1 + 5, ry2 - 5), self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return frame
    
    def _draw_speed_label(self, frame, bbox, speed_kmh):
        """Draw speed overlay near player."""
        if speed_kmh <= 0:
            return frame
        
        x1, y1 = int(bbox[0]), int(bbox[1])
        text = f"{speed_kmh:.0f}km/h"
        
        cv2.putText(frame, text, (x1, y1 - 5), self.font, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        return frame
    
    def draw_frame(self, frame, player_tracks, ball_track,
                   ball_holder=-1, passes_frame=0, interceptions_frame=0,
                   speeds=None):
        """
        Render all broadcast graphics on a single frame.
        
        Args:
            frame: The video frame (BGR numpy array)
            player_tracks: {player_id: {'bbox': [...], 'team': int}}
            ball_track: {1: {'bbox': [...]}} or {}
            ball_holder: Player ID holding the ball (-1 = none)
            passes_frame: 0/1/2 pass event this frame
            interceptions_frame: 0/1/2 interception event this frame
            speeds: {player_id: speed_kmh} or None
        """
        self.global_frame_counter += 1
        
        # Update cumulative stats
        if ball_holder != -1 and ball_holder in player_tracks:
            team = player_tracks[ball_holder].get('team', 0)
            if team == 1:
                self.team1_possession_frames += 1
            elif team == 2:
                self.team2_possession_frames += 1
        
        if passes_frame == 1:
            self.total_team1_passes += 1
        elif passes_frame == 2:
            self.total_team2_passes += 1
        
        if interceptions_frame == 1:
            self.total_team1_interceptions += 1
        elif interceptions_frame == 2:
            self.total_team2_interceptions += 1
        
        # Draw players with ellipses
        for p_id, data in player_tracks.items():
            bbox = data['bbox']
            team = data.get('team', 0)
            jersey = data.get('jersey_number', None)
            color = self.team_colors_bgr.get(team, (200, 200, 200))
            
            # Ellipse + ID label
            self._draw_ellipse(frame, bbox, color, track_id=p_id, jersey_number=jersey)
            
            # Possession triangle
            if p_id == ball_holder:
                self._draw_possession_triangle(frame, bbox)
            
            # Speed label
            if speeds and p_id in speeds:
                self._draw_speed_label(frame, bbox, speeds[p_id])
        
        # Draw ball
        if 1 in ball_track:
            bbox = ball_track[1]['bbox']
            conf = ball_track[1].get('conf', 1.0)
            
            if self.ball_annotator is not None:
                import supervision as sv
                import numpy as np
                # BallAnnotator expects sv.Detections
                detections = sv.Detections(
                    xyxy=np.array([bbox]),
                    confidence=np.array([conf])
                )
                with self.draw_lock:
                    frame = self.ball_annotator.annotate(frame, detections)
            else:
                x_center = int((bbox[0] + bbox[2]) / 2)
                y_center = int((bbox[1] + bbox[3]) / 2)
                cv2.circle(frame, (x_center, y_center), 8, (0, 165, 255), -1)
                cv2.circle(frame, (x_center, y_center), 8, (255, 255, 255), 2)
        
        # Draw HUDs
        self._draw_ball_control_hud(frame)
        self._draw_pass_stats_hud(frame)
        self._draw_frame_counter(frame)
        
        return frame
