import cv2
import numpy as np
import os
from typing import Optional, Dict, List, Tuple
import config

class CricketVisualizer:
    """
    Broadcast-Grade Cricket HUD & DRS Visualizer:
      - Ball flight trail & Hawkeye predictive path
      - Ground bounce landing ring with angle annotation
      - Top-down pitch minimap overlay with landing position
      - Speed gun speedometer badge (km/h)
      - Expected Dismissal (xD) probability telemetry badge
      - Batting shot type & match event banners (Four, Six, Wicket, Dot)
      - Official ICC DRS / Hawk-Eye decision review overlay card
      - Role-coded player tracking indicators
    """
    def __init__(self, pitch_template_path=None):
        self.pitch_template_path = pitch_template_path or config.CRICKET_PITCH_IMAGE_PATH
        self.pitch_minimap_img = None
        if os.path.exists(self.pitch_template_path):
            img = cv2.imread(self.pitch_template_path)
            if img is not None:
                self.pitch_minimap_img = cv2.resize(img, (260, 130))
        # Procedural fallback to guard against missing/corrupted pitch image (Bug 11)
        if self.pitch_minimap_img is None:
            self.pitch_minimap_img = self._generate_fallback_pitch(260, 130)

        # Cricket Field Ground Radar Image
        self.ground_template_path = getattr(config, 'CRICKET_GROUND_IMAGE_PATH', None)
        self.ground_radar_img = None
        if self.ground_template_path and os.path.exists(self.ground_template_path):
            g_img = cv2.imread(self.ground_template_path)
            if g_img is not None:
                self.ground_radar_img = cv2.resize(g_img, (150, 150))

        # Role colors (BGR)
        self.role_colors = {
            'batsman': (0, 215, 255),       # Gold / Yellow
            'bowler': (255, 144, 30),       # Deep Sky Blue
            'wicketkeeper': (200, 50, 200), # Magenta
            'player': (200, 200, 200),      # Silver/White
            'umpire': (50, 200, 50)         # Green
        }

        self.current_speed_kmh = 0.0
        self.current_event = None
        self.current_shot = None
        self.current_line_length = ""
        self.current_xd = None
        self.drs_banner_img = None
        self.drs_banner_frames_left = 0
        self.persistent_minimap_trail = []
        self.persistent_minimap_bounce = None
        self.persistent_minimap_projected = []
        self.minimap_persistence_frames_left = 0
        self.current_over = "0.1"

    def _generate_fallback_pitch(self, w: int = 260, h: int = 130) -> np.ndarray:
        """Procedural synthetic turf pitch graphic if image asset is unavailable."""
        canvas = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)  # grass green
        pw, ph = int(w * 0.75), int(h * 0.32)
        px1, py1 = (w - pw) // 2, (h - ph) // 2
        canvas[py1:py1+ph, px1:px1+pw] = (140, 180, 210)  # tan pitch turf
        # crease markings
        cv2.line(canvas, (px1 + 16, py1), (px1 + 16, py1 + ph), (255, 255, 255), 1)
        cv2.line(canvas, (px1 + pw - 16, py1), (px1 + pw - 16, py1 + ph), (255, 255, 255), 1)
        # stumps
        cv2.circle(canvas, (px1 + 6, py1 + ph // 2), 2, (255, 255, 255), -1)
        cv2.circle(canvas, (px1 + pw - 6, py1 + ph // 2), 2, (255, 255, 255), -1)
        return canvas

    def update_telemetry(
        self,
        speed_kmh: float = 0.0,
        event: Optional[str] = None,
        shot: Optional[str] = None,
        line_length: str = "",
        xd_info: Optional[dict] = None,
        over_str: Optional[str] = None
    ):
        if speed_kmh > 0:
            self.current_speed_kmh = speed_kmh
        if event:
            self.current_event = event
        if shot:
            self.current_shot = shot
        if line_length:
            self.current_line_length = line_length
        if xd_info:
            self.current_xd = xd_info
        if over_str:
            self.current_over = over_str

    def trigger_drs_display(self, drs_banner: np.ndarray, duration_frames: int = 45):
        """Displays the DRS adjudication card for a given number of frames."""
        self.drs_banner_img = drs_banner
        self.drs_banner_frames_left = duration_frames

    def draw_frame(
        self,
        frame: np.ndarray,
        player_track_frame: dict,
        ball_track_frame: dict,
        trajectory_trail=None,
        projected_trail=None,
        bounce_points=None,
        minimap_ball_pos=None,
        minimap_trail=None,
        minimap_bounce=None,
        minimap_projected=None,
        stumps_info=None,
        ground_ball_pos=None,
        ground_trail=None,
        ground_player_positions=None
    ) -> np.ndarray:
        """
        Draw complete cricket broadcast graphics onto a single video frame.
        """
        out_frame = frame.copy()
        h, w, _ = out_frame.shape

        # 1. Draw Player Boxes and Role Tags
        if player_track_frame:
            for pid, info in player_track_frame.items():
                bbox = info['bbox']
                role = info.get('role', 'player')
                color = self.role_colors.get(role, (200, 200, 200))

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)

                # Role Label Badge
                label = f"{role.upper()} #{pid}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out_frame, (x1, max(0, y1 - 20)), (x1 + lw + 8, y1), color, -1)
                cv2.putText(out_frame, label, (x1 + 4, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw Trajectory Trail (Smooth Glowing Arc)
        if trajectory_trail and len(trajectory_trail) > 1:
            for i in range(1, len(trajectory_trail)):
                p1 = trajectory_trail[i - 1]
                p2 = trajectory_trail[i]
                cv2.line(out_frame, p1, p2, (255, 220, 0), 3, cv2.LINE_AA)

        # 3. Draw Hawkeye Predictive Trail
        if projected_trail and len(projected_trail) > 1:
            for i in range(1, len(projected_trail)):
                p1 = projected_trail[i - 1]
                p2 = projected_trail[i]
                cv2.line(out_frame, p1, p2, (0, 255, 128), 2, cv2.LINE_AA)
                cv2.circle(out_frame, p2, 2, (0, 255, 128), -1)

        # 4. Draw Ball Indicator
        if ball_track_frame and 1 in ball_track_frame:
            b_info = ball_track_frame[1]
            bx, by = b_info['centroid']
            cv2.circle(out_frame, (int(bx), int(by)), 6, (0, 0, 255), -1)
            cv2.circle(out_frame, (int(bx), int(by)), 9, (0, 255, 255), 2, cv2.LINE_AA)

        # 5. Draw Bounce Landing Marker
        if bounce_points:
            for bp in bounce_points:
                bx, by = bp['coord']
                cv2.circle(out_frame, (int(bx), int(by)), 8, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.circle(out_frame, (int(bx), int(by)), 15, (0, 100, 255), 1, cv2.LINE_AA)
                cv2.putText(out_frame, f"PITCH {bp['angle_out']:.1f}°", (int(bx) + 12, int(by) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        # 6. Speed Gun & xD Telemetry Badge (Top Left HUD)
        speed_badge_w, speed_badge_h = 250, 75
        cv2.rectangle(out_frame, (20, 20), (20 + speed_badge_w, 20 + speed_badge_h), (25, 25, 25), -1)
        cv2.rectangle(out_frame, (20, 20), (20 + speed_badge_w, 20 + speed_badge_h), (0, 165, 255), 2)

        spd_text = f"{self.current_speed_kmh:.1f} KM/H" if self.current_speed_kmh > 0 else "---.- KM/H"
        cv2.putText(out_frame, "BALL SPEED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        if getattr(self, 'current_over', None):
            cv2.putText(out_frame, f"OV: {self.current_over}", (155, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(out_frame, spd_text, (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 255), 2, cv2.LINE_AA)

        # Expected Dismissal (xD) Badge next to speed gun
        if self.current_xd:
            xd_pct = self.current_xd.get('xd_pct', 0.0)
            threat = self.current_xd.get('threat_level', 'LOW')
            t_color = (0, 0, 255) if threat in ['HIGH', 'EXTREME'] else (0, 215, 255)

            xd_w, xd_h = 160, 75
            xd_x = 20 + speed_badge_w + 10
            cv2.rectangle(out_frame, (xd_x, 20), (xd_x + xd_w, 20 + xd_h), (25, 25, 25), -1)
            cv2.rectangle(out_frame, (xd_x, 20), (xd_x + xd_w, 20 + xd_h), t_color, 2)
            cv2.putText(out_frame, "EXPECTED OUT (xD)", (xd_x + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(out_frame, f"{xd_pct:.1f}%", (xd_x + 10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, t_color, 2, cv2.LINE_AA)

        # 7. Line & Length / Shot Banner (Bottom Left HUD)
        if self.current_line_length or self.current_shot:
            hud_y = h - 60
            cv2.rectangle(out_frame, (20, hud_y - 30), (420, hud_y + 30), (20, 20, 20), -1)
            cv2.rectangle(out_frame, (20, hud_y - 30), (420, hud_y + 30), (255, 255, 255), 1)

            banner_txt = f"{self.current_line_length}"
            if self.current_shot:
                banner_txt += f" | {self.current_shot.upper().replace('_', ' ')}"
            cv2.putText(out_frame, banner_txt, (32, hud_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # 8. Event Pop-up Banner (Top Center)
        if self.current_event in ["four", "six", "wicket", "lbw", "out"]:
            event_text = self.current_event.upper()
            if event_text == "FOUR":
                event_text = "FOUR! BOUNDARY 4"
                evt_color = (0, 200, 255)
            elif event_text == "SIX":
                event_text = "MAXIMUM! SIX 6"
                evt_color = (0, 255, 0)
            elif "LBW" in event_text:
                event_text = "LBW! HAWK-EYE REVIEW"
                evt_color = (0, 0, 255)
            else:
                event_text = "WICKET! OUT"
                evt_color = (0, 0, 255)

            cx = w // 2
            cv2.rectangle(out_frame, (cx - 180, 20), (cx + 180, 65), (15, 15, 15), -1)
            cv2.rectangle(out_frame, (cx - 180, 20), (cx + 180, 65), evt_color, 2)
            cv2.putText(out_frame, event_text, (cx - 160, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, evt_color, 2, cv2.LINE_AA)

        # 9. Top-Down Pitch Minimap with Real-Time Ball Flight & Landing Trail
        if self.pitch_minimap_img is not None:
            mw, mh = 260, 130
            mx_start = w - mw - 20
            my_start = 20

            # Place base pitch image
            out_frame[my_start:my_start + mh, mx_start:mx_start + mw] = self.pitch_minimap_img.copy()
            cv2.rectangle(out_frame, (mx_start, my_start), (mx_start + mw, my_start + mh), (255, 255, 255), 1)
            cv2.putText(out_frame, "22-YD PITCH TRACK", (mx_start + 6, my_start + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

            scale_x = mw / 600.0
            scale_y = mh / 320.0

            # Update persistent trail if active data provided
            if minimap_trail and len(minimap_trail) > 0:
                self.persistent_minimap_trail = list(minimap_trail)
                self.minimap_persistence_frames_left = 60
            if minimap_bounce is not None:
                self.persistent_minimap_bounce = minimap_bounce
                self.minimap_persistence_frames_left = 60
            if minimap_projected and len(minimap_projected) > 0:
                self.persistent_minimap_projected = list(minimap_projected)
                self.minimap_persistence_frames_left = 60

            # A. Draw Flight Trail on Mini Field (Cyan flight arc)
            active_trail = minimap_trail or (self.persistent_minimap_trail if self.minimap_persistence_frames_left > 0 else [])
            if active_trail and len(active_trail) > 1:
                for i in range(1, len(active_trail)):
                    pt1 = active_trail[i - 1]
                    pt2 = active_trail[i]
                    p1_sc = (int(mx_start + pt1[0] * scale_x), int(my_start + pt1[1] * scale_y))
                    p2_sc = (int(mx_start + pt2[0] * scale_x), int(my_start + pt2[1] * scale_y))
                    cv2.line(out_frame, p1_sc, p2_sc, (255, 220, 0), 2, cv2.LINE_AA)

            # B. Draw Projected Future Path on Mini Field (Lime Green dashed line towards stumps)
            active_proj = minimap_projected or (self.persistent_minimap_projected if self.minimap_persistence_frames_left > 0 else [])
            if active_proj and len(active_proj) > 1:
                for i in range(1, len(active_proj)):
                    pt1 = active_proj[i - 1]
                    pt2 = active_proj[i]
                    p1_sc = (int(mx_start + pt1[0] * scale_x), int(my_start + pt1[1] * scale_y))
                    p2_sc = (int(mx_start + pt2[0] * scale_x), int(my_start + pt2[1] * scale_y))
                    cv2.line(out_frame, p1_sc, p2_sc, (0, 255, 128), 2, cv2.LINE_AA)

            # C. Draw Bounce Landing Spot on Mini Field (Concentric target rings)
            active_bounce = minimap_bounce or (self.persistent_minimap_bounce if self.minimap_persistence_frames_left > 0 else None)
            if active_bounce is not None:
                bx_sc = int(mx_start + active_bounce[0] * scale_x)
                by_sc = int(my_start + active_bounce[1] * scale_y)
                cv2.circle(out_frame, (bx_sc, by_sc), 7, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.circle(out_frame, (bx_sc, by_sc), 3, (0, 255, 255), -1, cv2.LINE_AA)

            # D. Draw Current Ball Head Marker
            if minimap_ball_pos is not None and minimap_ball_pos[0] is not None and minimap_ball_pos[1] is not None:
                m_px = int(mx_start + minimap_ball_pos[0] * scale_x)
                m_py = int(my_start + minimap_ball_pos[1] * scale_y)
                cv2.circle(out_frame, (m_px, m_py), 4, (0, 0, 255), -1)
                cv2.circle(out_frame, (m_px, m_py), 6, (0, 255, 255), 1)

            if self.minimap_persistence_frames_left > 0:
                self.minimap_persistence_frames_left -= 1

        # 10. Cricket Ground Field Radar (Full Ground Overlay: Players & Ball Travel)
        if self.ground_radar_img is not None:
            gw, gh = 150, 150
            gx_start = w - gw - 20
            # Place directly below the 22-yd pitch minimap
            gy_start = 20 + 130 + 10 if self.pitch_minimap_img is not None else 20

            if gy_start + gh < h - 20:
                out_frame[gy_start:gy_start + gh, gx_start:gx_start + gw] = self.ground_radar_img.copy()
                cv2.rectangle(out_frame, (gx_start, gy_start), (gx_start + gw, gy_start + gh), (255, 255, 255), 1)
                cv2.putText(out_frame, "FIELD RADAR", (gx_start + 6, gy_start + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

                # Ground image scale factor: original image was 538x530, resized to 150x150
                g_scale_x = gw / 538.0
                g_scale_y = gh / 530.0

                # A. Draw Players on Field Radar
                if ground_player_positions:
                    for role, pos in ground_player_positions:
                        if pos and pos[0] is not None and pos[1] is not None:
                            gpx = int(gx_start + pos[0] * g_scale_x)
                            gpy = int(gy_start + pos[1] * g_scale_y)
                            p_color = self.role_colors.get(role, (220, 220, 220))
                            cv2.circle(out_frame, (gpx, gpy), 4, p_color, -1)
                            cv2.circle(out_frame, (gpx, gpy), 5, (0, 0, 0), 1)

                # B. Draw Ball Trail Across Ground
                active_gtrail = ground_trail or []
                if active_gtrail and len(active_gtrail) > 1:
                    for i in range(1, len(active_gtrail)):
                        pt1 = active_gtrail[i - 1]
                        pt2 = active_gtrail[i]
                        p1_sc = (int(gx_start + pt1[0] * g_scale_x), int(gy_start + pt1[1] * g_scale_y))
                        p2_sc = (int(gx_start + pt2[0] * g_scale_x), int(gy_start + pt2[1] * g_scale_y))
                        cv2.line(out_frame, p1_sc, p2_sc, (0, 255, 255), 1, cv2.LINE_AA)

                # C. Draw Current Ball Head Marker on Field Radar
                if ground_ball_pos and ground_ball_pos[0] is not None and ground_ball_pos[1] is not None:
                    gbx = int(gx_start + ground_ball_pos[0] * g_scale_x)
                    gby = int(gy_start + ground_ball_pos[1] * g_scale_y)
                    cv2.circle(out_frame, (gbx, gby), 4, (0, 0, 255), -1)
                    cv2.circle(out_frame, (gbx, gby), 6, (0, 255, 255), 1)

        # 11. DRS Hawk-Eye Overlay Banner (Center Screen Overlay)
        if self.drs_banner_frames_left > 0 and self.drs_banner_img is not None:
            bw, bh = self.drs_banner_img.shape[1], self.drs_banner_img.shape[0]
            # Resize if needed to fit comfortably
            target_bw = min(bw, int(w * 0.75))
            target_bh = int(bh * (target_bw / bw))
            resized_banner = cv2.resize(self.drs_banner_img, (target_bw, target_bh))

            bx_start = (w - target_bw) // 2
            by_start = (h - target_bh) // 2

            # Alpha blend onto frame
            roi = out_frame[by_start:by_start + target_bh, bx_start:bx_start + target_bw]
            cv2.addWeighted(roi, 0.15, resized_banner, 0.85, 0, roi)
            cv2.rectangle(out_frame, (bx_start, by_start), (bx_start + target_bw, by_start + target_bh), (255, 255, 255), 2)
            self.drs_banner_frames_left -= 1

        return out_frame
