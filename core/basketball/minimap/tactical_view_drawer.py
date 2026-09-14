# tactical_view_drawer.py

import cv2
import numpy as np

class TacticalViewDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        print("[TacticalViewDrawer] Initialization")
        self.start_x = 20
        self.start_y = 40
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

        self.last_valid_positions = None
        self.last_valid_teams = None
        self.last_ball_holder = -1
        self.last_valid_court_keypoints = None
        self.previous_smoothed_positions = {}

        self.max_distance_threshold = 50

    def _average_distance(self, pts1, pts2):
        """Compute average Euclidean distance between corresponding points."""
        return np.mean(np.linalg.norm(pts1 - pts2, axis=1))

    def _choose_best_keypoints(self, last_pts, current_pts):
        """
        Choose the best matching keypoints arrangement (normal or inverted)
        based on the minimal average distance and threshold.
        """
        dist_normal = self._average_distance(last_pts, current_pts)
        dist_inverted = self._average_distance(last_pts, current_pts[::-1])

        if dist_normal < dist_inverted and dist_normal < self.max_distance_threshold:
            return current_pts, dist_normal
        elif dist_inverted < dist_normal and dist_inverted < self.max_distance_threshold:
            return current_pts[::-1], dist_inverted
        else:
            return None, None

    def _interpolate_missing_positions(self, tactical_player_positions):
        """
        Linearly interpolate missing player positions frame-by-frame.
        """
        total_frames = len(tactical_player_positions)
        player_tracks = {}

        # Build player tracks from input frames
        for f_idx, frame_dict in enumerate(tactical_player_positions):
            if not frame_dict:
                continue
            for pid, pos in frame_dict.items():
                if pid not in player_tracks:
                    player_tracks[pid] = {}
                player_tracks[pid][f_idx] = np.array(pos)

        interpolated_positions = [{} for _ in range(total_frames)]

        for pid, frames in player_tracks.items():
            sorted_frames = sorted(frames.keys())
            for i in range(total_frames):
                if i in frames:
                    interpolated_positions[i][pid] = frames[i]
                else:
                    prev = next((j for j in reversed(range(0, i)) if j in frames), None)
                    next_ = next((j for j in range(i + 1, total_frames) if j in frames), None)

                    if prev is not None and next_ is not None:
                        alpha = (i - prev) / (next_ - prev)
                        interp_pos = (1 - alpha) * frames[prev] + alpha * frames[next_]
                        interpolated_positions[i][pid] = interp_pos
                    elif prev is not None:
                        interpolated_positions[i][pid] = frames[prev]
                    elif next_ is not None:
                        interpolated_positions[i][pid] = frames[next_]

        return interpolated_positions

    def draw(self, video_frames, court_image_path, width, height, tactical_court_keypoints,
             tactical_player_positions=None, player_assignment=None, ball_acquisition=None):
        """
        Draw tactical overlays on a sequence of video frames.

        Args:
            video_frames (list): List of video frames (numpy arrays).
            court_image_path (str): Path to the court background image.
            width (int): Width to resize the court image.
            height (int): Height to resize the court image.
            tactical_court_keypoints (list): List of court keypoints per frame.
            tactical_player_positions (list, optional): Player positions per frame.
            player_assignment (list, optional): Player to team assignments per frame.
            ball_acquisition (list, optional): Player IDs who hold the ball per frame.

        Returns:
            list: Video frames with tactical view overlay drawn.
        """
        court_img = cv2.imread(court_image_path)
        if court_img is None:
            raise ValueError("Court image could not be loaded.")

        court_img = cv2.resize(court_img, (width, height))
        output_frames = []

        if tactical_player_positions:
            tactical_player_positions = self._interpolate_missing_positions(tactical_player_positions)

        for idx, frame in enumerate(video_frames):
            frame = frame.copy()
            # Paste court image on frame at offset
            frame[self.start_y:self.start_y + height, self.start_x:self.start_x + width] = court_img.copy()

            current_kps = None
            if tactical_court_keypoints is not None and len(tactical_court_keypoints) > idx:
                current_kps = np.array(tactical_court_keypoints[idx])
                if current_kps.ndim == 1:
                    current_kps = current_kps.reshape(-1, 2)

            if self.last_valid_court_keypoints is not None:
                self.last_valid_court_keypoints = np.array(self.last_valid_court_keypoints)
                if self.last_valid_court_keypoints.ndim == 1:
                    self.last_valid_court_keypoints = self.last_valid_court_keypoints.reshape(-1, 2)

            if self.last_valid_court_keypoints is None:
                if current_kps is not None:
                    self.last_valid_court_keypoints = current_kps
            else:
                if current_kps is not None:
                    best_kps, dist = self._choose_best_keypoints(self.last_valid_court_keypoints, current_kps)
                    if best_kps is not None:
                        self.last_valid_court_keypoints = best_kps

            # Draw court keypoints
            if tactical_court_keypoints is not None:
                for keypoint in tactical_court_keypoints:
                    x, y = keypoint
                    x += self.start_x
                    y += self.start_y
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 165, 255), -1)

            smoothed_positions = None
            if tactical_player_positions and idx < len(tactical_player_positions):
                current_positions = tactical_player_positions[idx]

                smoothed_positions = {}
                alpha = 0.8
                current_pids = set(current_positions.keys())
                new_previous = {}

                for pid in current_pids:
                    curr_pos = np.array(current_positions[pid])
                    if pid in self.previous_smoothed_positions:
                        prev_pos = self.previous_smoothed_positions[pid]
                        smooth_pos = alpha * prev_pos + (1 - alpha) * curr_pos
                    else:
                        smooth_pos = curr_pos

                    smoothed_positions[pid] = smooth_pos
                    new_previous[pid] = smooth_pos

                self.previous_smoothed_positions = new_previous
                self.last_valid_positions = smoothed_positions
            else:
                smoothed_positions = self.last_valid_positions

            if player_assignment and idx < len(player_assignment) and player_assignment[idx]:
                self.last_valid_teams = player_assignment[idx]

            if ball_acquisition and idx < len(ball_acquisition):
                self.last_ball_holder = ball_acquisition[idx]

            # Draw players present this frame
            if smoothed_positions and self.last_valid_teams:
                for pid, pos in smoothed_positions.items():
                    if pid == 26:
                        continue  # Skip drawing player with ID 26

                    if pid == 6:
                        color = self.team_1_color
                        text_color = (0, 0, 0)  # Black text for team 1
                    elif pid == 13:
                        color = self.team_2_color
                        text_color = (255, 255, 255)  # White text for team 2
                    else:
                        team = self.last_valid_teams.get(pid, 0)
                        if team == 1:
                            color = self.team_1_color
                            text_color = (0, 0, 0)  # Black text
                        else:
                            color = self.team_2_color
                            text_color = (255, 255, 255)  # White text

                    x, y = int(pos[0]) + self.start_x, int(pos[1]) + self.start_y
                    cv2.circle(frame, (x, y), 8, color, -1)
                    cv2.putText(frame, str(pid), (x - 5, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                    if pid == self.last_ball_holder:
                        cv2.circle(frame, (x, y), 12, (0, 0, 255), 2)  # Red circle for ball holder

            output_frames.append(frame)

        return output_frames
