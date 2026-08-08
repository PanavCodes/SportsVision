import cv2
from typing import List, Tuple


class PassInterceptionDrawer:
    def __init__(
        self,
        team_1_color: List[int] = [255, 245, 238],
        team_2_color: List[int] = [128, 0, 0],
    ):
        """
        Initialize the drawer with team colors.
        """
        # Convert RGB to BGR for OpenCV (OpenCV uses BGR order)
        self.team_colors = {
            1: tuple(reversed(team_1_color)),
            2: tuple(reversed(team_2_color)),
        }

    def get_stats(
        self,
        passes: List[int],
        interceptions: List[int],
    ) -> Tuple[int, int, int, int]:
        """
        Count total passes and interceptions for each team.
        """
        team1_passes = 0
        team2_passes = 0
        team1_interceptions = 0
        team2_interceptions = 0

        for pass_frame, interception_frame in zip(passes, interceptions):
            if pass_frame == 1:
                team1_passes += 1
            elif pass_frame == 2:
                team2_passes += 1
            if interception_frame == 1:
                team1_interceptions += 1
            elif interception_frame == 2:
                team2_interceptions += 1

        return team1_passes, team2_passes, team1_interceptions, team2_interceptions

    def draw(
        self,
        video_frames: List,
        passes: List[int],
        interceptions: List[int],
    ) -> List:
        """
        Draw pass and interception stats on each video frame.
        """
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame_drawn = self.draw_frame(frame.copy(), frame_num, passes, interceptions)
            output_frames.append(frame_drawn)
        return output_frames

    def draw_frame(
        self,
        frame,
        frame_num: int,
        passes: List[int],
        interceptions: List[int],
    ):
        """
        Draw pass/interception stats box on a single frame.
        """
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1

        h, w = frame.shape[:2]

        # Box dimensions and positioning
        rect_w = int(w * 0.20)
        rect_h = int(h * 0.075)
        padding = 8
        spacing = 20
        y_offset = 40

        ball_ctrl_x1 = 10
        ball_ctrl_y2 = h - int(h * 0.15) - 15 + y_offset
        ball_ctrl_y1 = ball_ctrl_y2 - rect_h
        ball_ctrl_x2 = ball_ctrl_x1 + rect_w

        horizontal_gap = 20
        pi_x1 = ball_ctrl_x2 + horizontal_gap
        pi_y2 = ball_ctrl_y2
        pi_y1 = pi_y2 - rect_h
        pi_x2 = pi_x1 + rect_w

        header_h = int(rect_h * 0.5)
        header_y1 = pi_y1 - header_h
        header_y2 = pi_y1

        text_x = pi_x1 + padding
        text_y1 = pi_y1 + padding + 10
        text_y2 = text_y1 + spacing
        header_text_y = header_y1 + int(header_h * 0.7)

        # Colors in BGR
        brown_bg = (92, 64, 51)
        vivid_blue = (0, 187, 220)

        team_1_color = self.team_colors[1]
        team_2_color = self.team_colors[2]

        # Draw background box with border
        cv2.rectangle(frame, (pi_x1, header_y1), (pi_x2, pi_y2), brown_bg, thickness=-1)
        cv2.rectangle(frame, (pi_x1, header_y1), (pi_x2, pi_y2), (200, 200, 200), thickness=1)

        # Compute cumulative stats up to current frame
        passes_till_now = passes[: frame_num + 1]
        interceptions_till_now = interceptions[: frame_num + 1]
        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_now, interceptions_till_now
        )

        # Draw header text
        cv2.putText(
            frame,
            "Pass(P) & Interception(I)",
            (text_x, header_text_y),
            font,
            font_scale,
            vivid_blue,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

        # Draw team stats text
        cv2.putText(
            frame,
            f"Team 1: P {team1_passes} | I {team1_interceptions}",
            (text_x, text_y1),
            font,
            font_scale,
            team_1_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Team 2: P {team2_passes} | I {team2_interceptions}",
            (text_x, text_y2),
            font,
            font_scale,
            team_2_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

        return frame
