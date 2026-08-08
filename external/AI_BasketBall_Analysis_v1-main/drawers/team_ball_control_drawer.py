import cv2
import numpy as np

class TeamBallControlDrawer:
    """
    Calculates and draws team ball possession percentages over time on video frames.
    """

    def __init__(self):
        pass

    def get_team_ball_control(self, player_assignment, ball_aquisition):
        """
        Calculate which team has ball control for each frame.
        """
        team_ball_control = []
        for pa_frame, ba_frame in zip(player_assignment, ball_aquisition):
            if ba_frame == -1 or ba_frame not in pa_frame:
                team_ball_control.append(0)  # no control
            else:
                team = pa_frame[ba_frame]
                team_ball_control.append(team if team in [1, 2] else 0)
        return np.array(team_ball_control)

    def draw(self, video_frames, player_assignment, ball_aquisition):
        """
        Draw possession statistics on frames.
        """
        team_ball_control = self.get_team_ball_control(player_assignment, ball_aquisition)
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            # Include first frame as well
            frame_with_stats = self.draw_frame(frame, frame_num, team_ball_control)
            output_frames.append(frame_with_stats)

        return output_frames

    

    def draw_frame(self, frame, frame_num, team_ball_control):
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1

        h, w = frame.shape[:2]

        # Ball control box parameters
        rect_w = int(w * 0.20)
        rect_h = int(h * 0.075)
        padding = 8
        spacing = 20
        y_offset = 40  # décalage vertical supplémentaire

        # Position the ball control box at bottom left
        rect_x1 = 10
        rect_y2 = h - int(h * 0.15) - 15 + y_offset
        rect_y1 = rect_y2 - rect_h
        rect_x2 = rect_x1 + rect_w

        # Header height and positions
        header_h = int(rect_h * 0.5)
        header_y1 = rect_y1 - header_h
        header_y2 = rect_y1

        # Text positions inside ball control box
        text_x = rect_x1 + padding
        text_y1 = rect_y1 + padding + 10
        text_y2 = text_y1 + spacing
        header_text_y = header_y1 + int(header_h * 0.7)

        # Colors
        brown_bg = (92, 64, 51)
        vivid_blue = (0,187,220)
        # Team colors as given 
        team_1_color = (255, 245, 238)
        team_2_color = (128, 0, 0)    

        # Draw entire box background brown (including header)
        cv2.rectangle(frame, (rect_x1, header_y1), (rect_x2, rect_y2), brown_bg, -1)

        # Calculate possession percentages so far
        control_so_far = team_ball_control[:frame_num + 1]
        team1_frames = np.sum(control_so_far == 1)
        team2_frames = np.sum(control_so_far == 2)
        total = team1_frames + team2_frames

        if total == 0:
            team_1_pct, team_2_pct = 0.0, 0.0
        else:
            team_1_pct = (team1_frames / total) * 100
            team_2_pct = (team2_frames / total) * 100

        # Draw header text (vivid blue)
        cv2.putText(frame, "Ball Control", (text_x, header_text_y),
                    font, font_scale, vivid_blue, font_thickness, lineType=cv2.LINE_AA)

        # Draw team percentages using their colors (text only, no background box)
        cv2.putText(frame, f"Team 1: {team_1_pct:.1f}%", (text_x, text_y1),
                    font, font_scale, team_1_color, font_thickness, lineType=cv2.LINE_AA)

        cv2.putText(frame, f"Team 2: {team_2_pct:.1f}%", (text_x, text_y2),
                    font, font_scale, team_2_color, font_thickness, lineType=cv2.LINE_AA)

        return frame
