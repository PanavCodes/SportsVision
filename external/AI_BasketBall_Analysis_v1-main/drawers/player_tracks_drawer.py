# player_tracks_drawer.py

"""
Draws tracked player positions on video frames using ellipses and highlights the player in possession.
"""

import cv2
from typing import List, Dict, Any
from .utils import draw_ellipse, draw_triangle


class PlayerTracksDrawer:
    """
    Class for drawing tracked player positions on video frames using ellipses.
    """

    def __init__(self, team1_color: List[int] = [255, 245, 238], team2_color: List[int] = [128, 0, 0]):
        """
        Initializes the drawer with team colors.
        """
        self.default_player_team_id = 1
        self.team1_color = team1_color
        self.team2_color = team2_color

    def draw(
        self,
        video_frames: List[Any],
        tracks: List[Dict[int, Dict[str, Any]]],
        player_assignment: List[Dict[int, int]],
        ball_aquisition: List[int]
    ) -> List[Any]:
        """
        Draws ellipses around tracked player positions and a triangle for the player in possession.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame_copy = frame.copy()
            player_dict = tracks[frame_num]
            assignment_dict = player_assignment[frame_num]
            id_ball_handler = ball_aquisition[frame_num]

            for track_id, player_data in player_dict.items():
                team_id = assignment_dict.get(track_id, self.default_player_team_id)
                color = self.team1_color if team_id == 1 else self.team2_color
                bbox = player_data.get('bbox')

                # Highlight player in possession with a red triangle
                if track_id == id_ball_handler and bbox:
                    frame_copy = draw_triangle(frame_copy, bbox, color=(0, 0, 255))

                # Draw ellipse around player
                if bbox:
                    frame_copy = draw_ellipse(frame_copy, bbox, color=color)

            output_video_frames.append(frame_copy)

        return output_video_frames
