# 🖍️ Drawers

This folder contains modules responsible for **visualizing tracking results** on video frames. These modules add intuitive and minimalistic overlays to help highlight players, the ball, and court landmarks in sports videos—especially for basketball analytics.

---

## 📁 Contents

- `player_tracks_drawer.py`  
  ➤ Draws elegant **ellipses** under each tracked player and overlays their unique **track ID** in a filled rectangle above their position.

- `ball_tracks_drawer.py`  
  ➤ Draws a **green triangle pointer** to indicate the detected position of the ball in each frame.

- `team_ball_control_drawer.py`  
  ➤ Displays a **clean overlay box** on each frame showing **ball possession statistics** between the two teams over time. It calculates cumulative control percentages and updates them in real-time using distinct colors:  
  - **Team 1**: Blue  
  - **Team 2**: Crimson

- `court_keypoints_drawer.py`  
  ➤ Visualizes **keypoints** (e.g., court landmarks or reference points) on each video frame. Each point is marked with a red circle and a white label for easy interpretation.

- `tactical_view_drawer.py`  
  ➤ Overlays a **semi-transparent tactical court image** on the video frame to provide spatial context. You can place the overlay at a fixed position and control its transparency.

---

## ✅ Example Usage

```python
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer,
    CourtKeypointsDrawer,
    TacticalViewDrawer
)

# Initialize drawers
player_drawer = PlayerTracksDrawer()
ball_drawer = BallTracksDrawer()
team_control_drawer = TeamBallControlDrawer()
court_drawer = CourtKeypointsDrawer()
tactical_drawer = TacticalViewDrawer()

# Step 1 (optional): Draw tactical court overlay
court_overlay_frames = tactical_drawer.draw(video_frames, court_image_path="assets/court.png", width=300, height=160)

# Step 2 (optional): Draw court keypoints
keypoint_frames = court_drawer.draw(court_overlay_frames, court_keypoints)

# Step 3: Draw player ellipses
player_frames = player_drawer.draw(keypoint_frames, player_tracks, player_assignments, ball_acquisition)

# Step 4: Draw ball triangle
combined_frames = ball_drawer.draw(player_frames, ball_tracks)

# Step 5: Overlay team ball possession percentages
final_frames = team_control_drawer.draw(combined_frames, player_assignments, ball_acquisition)
