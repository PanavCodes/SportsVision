import supervision as sv


class CourtKeypointsDrawer:
    """
    Draws keypoints (e.g., court landmarks) on video frames using supervision.
    """

    def __init__(self, keypoint_color: str = "#FF2C2C"):
        """
        Initialize the drawer with customizable color.
        """
        self.keypoint_color = keypoint_color
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8
        )
        self.vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

    def draw(self, frames, court_keypoints):
        """
        Annotates keypoints on each frame.
        """
        output_frames = []

        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints = court_keypoints[index]
            keypoints_numpy=keypoints.cpu().numpy()
            # Annotate keypoints and optionally labels
            annotated_frame = self.vertex_annotator.annotate(scene=annotated_frame, key_points=keypoints)
            annotated_frame = self.vertex_label_annotator.annotate(scene=annotated_frame, key_points=keypoints_numpy)

            output_frames.append(annotated_frame)

        return output_frames
