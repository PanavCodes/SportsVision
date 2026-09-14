from collections import Counter
import config
from core.sport_router.frame_sampler import FrameSampler
from core.sport_router.classifier import SportClassifier

class SportRouter:
    """
    Automatic sport router for SportsVision:
      Samples representative video frames, evaluates visual sport signatures,
      and routes the input video to the appropriate sport pipeline.
    """
    def __init__(self, num_samples=5, confidence_threshold=0.60):
        self.sampler = FrameSampler(num_samples=num_samples)
        self.classifier = SportClassifier()
        self.confidence_threshold = confidence_threshold

    def route(self, video_path: str) -> str:
        """
        Determines the sport category of the video.
        Returns:
            "basketball" | "cricket" | "unknown"
        """
        print(f"[SportRouter] Analyzing video for sport detection: {video_path}")
        samples = self.sampler.sample_frames(video_path)

        if not samples:
            print("[SportRouter] Could not read frames. Defaulting to 'unknown'.")
            return "unknown"

        votes = []
        confidences = []

        for ts, frame in samples:
            sport, conf = self.classifier.classify_frame(frame)
            votes.append(sport)
            confidences.append(conf)

        counts = Counter(votes)
        majority_sport, vote_count = counts.most_common(1)[0]
        avg_confidence = sum(confidences) / len(confidences)

        print(f"[SportRouter] Detection Results: {dict(counts)} (Avg Conf: {avg_confidence:.2f})")

        # If majority is decisive
        if vote_count >= (len(samples) // 2 + 1) and avg_confidence >= self.confidence_threshold:
            print(f"[SportRouter] Video classified as: {majority_sport.upper()}")
            return majority_sport

        print("[SportRouter] Confidence below threshold. Flagged as 'unknown'.")
        return "unknown"


def detect_sport(video_path: str) -> str:
    """Convenience functional wrapper for sport routing."""
    router = SportRouter()
    return router.route(video_path)
