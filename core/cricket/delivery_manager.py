from typing import Dict, Optional, Tuple, List
from enum import Enum

class DeliveryState(Enum):
    IDLE = "IDLE"
    IN_FLIGHT = "IN_FLIGHT"
    COMPLETED = "COMPLETED"
    DEAD_TIME = "DEAD_TIME"

class DeliveryLifecycleManager:
    """
    Delivery Lifecycle & Long-Video Multi-Over Manager.
    
    Segments continuous broadcast cricket footage (multi-over highlights, full matches)
    into distinct individual deliveries:
      1. Detects bowler release & delivery start (ball in high-speed flight).
      2. Tracks active flight and impact window.
      3. Automatically detects delivery completion / dead ball.
      4. Tracks Over counts (0.1, 0.2 ... 0.6 -> 1.1, 1.2).
      5. Filters out non-pitch cuts (replays, crowd closeups, scoreboards).
      6. Triggers trajectory buffer resets so balls don't merge together.
    """
    def __init__(self, fps: float = 30.0, dead_time_frames_threshold: int = 16):
        self.fps = fps
        self.dead_time_threshold = dead_time_frames_threshold

        self.state = DeliveryState.IDLE
        self.total_deliveries = 0
        self.overs_completed = 0
        self.balls_in_current_over = 0

        self.active_delivery_frames = 0
        self.consecutive_lost_frames = 0
        self.current_delivery_points = []
        self.last_ball_pos = None

    def get_over_display(self, is_active: bool = False) -> str:
        """
        Returns standard broadcast over notation.
        When a delivery is actively in flight: displays the ball being bowled (e.g. '0.1').
        When between deliveries: displays the last completed ball notation or next upcoming ball.
        """
        if is_active:
            ball_num = self.balls_in_current_over + 1
            return f"{self.overs_completed}.{ball_num}"
        # Completed / current over status
        completed_balls = self.balls_in_current_over
        return f"{self.overs_completed}.{completed_balls}"

    def update(
        self,
        frame_idx: int,
        ball_info: Dict,
        ball_metric: Optional[Tuple[float, float]],
        player_track_frame: Dict
    ) -> Tuple[bool, bool, str]:
        """
        Process a single frame through the delivery state machine.
        Returns:
            is_delivery_active (bool): True if ball is actively in flight.
            is_delivery_just_completed (bool): True on the exact frame the ball is completed.
            current_over_str (str): Formatted over string e.g. '0.1'.
        """
        has_ball = (1 in ball_info)
        has_batsman = any(info.get('role') == 'batsman' for info in player_track_frame.values())

        is_delivery_just_completed = False

        if has_ball:
            cx, cy = ball_info[1]['centroid']
            self.consecutive_lost_frames = 0

            if self.state in [DeliveryState.IDLE, DeliveryState.DEAD_TIME]:
                # Start new delivery only if ball is tracked with batsman on screen
                # (prevents replay cuts and isolated ball detections from false-starting)
                if has_batsman:
                    self.state = DeliveryState.IN_FLIGHT
                    self.active_delivery_frames = 1
                    self.current_delivery_points = [(cx, cy)]
            elif self.state == DeliveryState.IN_FLIGHT:
                self.active_delivery_frames += 1
                self.current_delivery_points.append((cx, cy))

            self.last_ball_pos = (cx, cy)
        else:
            self.consecutive_lost_frames += 1

            if self.state == DeliveryState.IN_FLIGHT:
                # Delivery completes when ball is lost for > dead_time_threshold after a valid flight
                if self.consecutive_lost_frames >= self.dead_time_threshold:
                    if self.active_delivery_frames >= 5:
                        # Legitimate delivery completed
                        is_delivery_just_completed = True
                        self._advance_delivery()
                        over_str = f"{self.overs_completed}.{self.balls_in_current_over}"
                        if self.balls_in_current_over >= 6:
                            self.overs_completed += 1
                            self.balls_in_current_over = 0

                    self.state = DeliveryState.IDLE
                    self.active_delivery_frames = 0
                    self.current_delivery_points = []

        # If no batsman on screen and no ball, classify as dead time / replay
        if not has_batsman and not has_ball and self.state == DeliveryState.IDLE:
            self.state = DeliveryState.DEAD_TIME

        is_delivery_active = (self.state == DeliveryState.IN_FLIGHT)
        if not is_delivery_just_completed:
            over_str = self.get_over_display(is_active=is_delivery_active)

        return is_delivery_active, is_delivery_just_completed, over_str

    def _advance_delivery(self):
        """Advances ball counter."""
        self.total_deliveries += 1
        self.balls_in_current_over += 1

    def force_complete_if_active(self) -> bool:
        """Forces completion at video end if a delivery was in progress."""
        if self.state == DeliveryState.IN_FLIGHT and self.active_delivery_frames >= 4:
            self._advance_delivery()
            self.state = DeliveryState.IDLE
            return True
        return False
