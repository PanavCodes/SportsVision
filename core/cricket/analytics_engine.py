import numpy as np
from typing import Tuple, Dict, Optional

class ExpectedDismissalEngine:
    """
    Computes Expected Dismissal (xD) probability score (0.00 to 1.00 / 0-100%)
    for a cricket delivery based on kinematic line, length, deviation, and wicket-zone intersection.
    """
    def __init__(
        self,
        stump_half_width: float = 0.1143,
        stump_height: float = 0.7112,
        pitch_length: float = 20.12
    ):
        self.stump_half_width = stump_half_width
        self.stump_height = stump_height
        self.pitch_length = pitch_length

    def classify_length_zone(self, bounce_y: float) -> str:
        """Categorizes pitch bounce distance along pitch (0 to 20.12m)."""
        if bounce_y < 8.0:
            return "Bouncer"
        elif bounce_y < 10.5:
            return "Short / Back of Length"
        elif bounce_y <= 14.5:
            return "Good Length"
        elif bounce_y <= 17.5:
            return "Full / Driving Length"
        else:
            return "Yorker"

    def classify_line_zone(self, bounce_x: float, is_rhb: bool = True) -> str:
        """Categorizes lateral offset relative to batsman off/leg side."""
        sign = 1.0 if is_rhb else -1.0
        x_rel = bounce_x * sign
        stump_edge = self.stump_half_width

        if x_rel < -stump_edge * 2.5:
            return "Wide Outside Off"
        elif x_rel < -stump_edge:
            return "Corridor of Uncertainty / 4th Stump"
        elif abs(x_rel) <= stump_edge:
            return "On The Stumps"
        else:
            return "Down Leg Side"

    def calculate_xd(
        self,
        bounce_point: Optional[Tuple[float, float, float]],
        pad_point: Optional[Tuple[float, float, float]] = None,
        stump_point: Optional[Tuple[float, float, float]] = None,
        speed_kmh: float = 135.0,
        seam_deviation_deg: float = 0.0,
        is_rhb: bool = True
    ) -> Dict:
        """
        Calculates xD (Expected Dismissal) score and feature breakdown.
        xD represents the statistical probability of a wicket occurring on this delivery.
        """
        if bounce_point is None:
            # Full toss or unpitched delivery
            return {
                'xd_score': 0.15,
                'xd_pct': 15.0,
                'threat_level': 'LOW',
                'length_zone': 'Full Toss',
                'line_zone': 'On The Stumps'
            }

        bx, by, bz = bounce_point
        length_zone = self.classify_length_zone(by)
        line_zone = self.classify_line_zone(bx, is_rhb=is_rhb)

        # Baseline probability by length zone
        length_weights = {
            "Good Length": 0.42,
            "Yorker": 0.38,
            "Full / Driving Length": 0.28,
            "Short / Back of Length": 0.22,
            "Bouncer": 0.18
        }
        base_xd = length_weights.get(length_zone, 0.25)

        # Line multiplier
        if line_zone == "Corridor of Uncertainty / 4th Stump":
            line_mult = 1.35  # Edges behind to keeper/slips
        elif line_zone == "On The Stumps":
            line_mult = 1.25  # Bowled / LBW threat
        elif line_zone == "Wide Outside Off":
            line_mult = 0.50
        else:
            line_mult = 0.40  # Down leg

        # Speed multiplier (>140km/h creates higher dismissal pressure)
        speed_norm = np.clip((speed_kmh - 90.0) / 60.0, 0.0, 1.0)
        speed_mult = 0.85 + 0.35 * speed_norm

        # Seam/Spin movement bonus
        movement_bonus = min(0.20, abs(seam_deviation_deg) * 0.05)

        # Stumps intersection bonus if predicted path hits the stumps
        wicket_bonus = 0.0
        if stump_point is not None:
            sx, sy, sz = stump_point
            if abs(sx) <= self.stump_half_width and 0.0 <= sz <= self.stump_height:
                wicket_bonus = 0.18

        raw_xd = (base_xd * line_mult * speed_mult) + movement_bonus + wicket_bonus
        xd_score = float(np.clip(raw_xd, 0.02, 0.95))
        xd_pct = round(xd_score * 100.0, 1)

        if xd_score >= 0.65:
            threat = "EXTREME"
        elif xd_score >= 0.45:
            threat = "HIGH"
        elif xd_score >= 0.25:
            threat = "MEDIUM"
        else:
            threat = "LOW"

        return {
            'xd_score': round(xd_score, 3),
            'xd_pct': xd_pct,
            'threat_level': threat,
            'length_zone': length_zone,
            'line_zone': line_zone,
            'speed_factor': round(speed_mult, 2),
            'movement_bonus': round(movement_bonus, 2)
        }
