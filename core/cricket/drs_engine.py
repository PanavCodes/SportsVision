import numpy as np
import cv2
import os
from enum import Enum
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

class BatsmanHand(Enum):
    RIGHT_HAND = "RHB"
    LEFT_HAND = "LHB"

class OnFieldCall(Enum):
    OUT = "OUT"
    NOT_OUT = "NOT OUT"

class PitchingZone(Enum):
    IN_LINE = "IN-LINE"
    OUTSIDE_LEG = "OUTSIDE LEG"
    OUTSIDE_OFF = "OUTSIDE OFF"
    FULL_TOSS = "FULL TOSS"

class ImpactZone(Enum):
    IN_LINE = "IN-LINE"
    UMPIRES_CALL = "UMPIRE'S CALL"
    OUTSIDE_OFF = "OUTSIDE OFF"
    OUTSIDE_LEG = "OUTSIDE LEG"

class WicketsResult(Enum):
    HITTING = "HITTING"
    UMPIRES_CALL = "UMPIRE'S CALL"
    MISSING = "MISSING"

@dataclass
class DRSVerdict:
    pitching: PitchingZone
    pitching_coord: Optional[Tuple[float, float, float]]
    impact: ImpactZone
    impact_coord: Tuple[float, float, float]
    wickets: WicketsResult
    stump_coord: Tuple[float, float, float]
    on_field_call: OnFieldCall
    final_verdict: str
    reasons: str
    shot_offered: bool = True
    bat_edge_detected: bool = False
    is_3_meter_rule_triggered: bool = False
    is_close_proximity_bounce: bool = False
    seam_deviation_deg: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'pitching': self.pitching.value,
            'pitching_coord': self.pitching_coord,
            'impact': self.impact.value,
            'impact_coord': self.impact_coord,
            'wickets': self.wickets.value,
            'stump_coord': self.stump_coord,
            'on_field_call': self.on_field_call.value,
            'final_verdict': self.final_verdict,
            'reasons': self.reasons,
            'shot_offered': self.shot_offered,
            'bat_edge_detected': self.bat_edge_detected,
            'is_3_meter_rule': self.is_3_meter_rule_triggered,
            'is_close_proximity': self.is_close_proximity_bounce,
            'seam_deviation_deg': round(self.seam_deviation_deg, 1)
        }

class DRSEngine:
    """
    Official ICC Decision Review System (DRS) & LBW Adjudication Engine.
    
    Implements Laws of Cricket Law 36 (LBW) & ICC DRS Playing Conditions:
      - Pitching Zone (In-Line, Outside Leg, Outside Off, Full Toss)
      - Impact Zone with Shot-Offered Law (Law 36.1(e)) & Umpire's Call margins
      - Wicket Zone (Hitting, Umpire's Call, Missing) at Y = 20.12m
      - ICC 3-Meter Distance-From-Stumps Regulation (20.12 - Y_pad >= 3.0m)
      - Close Proximity Pitching Rule (Delta Y < 0.40m)
      - Bat Edge / UltraEdge Deflection check
    """
    def __init__(
        self,
        stump_height: float = 0.7112,
        stump_width: float = 0.2286,
        bails_height: float = 0.74,
        ball_radius: float = 0.036,
        pitch_length: float = 20.12
    ):
        self.stump_height = stump_height
        self.stump_width = stump_width
        self.stump_half_width = stump_width / 2.0  # 0.1143m (~4.5 inches each side of middle)
        self.bails_height = bails_height
        self.ball_radius = ball_radius
        self.pitch_length = pitch_length

    def evaluate_lbw(
        self,
        bounce_point: Optional[Tuple[float, float, float]],
        pad_impact_point: Tuple[float, float, float],
        predicted_stump_point: Tuple[float, float, float],
        batsman_hand: BatsmanHand = BatsmanHand.RIGHT_HAND,
        on_field_call: OnFieldCall = OnFieldCall.NOT_OUT,
        shot_offered: bool = True,
        bat_edge_detected: bool = False,
        seam_deviation_deg: float = 0.0
    ) -> DRSVerdict:
        """
        Executes complete LBW adjudication against official ICC playing conditions.
        Coordinates are assumed to be in pitch metric meters:
          X: lateral offset from pitch center (-1.52m to +1.52m)
          Y: length along pitch (0m bowler stumps to 20.12m batsman stumps)
          Z: vertical height off ground plane (0m to 2.5m)
        """
        r = self.ball_radius
        stump_edge = self.stump_half_width
        stump_h = self.stump_height
        bail_h = self.bails_height
        stump_y = self.pitch_length

        ix, iy, iz = pad_impact_point
        sx, sy, sz = predicted_stump_point

        # 1. BAT EDGE CHECK (UltraEdge / Snickometer)
        if bat_edge_detected:
            return DRSVerdict(
                pitching=PitchingZone.IN_LINE if bounce_point else PitchingZone.FULL_TOSS,
                pitching_coord=bounce_point,
                impact=ImpactZone.IN_LINE,
                impact_coord=pad_impact_point,
                wickets=WicketsResult.HITTING,
                stump_coord=predicted_stump_point,
                on_field_call=on_field_call,
                final_verdict="NOT OUT",
                reasons="Bat edge detected prior to pad contact (UltraEdge / Snicko). LBW invalidated.",
                shot_offered=shot_offered,
                bat_edge_detected=True,
                seam_deviation_deg=seam_deviation_deg
            )

        # 2. PITCHING EVALUATION
        is_close_proximity = False
        if bounce_point is None:
            pitching = PitchingZone.FULL_TOSS
        else:
            px, py, pz = bounce_point
            if (iy - py) < 0.40:
                is_close_proximity = True

            if batsman_hand == BatsmanHand.RIGHT_HAND:
                if px > stump_edge:
                    pitching = PitchingZone.OUTSIDE_LEG
                elif px < -stump_edge:
                    pitching = PitchingZone.OUTSIDE_OFF
                else:
                    pitching = PitchingZone.IN_LINE
            else:  # LEFT_HAND
                if px < -stump_edge:
                    pitching = PitchingZone.OUTSIDE_LEG
                elif px > stump_edge:
                    pitching = PitchingZone.OUTSIDE_OFF
                else:
                    pitching = PitchingZone.IN_LINE

        # 3. IMPACT EVALUATION (Law 36.1(e))
        abs_ix = abs(ix)
        if abs_ix <= stump_edge:
            impact = ImpactZone.IN_LINE
        elif abs_ix <= (stump_edge + r):
            impact = ImpactZone.UMPIRES_CALL
        else:
            if batsman_hand == BatsmanHand.RIGHT_HAND:
                impact = ImpactZone.OUTSIDE_OFF if ix < 0 else ImpactZone.OUTSIDE_LEG
            else:
                impact = ImpactZone.OUTSIDE_OFF if ix > 0 else ImpactZone.OUTSIDE_LEG

        # 4. WICKETS EVALUATION (at Y = 20.12m)
        abs_sx = abs(sx)
        # Direct Hit: Center inside outer stump boundary and height between ground and bails
        if (abs_sx <= stump_edge) and (r <= sz <= stump_h):
            wickets = WicketsResult.HITTING
        # Umpire's Call clipping: Ball center within 1 ball-radius margin of stumps/bails
        elif (abs_sx <= stump_edge + r) and (0.0 <= sz <= bail_h + r):
            wickets = WicketsResult.UMPIRES_CALL
        else:
            wickets = WicketsResult.MISSING

        # 5. ICC 3-METER DISTANCE-FROM-STUMPS REGULATION
        dist_to_stumps = max(0.0, stump_y - iy)
        is_3_meter_rule = (dist_to_stumps >= 3.0)

        # 6. FINAL ADJUDICATION RESOLUTION
        # Law 1: Pitching outside leg is NEVER out
        if pitching == PitchingZone.OUTSIDE_LEG:
            final_verdict = "NOT OUT"
            reasons = "Pitching outside leg stump line (Law 36.1(b))."
        # Law 2: Impact outside off is NOT OUT only if shot offered (Law 36.1(e))
        elif impact == ImpactZone.OUTSIDE_LEG:
            final_verdict = "NOT OUT"
            reasons = "Impact outside leg stump line."
        elif impact == ImpactZone.OUTSIDE_OFF and shot_offered:
            final_verdict = "NOT OUT"
            reasons = "Impact outside off stump line with shot offered (Law 36.1(e))."
        # Law 3: Missing wickets is NEVER out
        elif wickets == WicketsResult.MISSING:
            final_verdict = "NOT OUT"
            reasons = "Projected ball trajectory missing wickets."
        else:
            # Check Close Proximity Rule (<40cm between pitch and pad)
            if is_close_proximity and on_field_call == OnFieldCall.NOT_OUT:
                final_verdict = "NOT OUT"
                reasons = "Impact within 40cm of pitch bounce. On-field call upheld."
            # Check ICC 3-Meter Rule
            elif is_3_meter_rule and on_field_call == OnFieldCall.NOT_OUT:
                if (wickets == WicketsResult.UMPIRES_CALL) or (abs_sx > (stump_edge * 0.75)):
                    final_verdict = "NOT OUT"
                    reasons = f"ICC 3-Meter Law applies ({dist_to_stumps:.2f}m from stumps). On-field NOT OUT upheld."
                elif impact == ImpactZone.UMPIRES_CALL:
                    final_verdict = "NOT OUT"
                    reasons = f"ICC 3-Meter Law applies with Umpire's Call on impact. On-field NOT OUT upheld."
                else:
                    final_verdict = "OUT"
                    reasons = "Three Reds despite 3m distance (ball crashing into middle stump)."
            else:
                # Standard resolution
                if (impact == ImpactZone.UMPIRES_CALL) or (wickets == WicketsResult.UMPIRES_CALL):
                    final_verdict = on_field_call.value
                    reasons = f"Umpire's Call on {'Impact' if impact == ImpactZone.UMPIRES_CALL else 'Wickets'}. On-field verdict upheld."
                else:
                    final_verdict = "OUT"
                    reasons = "Three Reds: Pitching in-line, Impact in-line, Wickets hitting."

        return DRSVerdict(
            pitching=pitching,
            pitching_coord=bounce_point,
            impact=impact,
            impact_coord=pad_impact_point,
            wickets=wickets,
            stump_coord=predicted_stump_point,
            on_field_call=on_field_call,
            final_verdict=final_verdict,
            reasons=reasons,
            shot_offered=shot_offered,
            bat_edge_detected=bat_edge_detected,
            is_3_meter_rule_triggered=is_3_meter_rule,
            is_close_proximity_bounce=is_close_proximity,
            seam_deviation_deg=seam_deviation_deg
        )

    def render_drs_banner(
        self,
        verdict: DRSVerdict,
        width: int = 860,
        height: int = 240
    ) -> np.ndarray:
        """
        Renders a broadcast-quality DRS adjudication card (PITCHING | IMPACT | WICKETS).
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (20, 24, 33)  # Sleek dark navy slate

        # Header bar
        cv2.rectangle(img, (0, 0), (width, 42), (32, 41, 56), -1)
        cv2.putText(img, "DECISION REVIEW SYSTEM (DRS) | HAWK-EYE", (24, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"ON-FIELD: {verdict.on_field_call.value}", (width - 240, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (148, 163, 184), 1, cv2.LINE_AA)

        # 3 Adjudication Boxes
        boxes = [
            ("PITCHING", verdict.pitching.value, 30),
            ("IMPACT", verdict.impact.value, 300),
            ("WICKETS", verdict.wickets.value, 570)
        ]

        def get_box_color(text: str):
            if "IN-LINE" in text or "HITTING" in text:
                return (34, 197, 94)    # Green (Hitting/Red zone in DRS terms)
            elif "UMPIRE" in text:
                return (234, 179, 8)    # Amber
            else:
                return (59, 130, 246)   # Blue / Slate (Missing/Outside)

        for title, val, x_start in boxes:
            box_w, box_h = 250, 95
            y_start = 55
            color = get_box_color(val)
            cv2.rectangle(img, (x_start, y_start), (x_start + box_w, y_start + box_h), (28, 35, 48), -1)
            cv2.rectangle(img, (x_start, y_start), (x_start + box_w, y_start + box_h), color, 2)
            cv2.putText(img, title, (x_start + 16, y_start + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (203, 213, 225), 1, cv2.LINE_AA)
            cv2.putText(img, val, (x_start + 16, y_start + 72),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, color, 2, cv2.LINE_AA)

        # Footer Verdict Bar
        v_color = (34, 197, 94) if verdict.final_verdict == "OUT" else (59, 130, 246)
        cv2.rectangle(img, (30, 165), (width - 30, 220), (28, 35, 48), -1)
        cv2.rectangle(img, (30, 165), (width - 30, 220), v_color, 2)

        verdict_text = f"DECISION: {verdict.final_verdict}"
        cv2.putText(img, verdict_text, (50, 203), cv2.FONT_HERSHEY_DUPLEX, 0.95, v_color, 2, cv2.LINE_AA)
        cv2.putText(img, verdict.reasons[:65], (360, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return img
