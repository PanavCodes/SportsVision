import numpy as np
from typing import Dict, Optional, List, Tuple
from core.cricket.drs_engine import DRSEngine, DRSVerdict, OnFieldCall
from core.cricket.umpiring import UltraEdgeWaveformSimulator, UltraEdgeType

class CricketEventDetector:
    """
    Detects key match events from ball trajectories, player interactions, and pitch zones:
      - Release Event: delivery release from bowler hand
      - Pitch Bounce Event: landing point on the pitch
      - Bat Contact Event: impact moment with bat
      - Pad Impact Event: ball striking batsman pads for LBW adjudication
      - Hawk-Eye DRS Review: full ICC Law 36 adjudication (Pitching, Impact, Wickets)
      - Outcome Event: DOT, RUNS, FOUR (boundary), SIX (maximum), WICKET (bowled/LBW/catch)
    """
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.drs_engine = DRSEngine()
        self.snicko_simulator = UltraEdgeWaveformSimulator()

    def analyze_delivery(
        self,
        ball_tracks: List[dict],
        player_tracks: List[dict],
        bounce_info: Optional[dict] = None,
        stumps_info: Optional[List[dict]] = None,
        shot_info: Optional[dict] = None,
        pad_impact_metric: Optional[Tuple[float, float, float]] = None,
        predicted_stump_metric: Optional[Tuple[float, float, float]] = None
    ) -> Dict:
        """
        Analyze an entire delivery sequence to detect events, DRS LBW review, and match outcome.
        """
        outcome = "dot_ball"
        outcome_confidence = 0.8
        is_wicket = False
        is_boundary = False
        boundary_type = None
        impact_frame = None
        is_pad_impact = False
        drs_verdict = None

        if not ball_tracks:
            return {
                'outcome': outcome,
                'is_wicket': False,
                'is_boundary': False,
                'boundary_type': None,
                'impact_frame': None,
                'drs_verdict': None
            }

        # 1. Identify batsman, pads & stumps positions
        batsman_box = None
        batsman_pads = None
        batsman_end_stumps = None

        for frame_tracks in player_tracks:
            for pid, info in frame_tracks.items():
                if info.get('role') == 'batsman':
                    batsman_box = info['bbox']
                    bx1, by1, bx2, by2 = batsman_box
                    # Batsman pads are lower 45% of batsman bounding box
                    batsman_pads = (bx1, by1 + (by2 - by1) * 0.55, bx2, by2)
                    break
            if batsman_box is not None:
                break

        if stumps_info and len(stumps_info) > 0:
            first_stumps = stumps_info[0]
            if isinstance(first_stumps, dict):
                batsman_end_stumps = first_stumps.get('batsman_end')

        # 2. Check for Ball Centroids
        centroids = []
        for f_idx, b_track in enumerate(ball_tracks):
            if 1 in b_track:
                centroids.append((f_idx, b_track[1]['centroid']))

        # 3. Check for Bat Impact vs Pad Impact
        bat_impact_detected = False
        if len(centroids) >= 4 and batsman_box is not None:
            bx1, by1, bx2, by2 = batsman_box
            for i in range(1, len(centroids) - 2):
                f_idx, pt = centroids[i]
                # Ball near batsman zone
                if (bx1 - 50 <= pt[0] <= bx2 + 50) and (by1 - 30 <= pt[1] <= by2 + 30):
                    dx1 = pt[0] - centroids[i-1][1][0]
                    dy1 = pt[1] - centroids[i-1][1][1]
                    dx2 = centroids[i+1][1][0] - pt[0]
                    dy2 = centroids[i+1][1][1] - pt[1]

                    dot_prod = dx1 * dx2 + dy1 * dy2
                    if dot_prod < 0 or (abs(dx2 - dx1) > 20 and abs(dy2 - dy1) > 20):
                        impact_frame = f_idx
                        bat_impact_detected = True
                        break

            # If no bat deflection, check if ball struck the batsman pads
            if not bat_impact_detected and batsman_pads is not None:
                px1, py1, px2, py2 = batsman_pads
                for i in range(1, len(centroids)):
                    f_idx, pt = centroids[i]
                    if (px1 - 20 <= pt[0] <= px2 + 20) and (py1 - 20 <= pt[1] <= py2 + 20):
                        is_pad_impact = True
                        impact_frame = f_idx
                        break

        # 4. Check for Wicket (Bowled): ball terminates directly on stumps
        if batsman_end_stumps is not None and len(centroids) >= 3:
            sx, sy = batsman_end_stumps
            last_ball_pts = centroids[-3:]
            for _, pt in last_ball_pts:
                dist_to_stumps = np.hypot(pt[0] - sx, pt[1] - sy)
                if dist_to_stumps < 40:
                    is_wicket = True
                    outcome = "wicket"
                    outcome_confidence = 0.94
                    break

        # 5. Check for LBW / Hawk-Eye Review if pad impact occurred
        if is_pad_impact and not is_wicket:
            bounce_coord = bounce_info.get('coord_3d') if bounce_info else None
            pad_coord = pad_impact_metric or (0.02, 18.0, 0.45)
            stump_coord = predicted_stump_metric or (0.01, 20.12, 0.55)

            # Check acoustic snicko simulation
            snicko_res = self.snicko_simulator.generate_waveform(
                contact_type=UltraEdgeType.PAD_CONTACT,
                contact_frame=impact_frame or 30
            )

            drs_res = self.drs_engine.evaluate_lbw(
                bounce_point=bounce_coord,
                pad_impact_point=pad_coord,
                predicted_stump_point=stump_coord,
                on_field_call=OnFieldCall.OUT
            )
            drs_verdict = drs_res.to_dict()

            if drs_res.final_verdict == "OUT":
                is_wicket = True
                outcome = "lbw_wicket"
                outcome_confidence = 0.90
            else:
                outcome = "lbw_not_out"
                outcome_confidence = 0.85

        # 6. Check for Boundaries (Four or Six)
        if not is_wicket and bat_impact_detected and impact_frame is not None:
            shot_label = shot_info.get('shot_label', '') if shot_info else ''
            post_impact_pts = [pt for f, pt in centroids if f >= impact_frame]

            if len(post_impact_pts) >= 3:
                total_exit_dist = np.hypot(
                    post_impact_pts[-1][0] - post_impact_pts[0][0],
                    post_impact_pts[-1][1] - post_impact_pts[0][1]
                )
                if total_exit_dist > 180 or shot_label in ['loft_shot', 'pull_shot']:
                    is_boundary = True
                    if shot_label == 'loft_shot' and post_impact_pts[-1][1] < post_impact_pts[0][1]:
                        boundary_type = "six"
                        outcome = "six"
                    else:
                        boundary_type = "four"
                        outcome = "four"
                else:
                    outcome = "runs"

        return {
            'outcome': outcome,
            'is_wicket': is_wicket,
            'is_boundary': is_boundary,
            'boundary_type': boundary_type,
            'impact_frame': impact_frame,
            'is_pad_impact': is_pad_impact,
            'drs_verdict': drs_verdict
        }
