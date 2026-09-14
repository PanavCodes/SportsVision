import random
from typing import Dict, Optional

class CricketCommentaryGenerator:
    """
    Automated Ball-by-Ball Match Commentary & Narrative Generator.
    
    Inspired by broadcast commentary transcripts (Psychellic/Analysis-of-Cricket-Commentary).
    Transforms spatial telemetry and delivery metrics into realistic, expressive
    cricket broadcast commentary.
    """
    def __init__(self):
        self.bowler_styles = ["Fast Bowler", "Seam Bowler", "Pacer"]

    def generate_delivery_commentary(
        self,
        delivery_number: int,
        speeds_dict: Dict,
        line_length_tuple: tuple,
        shot_tuple: tuple,
        event_dict: Dict,
        xd_info: Optional[Dict] = None,
        over_number: Optional[str] = None
    ) -> str:
        """
        Generate contextual ball-by-ball commentary line.
        """
        speed_kmh = speeds_dict.get('release_speed_kmh', 0.0)
        speed_str = f" ({speed_kmh:.1f} km/h)" if speed_kmh > 0 else ""

        line = line_length_tuple[0] if line_length_tuple else "On the stumps"
        length = line_length_tuple[1] if len(line_length_tuple) > 1 else "Good length"
        shot = shot_tuple[0] if shot_tuple else "defense"
        outcome = event_dict.get('outcome', 'dot_ball')
        is_wicket = event_dict.get('is_wicket', False)
        drs_verdict = event_dict.get('drs_verdict')

        over_prefix = f"Over {over_number}:" if over_number else f"Over 0.{delivery_number}:"

        # 1. Wicket / Dismissal Commentary
        if is_wicket:
            if drs_verdict:
                decision = drs_verdict.get('final_verdict', 'OUT')
                reasons = drs_verdict.get('reasons', '')
                return (
                    f"{over_prefix}{speed_str} HUGE APPEAL for LBW! Pitched {drs_verdict.get('pitching')}, "
                    f"struck pad {drs_verdict.get('impact')}. Hawk-Eye projection confirms {drs_verdict.get('wickets')}! "
                    f"DECISION: {decision}. {reasons}"
                )
            elif outcome == "wicket":
                return (
                    f"{over_prefix}{speed_str} BOWLED HIM! Timber! Pitched on {length.lower()} and crashed "
                    f"straight through the batsman's defense into the stumps! Crucial breakthrough!"
                )
            else:
                return (
                    f"{over_prefix}{speed_str} OUT! High in the air and taken! A magnificent delivery on {length.lower()} "
                    f"forces the false shot."
                )

        # 2. Boundaries (Four or Six)
        if outcome == "six":
            return (
                f"{over_prefix}{speed_str} BANG! MAXIMUM! Picked up cleanly off a {length.lower()} delivery, "
                f"launched high over the ropes for a towering SIX!"
            )
        elif outcome == "four":
            return (
                f"{over_prefix}{speed_str} FOUR! What a gorgeous {shot.replace('_', ' ')}! Fed on {length.lower()} "
                f"{line.lower()}, and crunched through the gap to the boundary fence."
            )

        # 3. LBW Review Not Out
        if outcome == "lbw_not_out" and drs_verdict:
            return (
                f"{over_prefix}{speed_str} Shouted appeal for LBW! Turned down by the umpire. "
                f"Hawk-Eye shows {drs_verdict.get('reasons')}. Not out stands."
            )

        # 4. Runs Scored
        runs = event_dict.get('runs', 0)
        if outcome == "runs" or runs > 0:
            return (
                f"{over_prefix}{speed_str} Pushed gently into the off-side gap with a neat {shot.replace('_', ' ')}. "
                f"The batsmen scamper across for a quick {runs if runs > 1 else 'single'}."
            )

        # 5. Dot Balls & Defensive Plays
        xd_tag = ""
        if xd_info:
            xd_pct = xd_info.get('xd_pct', 0.0)
            if xd_pct >= 40.0:
                xd_tag = f" [Threat Index xD: {xd_pct:.1f}%]"

        dot_templates = [
            f"{over_prefix}{speed_str} Probing delivery on {length.lower()}, {line.lower()}. The batsman plays a measured {shot.replace('_', ' ')} towards mid-on. No run.{xd_tag}",
            f"{over_prefix}{speed_str} Bowled at good pace on {length.lower()}. Batsman defends solidly behind the line of the ball. Dot ball.{xd_tag}",
            f"{over_prefix}{speed_str} Tight bowling! Pitched on {length.lower()}, beaten on the outside edge as it zips past the off pole.{xd_tag}"
        ]
        return random.choice(dot_templates)

    def generate_over_summary(self, over_number: int, runs: int, wickets: int, deliveries_count: int) -> str:
        """
        Generate end-of-over narrative recap.
        """
        w_text = f", {wickets} wicket{'s' if wickets > 1 else ''}" if wickets > 0 else ""
        return (
            f"End of Over {over_number}: {runs} run{'s' if runs != 1 else ''}{w_text} off {deliveries_count} balls. "
            f"{'A testing over by the bowling side.' if runs <= 3 else 'A productive over for the batting side.'}"
        )
