class CourtVisionPassDetector:
    """
    Detects passes (same-team ball transfers) and interceptions (cross-team steals)
    based on ball possession changes and team assignments.
    Adapted from Repo 1's PassAndInterceptionDetector.
    """
    
    def __init__(self):
        pass
    
    def detect_passes(self, ball_possession: list, player_tracks: list) -> list:
        """
        Detect passes: ball changes holder between players on the SAME team.
        
        Args:
            ball_possession: List of player IDs per frame (-1 = no possession)
            player_tracks: List of dicts per frame, each containing {player_id: {'team': int, ...}}
            
        Returns:
            List of ints per frame: 0=no pass, 1=team 1 pass, 2=team 2 pass
        """
        passes = [0] * len(ball_possession)
        
        for frame in range(1, len(ball_possession)):
            prev_holder = ball_possession[frame - 1]
            curr_holder = ball_possession[frame]
            
            # Skip if no possession or same player
            if prev_holder == -1 or curr_holder == -1 or prev_holder == curr_holder:
                continue
            
            # Get team assignments
            prev_data = player_tracks[frame - 1].get(prev_holder, {})
            curr_data = player_tracks[frame].get(curr_holder, {})
            prev_team = prev_data.get('team', -1)
            curr_team = curr_data.get('team', -1)
            
            # Pass = same team
            if prev_team != -1 and curr_team != -1 and prev_team == curr_team:
                passes[frame] = curr_team
        
        return passes
    
    def detect_interceptions(self, ball_possession: list, player_tracks: list) -> list:
        """
        Detect interceptions: ball changes from one team to another.
        
        Args:
            ball_possession: List of player IDs per frame (-1 = no possession)
            player_tracks: List of dicts per frame, each containing {player_id: {'team': int, ...}}
            
        Returns:
            List of ints per frame: 0=no interception, 1=team 1 intercept, 2=team 2 intercept
        """
        interceptions = [0] * len(ball_possession)
        
        prev_holder = -1
        prev_team = -1
        
        for frame in range(1, len(ball_possession)):
            curr_holder = ball_possession[frame]
            curr_data = player_tracks[frame].get(curr_holder, {})
            curr_team = curr_data.get('team', -1)
            
            # Interception = different team
            if (prev_holder != -1 and curr_holder != -1 and curr_holder != prev_holder and
                prev_team != -1 and curr_team != -1 and curr_team != prev_team):
                interceptions[frame] = curr_team
            
            # Update previous state
            if curr_holder != -1:
                prev_holder = curr_holder
                prev_team = curr_team
        
        return interceptions
