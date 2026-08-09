import cv2
import numpy as np
from sklearn.cluster import KMeans

import config


class CourtVisionTeamClassifier:
    """
    Dynamic Team Classifier.
    - Automatically discovers the two main jersey colors on the court using K-Means clustering 
      on the first few frames.
    - Classifies players by finding the closest team color via BGR Euclidean distance.
    - Naturally supports all colors including white, black, and gray.
    """
    def __init__(self):
        # Discovered team colors
        self.team_1_bgr = None
        self.team_2_bgr = None
        self.is_initialized = False
        self.pixel_samples = []
        self.max_samples = 15000  # Number of pixels to sample before discovering colors
        
        # Track history for temporal stability
        self.track_history = {}
        self.check_interval = 5  # Run color check every N frames per track
        self.frame_count = 0
        
        # Phase 2: SigLIP + UMAP
        self.use_siglip = getattr(config, 'USE_SIGLIP', False)
        if self.use_siglip:
            from transformers import SiglipImageProcessor, SiglipVisionModel
            import torch
            import umap
            
            print(f"Loading SigLIP model on {config.DEVICE}...")
            self.siglip_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.siglip_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(config.DEVICE)
            self.siglip_model.eval()
            
            self.umap_reducer = umap.UMAP(n_components=2, random_state=42)
            self.cluster_model = KMeans(n_clusters=2, n_init=5, random_state=42)
            
            self.embedding_samples = []  # Store (player_id, embedding)
            self.embedding_colors = []   # Store median colors for visualization fallback
            self.max_embedding_samples = 200 # Need around 200 player crops to cluster teams

        # Referee IDs from detector (updated each batch)
        self.referee_ids = set()

    def _discover_team_colors(self):
        """Runs K-Means to discover the two team colors."""
        if not self.pixel_samples:
            return
            
        pixels = np.vstack(self.pixel_samples)
        if len(pixels) > 20000:
            pixels = pixels[np.random.choice(len(pixels), 20000, replace=False)]
            
        # We look for 2 clusters: Team 1 and Team 2 (court/skin is masked out)
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42).fit(pixels)
        colors = kmeans.cluster_centers_
        
        # Assign colors
        self.team_1_bgr = colors[0]
        self.team_2_bgr = colors[1]
        self.is_initialized = True
        print(f"\n[Team Classifier] Dynamic Team Colors Discovered!")
        print(f"  Team 1 BGR: {self.team_1_bgr.astype(int)}")
        print(f"  Team 2 BGR: {self.team_2_bgr.astype(int)}\n")

    def _discover_teams_siglip(self):
        """Fits UMAP and KMeans on the collected SigLIP embeddings."""
        if not self.embedding_samples:
            return
            
        embeddings = np.vstack(self.embedding_samples)
        
        # Fit UMAP to reduce to 2D
        print("\n[Team Classifier] Fitting UMAP on SigLIP embeddings...")
        emb_2d = self.umap_reducer.fit_transform(embeddings)
        
        # Fit KMeans to cluster the 2D embeddings into 2 teams
        self.cluster_model.fit(emb_2d)
        
        # Determine average colors for HUD
        labels = self.cluster_model.labels_
        colors = np.vstack(self.embedding_colors)
        
        t1_colors = colors[labels == 0]
        t2_colors = colors[labels == 1]
        
        if len(t1_colors) > 0:
            self.team_1_bgr = np.median(t1_colors, axis=0)
        else:
            self.team_1_bgr = np.array([255, 0, 0])
            
        if len(t2_colors) > 0:
            self.team_2_bgr = np.median(t2_colors, axis=0)
        else:
            self.team_2_bgr = np.array([0, 255, 0])
            
        self.is_initialized = True
        print(f"[Team Classifier] SigLIP Teams Discovered!")
        print(f"  Team 1 Avg BGR: {self.team_1_bgr.astype(int)}")
        print(f"  Team 2 Avg BGR: {self.team_2_bgr.astype(int)}\n")

    def classify_players(self, video_frames: list, player_tracks: list, referee_ids: set = None):
        """
        Classifies each player into a team based on Euclidean distance to dynamic team colors.
        """
        if referee_ids is not None:
            self.referee_ids = referee_ids
            
        # First pass: collect pixels for initialization if needed
        if not self.is_initialized:
            for frame_idx, frame in enumerate(video_frames):
                tracks = player_tracks[frame_idx]
                for player_id, data in tracks.items():
                    if player_id in self.referee_ids: continue
                    
                    bbox = data['bbox']
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    crop_h = y2 - y1
                    crop_w = x2 - x1
                    
                    # Skip tiny detections (likely crowd members in the background)
                    # For a 720p/1080p video, a player on the court is usually > 50px tall
                    if crop_h < 50 or crop_w < 15:
                        continue
                        
                    # Just take the very center pixels (e.g. chest patch) to avoid
                    # issues with FIBA courts (light wood/grey) not being filtered by HSV.
                    center_y = int(y1 + crop_h * 0.3)
                    center_x = int(x1 + crop_w * 0.5)
                    
                    
                    # Ensure patch is within bounds and dynamically sized based on bounding box
                    radius_y = max(2, int(crop_h * 0.1))
                    radius_x = max(2, int(crop_w * 0.2))
                    
                    py1 = max(0, center_y - radius_y)
                    py2 = min(frame.shape[0], center_y + radius_y)
                    px1 = max(0, center_x - radius_x)
                    px2 = min(frame.shape[1], center_x + radius_x)
                    
                    if py2 > py1 and px2 > px1:
                        if self.use_siglip:
                            import torch
                            # Use full player crop for SigLIP
                            crop = frame[y1:y2, x1:x2].copy()
                            
                            # Apply SAM2 mask if available to blackout background
                            if 'mask' in data:
                                m = data['mask'][y1:y2, x1:x2]
                                if m.shape == crop.shape[:2]:
                                    crop[m == 0] = 0
                                    
                            # Get SigLIP embedding
                            inputs = self.siglip_processor(images=crop, return_tensors="pt").to(config.DEVICE)
                            with torch.no_grad():
                                outputs = self.siglip_model(**inputs)
                                emb = outputs.pooler_output.cpu().numpy().flatten()
                            
                            # Also get median color for HUD drawing
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            mask_court = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
                            mask_valid = cv2.bitwise_not(mask_court)
                            valid_pixels = crop[mask_valid == 255]
                            med_color = np.median(valid_pixels, axis=0) if len(valid_pixels) > 0 else np.array([128,128,128])
                                
                            self.embedding_samples.append(emb)
                            self.embedding_colors.append(med_color)
                        else:
                            crop = frame[py1:py2, px1:px2]
                            valid_pixels = crop.reshape(-1, 3)
                            self.pixel_samples.extend(valid_pixels.tolist())
                            
                if self.use_siglip:
                    if len(self.embedding_samples) >= self.max_embedding_samples and not self.is_initialized:
                        self._discover_teams_siglip()
                        break
                else:
                    if len(self.pixel_samples) >= self.max_samples and not self.is_initialized:
                        self._discover_team_colors()
                        break # Stop collecting once we have enough

        # If STILL not initialized (video too short?), force initialization
        if not self.is_initialized and self.pixel_samples:
            self._discover_team_colors()
            
        # If absolutely no players, just return
        if not self.is_initialized:
            return player_tracks

        # Second pass: classify
        for frame_idx, frame in enumerate(video_frames):
            self.frame_count += 1
            tracks = player_tracks[frame_idx]
            
            for player_id, data in tracks.items():
                bbox = data['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # If this player is a referee, auto-assign team 0 and skip
                if player_id in self.referee_ids:
                    data['team'] = 0
                    if player_id not in self.track_history:
                        self.track_history[player_id] = {
                            'team_votes': {1: 0, 2: 0},
                            'last_update_frame': 0,
                            'current_team': 0,
                            'is_referee': True
                        }
                    self.track_history[player_id]['is_referee'] = True
                    self.track_history[player_id]['current_team'] = 0
                    continue
                
                # Retrieve or initialize history for this track
                if player_id not in self.track_history:
                    self.track_history[player_id] = {
                        'team_votes': {1: 0, 2: 0},
                        'last_update_frame': -self.check_interval,
                        'current_team': 0,
                        'is_referee': False
                    }
                
                history = self.track_history[player_id]
                
                # Skip if already confirmed as referee
                if history.get('is_referee', False):
                    data['team'] = 0
                    continue
                
                # Only perform extraction if it's a new track or interval elapsed
                if self.frame_count - history['last_update_frame'] >= self.check_interval:
                    history['last_update_frame'] = self.frame_count
                    
                    if self.use_siglip:
                        import torch
                        crop = frame[y1:y2, x1:x2].copy()
                        if 'mask' in data:
                            m = data['mask'][y1:y2, x1:x2]
                            if m.shape == crop.shape[:2]:
                                crop[m == 0] = 0
                                
                        inputs = self.siglip_processor(images=crop, return_tensors="pt").to(config.DEVICE)
                        with torch.no_grad():
                            emb = self.siglip_model(**inputs).pooler_output.cpu().numpy().flatten()
                            
                        # Transform embedding and predict
                        emb_2d = self.umap_reducer.transform([emb])
                        cluster_idx = self.cluster_model.predict(emb_2d)[0]
                        # cluster 0 -> team 1, cluster 1 -> team 2
                        team_id = 1 if cluster_idx == 0 else 2
                        history['team_votes'][team_id] += 1
                        
                    else:
                        crop_h = y2 - y1
                        crop_w = x2 - x1
                        cy1, cy2 = int(y1 + crop_h*0.1), int(y1 + crop_h*0.5)
                        cx1, cx2 = int(x1 + crop_w*0.2), int(x2 - crop_w*0.2)
                        
                        crop = frame[max(0, cy1):min(frame.shape[0], cy2), max(0, cx1):min(frame.shape[1], cx2)]
                        
                        if crop.size > 0:
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            mask_court = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
                            mask_valid = cv2.bitwise_not(mask_court)
                            
                            valid_pixels = crop[mask_valid == 255]
                            if len(valid_pixels) > 0:
                                median_color = np.median(valid_pixels, axis=0)
                                
                                # Euclidean distance in BGR
                                dist_1 = np.linalg.norm(median_color - self.team_1_bgr)
                                dist_2 = np.linalg.norm(median_color - self.team_2_bgr)
                                
                                if dist_1 < dist_2:
                                    history['team_votes'][1] += 1
                                else:
                                    history['team_votes'][2] += 1

                # Determine current team via voting
                t1_votes = history['team_votes'].get(1, 0)
                t2_votes = history['team_votes'].get(2, 0)
                
                if t1_votes + t2_votes == 0:
                    best_team = 1
                elif t1_votes >= t2_votes:
                    best_team = 1
                else:
                    best_team = 2
                
                history['current_team'] = best_team
                data['team'] = best_team
                    
        return player_tracks
