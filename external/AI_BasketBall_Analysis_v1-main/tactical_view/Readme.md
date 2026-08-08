# 🎯 Tactical View Converter & Homography Estimation 

This repository contains utilities to estimate and refine homographies between image points and court coordinates, specifically designed for tactical view conversion in sports analytics.


## Features:
- Robust homography estimation using RANSAC with outlier rejection 
- Automatic detection and correction of horizontal flips in homography 
- Reprojection error checking for quality control 
- Temporal smoothing by blending with previous homography estimates 
- Keypoint order optimization for consistent tracking 


## Usage 🛠:
```python
# Initialize with matched points (image_points, court_points)
homography = Homography(image_points, court_points, prev_H=prev_homography, prev_image_points=prev_points)

# Map image points to court coordinates
court_coords = homography.transform_points(image_points_to_transform)
