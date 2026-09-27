import os
import sys
import time
import json
import cv2
import torch
import numpy as np
import tqdm
from typing import Optional, List, Tuple, Dict

import config
from core.shared.gpu_diagnostics import verify_cuda
from core.shared.video_io import format_time

from core.cricket.detector import CricketDetector
from core.cricket.ball_tracker import CricketBallTracker
from core.cricket.trajectory import CricketTrajectoryAnalyzer
from core.cricket.pitch_mapper import CricketPitchMapper
from core.cricket.speed_calculator import CricketSpeedCalculator
from core.cricket.shot_classifier import CricketShotClassifier
from core.cricket.event_detector import CricketEventDetector
from core.cricket.stats_extractor import CricketStatsExtractor
from core.cricket.visualizer import CricketVisualizer
from core.cricket.highlight_generator import CricketHighlightGenerator
from core.cricket.analytics_report import CricketAnalyticsReport
from core.cricket.commentary_generator import CricketCommentaryGenerator
from core.cricket.drs_engine import DRSEngine, DRSVerdict, PitchingZone, ImpactZone, WicketsResult, OnFieldCall
from core.cricket.analytics_engine import ExpectedDismissalEngine
from core.cricket.umpiring import NoBallDetector
from core.cricket.delivery_manager import DeliveryLifecycleManager


def _finalize_delivery(
    delivery_number: int,
    current_over: str,
    trajectory_analyzer: CricketTrajectoryAnalyzer,
    speed_calculator: CricketSpeedCalculator,
    pitch_mapper: CricketPitchMapper,
    xd_engine: ExpectedDismissalEngine,
    shot_classifier: CricketShotClassifier,
    event_detector: CricketEventDetector,
    drs_engine: DRSEngine,
    stats_extractor: CricketStatsExtractor,
    visualizer: CricketVisualizer,
    commentary_gen: CricketCommentaryGenerator,
    analytics: CricketAnalyticsReport,
    ball_metric_coords: dict,
    ball_tracks: list,
    player_tracks: list,
    stumps_info: list,
    highlight_frames: list,
    global_frame: int,
    noball_detector: Optional[NoBallDetector] = None
):
    """Finalizes analytics, commentary, and DRS adjudication for a completed delivery."""
    bounce_pts = trajectory_analyzer.bounce_points
    last_bounce = bounce_pts[-1] if bounce_pts else {}
    speeds_data = speed_calculator.calculate_delivery_speeds(
        ball_metric_coords,
        bounce_frame=last_bounce.get('frame')
    )

    bounce_metric = None
    if last_bounce and 'coord' in last_bounce:
        bmx, bmy = pitch_mapper.to_metric(last_bounce['coord'][0], last_bounce['coord'][1])
        if bmx is not None:
            bounce_metric = (bmx, bmy)
            last_bounce['coord_metric'] = bounce_metric

    metric_x_val = bounce_metric[0] if bounce_metric else (list(ball_metric_coords.values())[0][0] if ball_metric_coords else 10.0)
    metric_y_val = bounce_metric[1] if bounce_metric else 0.0
    line_len = pitch_mapper.classify_line_and_length(metric_x_val, metric_y_val)

    bounce_3d = None
    if bounce_metric:
        bounce_3d = (float(bounce_metric[1]), float(bounce_metric[0]), 0.036)

    nominal_impact = (0.02, 18.0, 0.45)
    nominal_vel = (0.01, speeds_data.get('bounce_speed_kmh', 110.0) / 3.6, 1.2)
    projected_3d_pts, stump_impact_point = trajectory_analyzer.project_ballistic_to_stumps(
        nominal_impact, nominal_vel
    )

    xd_data = xd_engine.calculate_xd(
        bounce_point=bounce_3d,
        pad_point=nominal_impact,
        stump_point=stump_impact_point,
        speed_kmh=speeds_data.get('release_speed_kmh', 125.0)
    )

    # Dynamic impact index detection based on actual bounce and trajectory turning point (Bug 8)
    all_pts = trajectory_analyzer.all_centroids
    impact_idx = None
    if last_bounce and 'frame' in last_bounce:
        b_frame = last_bounce['frame']
        for idx, (f, _) in enumerate(all_pts):
            if f >= b_frame + 4:
                impact_idx = idx
                break
    if impact_idx is None:
        impact_idx = max(2, int(len(all_pts) * 0.60)) if len(all_pts) >= 4 else max(1, len(all_pts) // 2)

    pre_impact = [pt[1] for pt in all_pts[:impact_idx]]
    post_impact = [pt[1] for pt in all_pts[impact_idx:]]
    shot_res = shot_classifier.classify_shot(None, pre_impact, post_impact)

    event_res = event_detector.analyze_delivery(
        ball_tracks, player_tracks,
        bounce_info={'coord_3d': bounce_3d} if bounce_3d else last_bounce,
        stumps_info=stumps_info,
        shot_info={'shot_label': shot_res[0]},
        pad_impact_metric=nominal_impact,
        predicted_stump_metric=stump_impact_point
    )

    if event_res.get('drs_verdict'):
        drs_v = DRSVerdict(
            pitching=PitchingZone(event_res['drs_verdict']['pitching']),
            pitching_coord=event_res['drs_verdict']['pitching_coord'],
            impact=ImpactZone(event_res['drs_verdict']['impact']),
            impact_coord=event_res['drs_verdict']['impact_coord'],
            wickets=WicketsResult(event_res['drs_verdict']['wickets']),
            stump_coord=event_res['drs_verdict']['stump_coord'],
            on_field_call=OnFieldCall(event_res['drs_verdict']['on_field_call']),
            final_verdict=event_res['drs_verdict']['final_verdict'],
            reasons=event_res['drs_verdict']['reasons']
        )
        banner = drs_engine.render_drs_banner(drs_v)
        visualizer.trigger_drs_display(banner, duration_frames=60)

    if event_res.get('is_wicket') or event_res.get('is_boundary'):
        highlight_frames.append(global_frame)

    # Wire over_number into commentary (Bug 16)
    commentary_line = commentary_gen.generate_delivery_commentary(
        delivery_number=delivery_number,
        speeds_dict=speeds_data,
        line_length_tuple=line_len,
        shot_tuple=shot_res,
        event_dict=event_res,
        xd_info=xd_data,
        over_number=current_over
    )

    delivery_metric = stats_extractor.record_delivery(
        delivery_number=delivery_number,
        speeds_dict=speeds_data,
        bounce_dict=last_bounce,
        line_length_tuple=line_len,
        shot_tuple=shot_res,
        event_dict=event_res,
        over_str=current_over
    )
    delivery_metric['expected_dismissal_xd'] = xd_data
    if event_res.get('drs_verdict'):
        delivery_metric['drs_adjudication'] = event_res['drs_verdict']

    # Wire NoBallDetector front-foot crease check (Bug 6)
    if noball_detector is not None:
        bowler_box = None
        for frame_tracks in player_tracks:
            for pid, info in frame_tracks.items():
                if info.get('role') == 'bowler':
                    bowler_box = info.get('bbox')
                    break
            if bowler_box is not None:
                break
        if bowler_box is not None:
            foot_mx, _ = pitch_mapper.to_metric((bowler_box[0] + bowler_box[2]) / 2.0, bowler_box[3])
            foot_y = foot_mx if foot_mx is not None else 1.15
            nb_verdict = noball_detector.evaluate_front_foot(foot_y)
            delivery_metric['no_ball_adjudication'] = nb_verdict.to_dict()

    delivery_metric['commentary'] = commentary_line
    analytics.update_delivery(delivery_metric, commentary_text=commentary_line)

    # Line is index 0, length is index 1
    visualizer.update_telemetry(
        speed_kmh=speeds_data.get('release_speed_kmh', 0.0) or speeds_data.get('avg_speed_kmh', 0.0),
        event=event_res.get('outcome'),
        shot=shot_res[0],
        line_length=f"{line_len[0]} | {line_len[1]}",
        xd_info=xd_data,
        over_str=current_over
    )


def run_cricket_pipeline(video_path: str, max_frames=None, frame_skip=None, batch_size=None, resume=False):
    """
    Complete end-to-end execution of the SportsVision Cricket Pipeline with DRS Hawk-Eye & 3D Kinematics.
    """
    print("=" * 60)
    print(" SportsVision Cricket Spatial Analytics Engine v2.1")
    print(" DRS Hawk-Eye & Ballistic Telemetry Pipeline")
    print("=" * 60)

    # 1. Hardware Verification
    print("\n[1/7] Initializing Hardware...")
    verify_cuda()

    BATCH_SIZE = batch_size or getattr(config, 'BATCH_SIZE', 250)
    FRAME_SKIP = frame_skip or getattr(config, 'FRAME_SKIP', 1)

    # 2. Initialize AI Engines
    print("\n[2/7] Initializing Cricket Analytics Engines...")
    detector = CricketDetector()
    ball_tracker = CricketBallTracker(fps=30)
    trajectory_analyzer = CricketTrajectoryAnalyzer(fps=30)
    pitch_mapper = CricketPitchMapper()
    speed_calculator = CricketSpeedCalculator(fps=30)
    shot_classifier = CricketShotClassifier()
    event_detector = CricketEventDetector(fps=30)
    stats_extractor = CricketStatsExtractor()
    visualizer = CricketVisualizer()
    highlight_gen = CricketHighlightGenerator()
    analytics = CricketAnalyticsReport()
    commentary_gen = CricketCommentaryGenerator()
    drs_engine = DRSEngine()
    xd_engine = ExpectedDismissalEngine()
    noball_detector = NoBallDetector()

    # 3. Open Video
    print("\n[3/7] Opening Video Source...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open cricket video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    analytics.fps = fps
    analytics.total_frames = total_frames
    speed_calculator.fps = fps
    speed_calculator.dt = 1.0 / max(fps, 1)
    ball_tracker.fps = fps
    ball_tracker.dt = 1.0 / max(fps, 1)
    trajectory_analyzer.fps = fps
    delivery_mgr = DeliveryLifecycleManager(fps=fps)

    out_name = os.path.splitext(os.path.basename(video_path))[0]
    cricket_out_dir = getattr(config, 'CRICKET_OUTPUT_DIR', os.path.join(config.WORKSPACE_ROOT, "data", "output", "cricket"))
    os.makedirs(cricket_out_dir, exist_ok=True)
    out_video_path = os.path.join(cricket_out_dir, f"{out_name}_annotated.mp4")

    out_res = getattr(config, 'OUTPUT_RESOLUTION', (1280, 720))
    fourcc = cv2.VideoWriter_fourcc(*getattr(config, 'OUTPUT_CODEC', 'mp4v'))
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, out_res)

    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height} -> {out_res[0]}x{out_res[1]} @ {fps} FPS")
    print(f"  Total Frames: {total_frames}")
    print(f"  Output: {out_video_path}")

    # 4. Processing Streaming Batches
    print(f"\n[4/7] Processing Cricket Video (Streaming Batches)...")
    frame_count = 0
    pipeline_start = time.time()
    pbar = tqdm.tqdm(total=total_frames, desc="Cricket Processing", unit="frames")

    highlight_frames = []
    delivery_number = 1
    ball_metric_coords = {}
    ball_tracks = []
    player_tracks = []
    stumps_info = []

    # Delivery-scoped frame buffers to prevent cross-batch data loss (Bug 3 & Bug 7)
    curr_delivery_ball_tracks = []
    curr_delivery_player_tracks = []
    curr_delivery_stumps_info = []

    while True:
        video_frames = []
        while len(video_frames) < BATCH_SIZE:
            if max_frames and frame_count >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                frame_count += 1
                continue

            frame = cv2.resize(frame, out_res, interpolation=cv2.INTER_CUBIC)
            video_frames.append(frame)
            frame_count += 1

        if not video_frames:
            break

        # A. Detect players, ball, and pitch equipment
        player_tracks, raw_ball_tracks, stumps_info = detector.track(video_frames)

        # B. Smooth Ball Tracks with Kalman Filter
        ball_tracks = ball_tracker.track_batch(raw_ball_tracks, start_frame_idx=frame_count - len(video_frames))

        for idx, frame in enumerate(video_frames):
            global_frame = frame_count - len(video_frames) + idx
            b_info = ball_tracks[idx]
            p_frame = player_tracks[idx] if idx < len(player_tracks) else {}
            s_frame = stumps_info[idx] if idx < len(stumps_info) else None

            # Add ball point if detected
            b_metric = None
            if 1 in b_info:
                cx, cy = b_info[1]['centroid']
                trajectory_analyzer.add_point(global_frame, (cx, cy))
                mx, my = pitch_mapper.to_metric(cx, cy)
                if mx is not None:
                    b_metric = (mx, my)
                    ball_metric_coords[global_frame] = b_metric

            # Update Delivery Lifecycle Manager (multi-over segmentation)
            is_active, is_completed, current_over = delivery_mgr.update(
                frame_idx=global_frame,
                ball_info=b_info,
                ball_metric=b_metric,
                player_track_frame=p_frame
            )

            # Accumulate per-delivery tracks across batch boundaries (Bug 3 & Bug 7)
            if is_active or is_completed or len(trajectory_analyzer.all_centroids) > 0:
                curr_delivery_ball_tracks.append(b_info)
                curr_delivery_player_tracks.append(p_frame)
                curr_delivery_stumps_info.append(s_frame)

            # Live speed update if ball is moving
            if len(trajectory_analyzer.all_centroids) >= 3 and global_frame % 4 == 0:
                live_spd = speed_calculator.calculate_delivery_speeds(ball_metric_coords)
                visualizer.update_telemetry(
                    speed_kmh=live_spd.get('release_speed_kmh', 0.0) or live_spd.get('avg_speed_kmh', 0.0),
                    over_str=current_over
                )

            # Mini field mapping (22-yd pitch strip):
            # 1. Current ball head marker
            minimap_pos = None
            if 1 in b_info:
                bx, by = b_info[1]['centroid']
                minimap_pos = pitch_mapper.to_minimap(bx, by)

            # 2. Entire flight path mapped to 22-yard pitch strip
            cam_pts = [pt[1] for pt in trajectory_analyzer.all_centroids]
            minimap_trail = pitch_mapper.to_minimap_multi(cam_pts)

            # 3. Bounce landing spot on minimap
            minimap_bounce = None
            if trajectory_analyzer.bounce_points:
                last_bp = trajectory_analyzer.bounce_points[-1]
                minimap_bounce = pitch_mapper.to_minimap(last_bp['coord'][0], last_bp['coord'][1])
                if minimap_bounce is None:
                    bmx, bmy = pitch_mapper.to_metric(last_bp['coord'][0], last_bp['coord'][1])
                    if bmx is not None:
                        minimap_bounce = pitch_mapper.metric_to_minimap(bmx, bmy)

            # 4. Ballistic continuation towards wickets on minimap
            minimap_projected = []
            if len(cam_pts) >= 4:
                nom_impact = (0.02, 14.0, 0.35)
                cur_spd = visualizer.current_speed_kmh or 115.0
                nom_vel = (0.01, cur_spd / 3.6, 0.8)
                proj_pts_3d, _ = trajectory_analyzer.project_ballistic_to_stumps(nom_impact, nom_vel)
                for pt3d in proj_pts_3d:
                    m_pt = pitch_mapper.metric_to_minimap(pt3d[1], pt3d[0])
                    if m_pt:
                        minimap_projected.append(m_pt)

            # 5. Full Ground Field Radar mapping (Players & Ball Travel)
            ground_pos = None
            if 1 in b_info:
                bx, by = b_info[1]['centroid']
                ground_pos = pitch_mapper.to_ground_map(bx, by)

            ground_trail = pitch_mapper.to_ground_map_multi(cam_pts)

            ground_players = []
            if p_frame:
                for pid, pinfo in p_frame.items():
                    p_bbox = pinfo.get('bbox')
                    if p_bbox:
                        fx = (p_bbox[0] + p_bbox[2]) / 2.0
                        fy = p_bbox[3]
                        g_pos = pitch_mapper.to_ground_map(fx, fy)
                        if g_pos:
                            ground_players.append((pinfo.get('role', 'player'), g_pos))

            # Render frame with pitch minimap & ground radar tracking
            annotated_frame = visualizer.draw_frame(
                frame,
                p_frame,
                b_info,
                trajectory_trail=trajectory_analyzer.get_smoothed_trail(),
                projected_trail=trajectory_analyzer.project_future_path_pixels(),
                bounce_points=trajectory_analyzer.bounce_points[-1:] if trajectory_analyzer.bounce_points else [],
                minimap_ball_pos=minimap_pos,
                minimap_trail=minimap_trail,
                minimap_bounce=minimap_bounce,
                minimap_projected=minimap_projected,
                stumps_info=s_frame,
                ground_ball_pos=ground_pos,
                ground_trail=ground_trail,
                ground_player_positions=ground_players
            )
            for _ in range(FRAME_SKIP if FRAME_SKIP > 1 else 1):
                out_video.write(annotated_frame)

            # Finalize completed delivery, reset trajectory, advance over
            if is_completed:
                delivery_frames_ball = curr_delivery_ball_tracks if curr_delivery_ball_tracks else ball_tracks
                delivery_frames_players = curr_delivery_player_tracks if curr_delivery_player_tracks else player_tracks
                delivery_frames_stumps = curr_delivery_stumps_info if curr_delivery_stumps_info else stumps_info

                _finalize_delivery(
                    delivery_number=delivery_number,
                    current_over=current_over,
                    trajectory_analyzer=trajectory_analyzer,
                    speed_calculator=speed_calculator,
                    pitch_mapper=pitch_mapper,
                    xd_engine=xd_engine,
                    shot_classifier=shot_classifier,
                    event_detector=event_detector,
                    drs_engine=drs_engine,
                    stats_extractor=stats_extractor,
                    visualizer=visualizer,
                    commentary_gen=commentary_gen,
                    analytics=analytics,
                    ball_metric_coords=ball_metric_coords,
                    ball_tracks=delivery_frames_ball,
                    player_tracks=delivery_frames_players,
                    stumps_info=delivery_frames_stumps,
                    highlight_frames=highlight_frames,
                    global_frame=global_frame,
                    noball_detector=noball_detector
                )
                delivery_number += 1
                trajectory_analyzer.reset_delivery()
                detector.reset_ball_position()
                ball_metric_coords.clear()
                curr_delivery_ball_tracks.clear()
                curr_delivery_player_tracks.clear()
                curr_delivery_stumps_info.clear()

        # Free PyTorch CUDA cache between streaming batches for extended 30+ min endurance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - pipeline_start
        fps_avg = frame_count / elapsed if elapsed > 0 else 0
        remaining_frames = max(0, total_frames - frame_count)
        eta_sec = remaining_frames / fps_avg if fps_avg > 0 else 0

        pbar.update(len(video_frames))
        pbar.set_postfix({
            'FPS': f'{fps_avg:.1f}',
            'ETA': format_time(eta_sec),
            'VRAM': f'{torch.cuda.memory_allocated() / 1024**2:.0f}MB' if torch.cuda.is_available() else 'N/A'
        })

    cap.release()
    out_video.release()
    pbar.close()

    # Finalize any in-progress delivery if video ended while active
    if delivery_mgr.force_complete_if_active() or (delivery_number == 1 and len(trajectory_analyzer.all_centroids) >= 3):
        delivery_frames_ball = curr_delivery_ball_tracks if curr_delivery_ball_tracks else ball_tracks
        delivery_frames_players = curr_delivery_player_tracks if curr_delivery_player_tracks else player_tracks
        delivery_frames_stumps = curr_delivery_stumps_info if curr_delivery_stumps_info else stumps_info

        _finalize_delivery(
            delivery_number=delivery_number,
            current_over=delivery_mgr.get_over_display(),
            trajectory_analyzer=trajectory_analyzer,
            speed_calculator=speed_calculator,
            pitch_mapper=pitch_mapper,
            xd_engine=xd_engine,
            shot_classifier=shot_classifier,
            event_detector=event_detector,
            drs_engine=drs_engine,
            stats_extractor=stats_extractor,
            visualizer=visualizer,
            commentary_gen=commentary_gen,
            analytics=analytics,
            ball_metric_coords=ball_metric_coords,
            ball_tracks=delivery_frames_ball,
            player_tracks=delivery_frames_players,
            stumps_info=delivery_frames_stumps,
            highlight_frames=highlight_frames,
            global_frame=frame_count,
            noball_detector=noball_detector
        )
        trajectory_analyzer.reset_delivery()
        ball_metric_coords.clear()
        curr_delivery_ball_tracks.clear()
        curr_delivery_player_tracks.clear()
        curr_delivery_stumps_info.clear()

    total_time = time.time() - pipeline_start
    print(f"\n[5/7] Cricket Video processing complete!")
    print(f"  Annotated video: {out_video_path}")
    print(f"  Time taken: {format_time(total_time)} ({(frame_count / max(total_time, 0.001)):.1f} FPS)")

    # 6. Highlights (Audio-Visual Multi-Modal Fusion)
    print("\n[6/7] Extracting Audio-Visual Highlights...")
    highlight_path = os.path.join(cricket_out_dir, f"{out_name}_highlights.mp4")
    fused_hl_frames = highlight_gen.extract_highlights(video_path, highlight_frames, highlight_path, fps=fps)
    analytics.record_highlights(fused_hl_frames)

    # 7. Analytics Report
    print("\n[7/7] Generating Cricket Analytics & Commentary Report...")
    report_file = analytics.save_report(out_name)

    return {
        'annotated_video': out_video_path,
        'highlights_video': highlight_path,
        'stats_report': report_file
    }
