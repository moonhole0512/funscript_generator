import os
import sys
import cv2
import numpy as np
import json
from algorithms import (
    OpticalFlowEstimator,
    ROIDetector,
    QuickSceneDetector,
    DynamicROITracker,
    SceneSegmenter,
    PositionEstimator,
    ActionPointGenerator,
    ScriptPostProcessor,
    MotionExtractor,
    RESIZE_WIDTH
)
from scipy.signal import savgol_filter

# Scene count threshold: videos with 3+ scenes use global ROI strategy
MULTI_SCENE_THRESHOLD = 3


def _classify_video(scene_boundaries):
    """
    Classify video based on scene count.
    Returns 'single_scene' (1-2 scenes, per-scene ROI) or 'multi_scene' (3+, global ROI).
    """
    if len(scene_boundaries) < MULTI_SCENE_THRESHOLD:
        return "single_scene"
    return "multi_scene"


def _get_roi_for_frame(frame_idx, scene_boundaries, per_scene_rois):
    """Return the ROI for a given frame index based on which scene it belongs to."""
    for i, (start, end) in enumerate(scene_boundaries):
        if start <= frame_idx < end:
            return per_scene_rois[i]
    # Fallback to last scene's ROI
    return per_scene_rois[-1] if per_scene_rois else (0.15, 0.85, 0.15, 0.85)


def run_analysis(video_path, progress_callback=None):
    """
    Two-pass funscript generation pipeline with per-scene adaptive ROI.

    Pipeline:
    Pass 1 (lightweight):
      1. Quick scene detection (histogram-based)
      2. Per-scene ROI detection (optical flow only within each scene)

    Pass 2 (full extraction):
      3. Frame-by-frame motion extraction using per-scene ROIs
      4. Scene segmentation with per-scene thresholds
      5. Position estimation with adaptive HPF
      6. Per-segment normalization with noise guard
      7. Adaptive contrast expansion
      8. Frequency-adaptive action point generation
      9. Physics validation
      10. Save output
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return False

    # Open video to get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    video_duration_ms = int((total_frames / fps) * 1000)
    cap.release()

    if total_frames < 10:
        print("Error: Video too short")
        return False

    # ══════════════════════════════════════════════
    # PASS 1: Scene Detection + Per-Scene ROI
    # ══════════════════════════════════════════════

    # ── Phase 1: Quick Scene Detection ──
    if progress_callback:
        progress_callback(0, 100, "Detecting scenes...")

    scene_detector = QuickSceneDetector(threshold=0.7)
    scene_boundaries = scene_detector.detect(video_path, sample_interval=5)
    print(f"Detected {len(scene_boundaries)} scene(s): {[(s, e) for s, e in scene_boundaries]}")

    # ── Classify video type ──
    video_type = _classify_video(scene_boundaries)
    print(f"Video classification: {video_type} ({len(scene_boundaries)} scenes)")

    if progress_callback:
        progress_callback(3, 100, f"classified:{video_type}")

    # ── Phase 2: ROI Detection (strategy depends on video type) ──
    if progress_callback:
        progress_callback(5, 100, "Detecting ROI...")

    flow_estimator = OpticalFlowEstimator()
    roi_detector = ROIDetector(flow_estimator)

    if video_type == "single_scene":
        # Per-scene ROI: proven reliable for 1-2 scene videos
        per_scene_rois = []
        for i, (scene_start, scene_end) in enumerate(scene_boundaries):
            scene_length = scene_end - scene_start
            scene_sample_count = min(40, max(10, scene_length // 5))
            roi = roi_detector.detect_roi(video_path, sample_count=scene_sample_count,
                                           frame_range=(scene_start, scene_end))
            per_scene_rois.append(roi)
            print(f"  Scene {i+1} [{scene_start}-{scene_end}]: "
                  f"ROI y=[{roi[0]:.2f}-{roi[1]:.2f}], x=[{roi[2]:.2f}-{roi[3]:.2f}]")
        roi_strategy = "per_scene"
    else:
        # Global ROI: single ROI from entire video, reliable for multi-scene
        global_roi = roi_detector.detect_roi(video_path, sample_count=80,
                                              frame_range=None)
        per_scene_rois = [global_roi] * len(scene_boundaries)
        print(f"  Global ROI: y=[{global_roi[0]:.2f}-{global_roi[1]:.2f}], "
              f"x=[{global_roi[2]:.2f}-{global_roi[3]:.2f}]")
        roi_strategy = "global"

    if progress_callback:
        progress_callback(15, 100, "ROI detection complete.")

    # ══════════════════════════════════════════════
    # PASS 2: Full Motion Extraction + Processing
    # ══════════════════════════════════════════════

    # ── Phase 3: Frame-by-frame motion extraction with dynamic ROI tracking ──
    motion_extractor = MotionExtractor(flow_estimator)
    roi_tracker = DynamicROITracker(fps)

    # Initialize tracker with first scene's ROI
    initial_roi = _get_roi_for_frame(0, scene_boundaries, per_scene_rois)
    roi_tracker.reset(initial_roi)

    # Pre-compute scene boundary frame set for fast lookup
    scene_change_set = {s for s, _ in scene_boundaries if s > 0}

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return False

    prev_frame = cv2.resize(
        prev_frame,
        (RESIZE_WIDTH, int(prev_frame.shape[0] * (RESIZE_WIDTH / prev_frame.shape[1])))
    )

    velocity_signal = []
    magnitude_signal = []

    for i in range(total_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(
            frame,
            (RESIZE_WIDTH, int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1])))
        )

        frame_h, frame_w = frame_resized.shape[:2]

        # At scene boundaries, reset the ROI tracker with the new scene's ROI
        if i in scene_change_set:
            new_scene_roi = _get_roi_for_frame(i, scene_boundaries, per_scene_rois)
            roi_tracker.reset(new_scene_roi)

        # Get optical flow once (used for both ROI tracking and motion extraction)
        flow = flow_estimator.estimate_flow(prev_frame, frame_resized)

        # Update ROI position using camera motion from optical flow
        current_roi = roi_tracker.update(flow, frame_h, frame_w)

        # Extract motion using the dynamically tracked ROI and pre-computed flow
        velocity, magnitude, _ = motion_extractor.extract_velocity_signal(
            prev_frame, frame_resized, current_roi, precomputed_flow=flow
        )

        velocity_signal.append(velocity)
        magnitude_signal.append(magnitude)

        prev_frame = frame_resized

        if progress_callback and i % 5 == 0:
            pct = 15 + int((i / total_frames) * 55)  # 15-70%
            progress_callback(pct, 100, f"Extracting motion... ({i+1}/{total_frames})")

    cap.release()

    velocity_signal = np.array(velocity_signal, dtype=np.float64)
    magnitude_signal = np.array(magnitude_signal, dtype=np.float64)

    if len(velocity_signal) < 10:
        print("Error: Not enough frames processed")
        return False

    # ── Phase 4: Scene Segmentation with per-scene thresholds ──
    if progress_callback:
        progress_callback(70, 100, "Analyzing scenes...")

    scene_segmenter = SceneSegmenter(fps)

    # Use scene boundaries as change points for segmentation
    scene_change_frames = [s for s, _ in scene_boundaries]

    segments = scene_segmenter.classify_segments(
        magnitude_signal, scene_change_frames, len(velocity_signal)
    )

    active_count = sum(1 for _, _, t in segments if t == 'ACTIVE')
    quiet_count = sum(1 for _, _, t in segments if t == 'QUIET')
    trans_count = sum(1 for _, _, t in segments if t == 'TRANSITION')
    print(f"Segments: {len(segments)} total "
          f"({active_count} active, {trans_count} transition, {quiet_count} quiet)")

    # ── Phase 5: Position Estimation ──
    if progress_callback:
        progress_callback(75, 100, "Estimating position signal...")

    # Minimal smoothing on velocity before integration
    if len(velocity_signal) > 5:
        velocity_smoothed = savgol_filter(velocity_signal, 5, 1)
    else:
        velocity_smoothed = velocity_signal

    position_estimator = PositionEstimator(fps)

    # Convert velocity to drift-free position (adaptive HPF, per-scene filtering)
    position_raw = position_estimator.velocity_to_position(
        velocity_smoothed, segments, scene_boundaries=scene_boundaries
    )

    # Per-segment normalization with noise guard
    position_normalized = position_estimator.normalize_per_segment(position_raw, segments)

    # Adaptive contrast expansion per segment
    position_normalized = position_estimator.expand_contrast(position_normalized, segments=segments)

    # ── Phase 6: Action Point Generation ──
    if progress_callback:
        progress_callback(85, 100, "Generating action points...")

    action_generator = ActionPointGenerator(fps)
    actions = action_generator.generate(position_normalized, segments)

    # ── Phase 7: Post-processing ──
    if progress_callback:
        progress_callback(90, 100, "Post-processing...")

    post_processor = ScriptPostProcessor()
    actions = post_processor.validate_and_fix(actions, max_speed=500)

    # ── Phase 8: Save ──
    video_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(video_dir, base_name + ".funscript")

    # Build ROI info for metadata
    roi_info = []
    for i, (roi, (s, e)) in enumerate(zip(per_scene_rois, scene_boundaries)):
        roi_info.append({
            "scene": i + 1,
            "frames": [s, e],
            "y_range": [round(roi[0], 2), round(roi[1], 2)],
            "x_range": [round(roi[2], 2), round(roi[3], 2)]
        })

    output_script = {
        "actions": actions,
        "inverted": False,
        "metadata": {
            "version": "3.1",
            "creator": "Eroscript Generator AI",
            "video_type": video_type,
            "roi_strategy": roi_strategy,
            "pipeline": [
                "Quick scene detection",
                f"Video classification ({video_type})",
                f"ROI detection ({roi_strategy})",
                "Signed vertical flow extraction",
                "Per-scene adaptive threshold segmentation",
                "Adaptive Butterworth HPF drift removal",
                "Per-segment normalization with noise guard",
                "Adaptive contrast expansion",
                "Frequency-adaptive action sampling",
                "Physics validation"
            ],
            "total_actions": len(actions),
            "scenes": len(scene_boundaries),
            "roi_per_scene": roi_info,
            "segments": {
                "total": len(segments),
                "active": active_count,
                "transition": trans_count,
                "quiet": quiet_count
            }
        },
        "range": 100
    }

    with open(output_path, 'w') as f:
        json.dump(output_script, f, indent=2)

    if progress_callback:
        progress_callback(100, 100, f"Done! Saved to {os.path.basename(output_path)}")

    print(f"\nGenerated {len(actions)} actions")
    print(f"Saved to: {output_path}")

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        print("Usage: python main.py <video_path>")
