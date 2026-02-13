import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from scipy.signal import find_peaks, savgol_filter, butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d

# Internal Defaults
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RAFT_SMALL = False
RESIZE_WIDTH = 512


class QuickSceneDetector:
    """
    Pass 1: Lightweight histogram-based scene boundary detection.
    No optical flow needed — fast enough to run before ROI detection.
    """

    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def detect(self, video_path, sample_interval=5):
        """
        Detect scene boundaries by comparing histograms every sample_interval frames.
        Returns list of (scene_start_frame, scene_end_frame) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            cap.release()
            return [(0, total_frames)]

        change_frames = [0]
        prev_hist = None

        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            if prev_hist is not None:
                score = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                if score < self.threshold:
                    change_frames.append(frame_idx)

            prev_hist = hist

        cap.release()

        # Build scene ranges
        scenes = []
        for i in range(len(change_frames)):
            start = change_frames[i]
            end = change_frames[i + 1] if i + 1 < len(change_frames) else total_frames
            if end - start >= 10:  # minimum 10 frames per scene
                scenes.append((start, end))

        # Merge very short scenes (< 30 frames) into neighbors
        if len(scenes) > 1:
            merged = [scenes[0]]
            for s, e in scenes[1:]:
                if e - s < 30 and merged:
                    prev_s, prev_e = merged[-1]
                    merged[-1] = (prev_s, e)
                else:
                    merged.append((s, e))
            scenes = merged

        return scenes if scenes else [(0, total_frames)]


class OpticalFlowEstimator:
    def __init__(self):
        self.device = DEVICE
        self.weights = Raft_Small_Weights.DEFAULT if RAFT_SMALL else Raft_Large_Weights.DEFAULT
        self.model = raft_small(weights=self.weights) if RAFT_SMALL else raft_large(weights=self.weights)
        self.model = self.model.to(self.device).eval()
        self.transforms = self.weights.transforms()

    def estimate_flow(self, img1, img2):
        """
        Estimate optical flow between two frames using RAFT.
        img1, img2: numpy arrays (H, W, 3) in BGR (OpenCV format)
        Returns: flow tensor (2, H, W) on GPU
        """
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2_tensor = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        img1_batch, img2_batch = self.transforms(img1_tensor, img2_tensor)

        with torch.no_grad():
            list_of_flows = self.model(img1_batch, img2_batch)
            predicted_flow = list_of_flows[-1][0]  # (2, H, W)

        return predicted_flow


class ROIDetector:
    """Detects the region of interest where the primary vertical motion occurs."""

    def __init__(self, flow_estimator):
        self.flow_estimator = flow_estimator

    def detect_roi(self, video_path, sample_count=80, frame_range=None):
        """
        Sample frames from the video and find the region with the most
        consistent vertical oscillation.
        frame_range: optional (start_frame, end_frame) to limit detection to a scene.
        Returns: (y_start, y_end, x_start, x_end) as fractions of frame size (0-1).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (0.15, 0.85, 0.15, 0.85)  # fallback

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_range is not None:
            range_start, range_end = frame_range
            range_start = max(0, range_start)
            range_end = min(total_frames, range_end)
        else:
            range_start, range_end = 0, total_frames

        range_length = range_end - range_start
        if range_length < 10:
            cap.release()
            return (0.15, 0.85, 0.15, 0.85)

        # Sample frames evenly distributed across the range
        effective_count = min(sample_count, range_length // 2)
        step = max(1, range_length // (effective_count + 1))
        sample_indices = list(range(range_start + step, range_end - 1, step))[:effective_count]

        if len(sample_indices) < 5:
            cap.release()
            return (0.15, 0.85, 0.15, 0.85)

        # Accumulate vertical flow variance in a grid
        first_frame = self._read_frame(cap, sample_indices[0])
        if first_frame is None:
            cap.release()
            return (0.15, 0.85, 0.15, 0.85)

        h, w = first_frame.shape[:2]
        grid_h, grid_w = 8, 8
        cell_h, cell_w = h // grid_h, w // grid_w

        # Collect vertical flow per grid cell over time
        grid_flows = np.zeros((grid_h, grid_w, len(sample_indices) - 1))

        prev_frame = first_frame
        for i in range(1, len(sample_indices)):
            curr_frame = self._read_frame(cap, sample_indices[i])
            if curr_frame is None:
                break

            flow = self.flow_estimator.estimate_flow(prev_frame, curr_frame)
            flow_y = flow[1].cpu().numpy()  # vertical component

            for gy in range(grid_h):
                for gx in range(grid_w):
                    y_s, y_e = gy * cell_h, (gy + 1) * cell_h
                    x_s, x_e = gx * cell_w, (gx + 1) * cell_w
                    grid_flows[gy, gx, i - 1] = np.mean(flow_y[y_s:y_e, x_s:x_e])

            prev_frame = curr_frame

        cap.release()

        # Compute variance of vertical flow for each grid cell
        grid_variance = np.var(grid_flows, axis=2)

        # Find the region with highest variance (most vertical oscillation)
        # Use a 3x3 sliding window to find the best region
        best_score = 0
        best_region = (0, grid_h, 0, grid_w)

        for size_y in range(3, grid_h + 1):
            for size_x in range(3, grid_w + 1):
                for gy in range(grid_h - size_y + 1):
                    for gx in range(grid_w - size_x + 1):
                        score = np.sum(grid_variance[gy:gy + size_y, gx:gx + size_x])
                        # Prefer smaller, focused regions
                        area_penalty = (size_y * size_x) / (grid_h * grid_w)
                        adjusted_score = score / (area_penalty ** 0.3)
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_region = (gy, gy + size_y, gx, gx + size_x)

        gy1, gy2, gx1, gx2 = best_region
        roi = (
            gy1 / grid_h,
            gy2 / grid_h,
            gx1 / grid_w,
            gx2 / grid_w,
        )
        return roi

    def _read_frame(self, cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1]))))


class DynamicROITracker:
    """
    Tracks ROI position dynamically using optical flow propagation.
    Instead of a static ROI per scene, this adjusts the ROI frame-by-frame
    to follow camera pans and zooms within a scene.
    """

    def __init__(self, fps, revalidate_interval_sec=2.0):
        self.fps = fps
        self.revalidate_interval = int(fps * revalidate_interval_sec)
        self.current_roi = None
        self.frame_count_since_reset = 0
        # Accumulate flow-based shifts for smooth tracking
        self._shift_y = 0.0
        self._shift_x = 0.0
        # Smoothing factor to prevent jitter
        self._ema_alpha = 0.3

    def reset(self, initial_roi):
        """Reset tracker with a new base ROI (e.g., at scene boundary)."""
        self.current_roi = initial_roi
        self.frame_count_since_reset = 0
        self._shift_y = 0.0
        self._shift_x = 0.0

    def update(self, flow_tensor, frame_h, frame_w):
        """
        Update ROI position based on the optical flow of the current frame.
        Uses background flow (outside ROI) to estimate camera motion,
        then shifts the ROI accordingly.

        flow_tensor: (2, H, W) optical flow tensor
        frame_h, frame_w: frame dimensions
        Returns: updated roi_fractions (y_start, y_end, x_start, x_end)
        """
        if self.current_roi is None:
            return (0.15, 0.85, 0.15, 0.85)

        y1_frac, y2_frac, x1_frac, x2_frac = self.current_roi
        y1 = int(y1_frac * frame_h)
        y2 = int(y2_frac * frame_h)
        x1 = int(x1_frac * frame_w)
        x2 = int(x2_frac * frame_w)

        # Compute background flow (camera motion estimate)
        bg_mask = torch.ones(flow_tensor.shape[1], flow_tensor.shape[2],
                             dtype=torch.bool, device=flow_tensor.device)
        bg_mask[y1:y2, x1:x2] = False

        if bg_mask.sum() > 100:
            cam_dy = flow_tensor[1][bg_mask].median().item()
            cam_dx = flow_tensor[0][bg_mask].median().item()
        else:
            cam_dy = 0.0
            cam_dx = 0.0

        # Convert pixel shift to fractional shift
        dy_frac = cam_dy / frame_h
        dx_frac = cam_dx / frame_w

        # EMA smoothing to prevent jitter
        self._shift_y = self._ema_alpha * dy_frac + (1 - self._ema_alpha) * self._shift_y
        self._shift_x = self._ema_alpha * dx_frac + (1 - self._ema_alpha) * self._shift_x

        # Apply shift to ROI (move ROI with camera)
        roi_height = y2_frac - y1_frac
        roi_width = x2_frac - x1_frac

        new_y1 = y1_frac + self._shift_y
        new_y2 = new_y1 + roi_height
        new_x1 = x1_frac + self._shift_x
        new_x2 = new_x1 + roi_width

        # Clamp to frame boundaries
        if new_y1 < 0:
            new_y1, new_y2 = 0.0, roi_height
        if new_y2 > 1:
            new_y1, new_y2 = 1.0 - roi_height, 1.0
        if new_x1 < 0:
            new_x1, new_x2 = 0.0, roi_width
        if new_x2 > 1:
            new_x1, new_x2 = 1.0 - roi_width, 1.0

        self.current_roi = (
            max(0.0, new_y1),
            min(1.0, new_y2),
            max(0.0, new_x1),
            min(1.0, new_x2),
        )
        self.frame_count_since_reset += 1

        return self.current_roi

    def needs_revalidation(self):
        """Check if enough frames have passed to warrant a full ROI revalidation."""
        return self.frame_count_since_reset >= self.revalidate_interval


class SceneSegmenter:
    """Detects scene changes and classifies segments as ACTIVE / QUIET / TRANSITION."""

    def __init__(self, fps):
        self.fps = fps
        self.prev_hist = None
        self.scene_change_threshold = 0.7

    def detect_scene_changes(self, frames_gray):
        """
        Detect scene change frame indices from grayscale frame list.
        Returns list of frame indices where scene changes occur.
        """
        changes = [0]
        prev_hist = None

        for i, frame in enumerate(frames_gray):
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            if prev_hist is not None:
                score = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                if score < self.scene_change_threshold:
                    changes.append(i)

            prev_hist = hist

        return changes

    def classify_segments(self, motion_magnitudes, scene_changes, total_frames):
        """
        Classify each segment between scene changes using per-scene local thresholds.
        Returns list of (start_frame, end_frame, segment_type) tuples.
        """
        # Add final boundary
        boundaries = sorted(set(scene_changes + [total_frames]))
        if boundaries[0] != 0:
            boundaries = [0] + boundaries

        segments = []
        min_quiet_frames = int(self.fps * 2.0)  # 2.0 seconds minimum for QUIET

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if end - start < 2:
                continue

            seg_motion = motion_magnitudes[start:end]

            # Per-scene LOCAL threshold instead of global
            local_threshold = self._compute_local_threshold(seg_motion)

            avg_motion = np.mean(np.abs(seg_motion))
            max_motion = np.max(np.abs(seg_motion)) if len(seg_motion) > 0 else 0

            if avg_motion < local_threshold and max_motion < local_threshold * 5:
                seg_type = 'QUIET'
            elif end - start < min_quiet_frames:
                seg_type = 'TRANSITION'
            else:
                seg_type = 'ACTIVE'

            segments.append((start, end, seg_type))

        # Safety check: if too many segments are QUIET, reclassify aggressively
        total_duration = sum(e - s for s, e, _ in segments)
        quiet_duration = sum(e - s for s, e, t in segments if t == 'QUIET')
        if total_duration > 0 and quiet_duration / total_duration > 0.70:
            segments = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                if end - start < 2:
                    continue
                seg_motion = motion_magnitudes[start:end]
                local_threshold = self._compute_local_threshold(seg_motion) * 0.3
                avg_motion = np.mean(np.abs(seg_motion))
                max_motion = np.max(np.abs(seg_motion)) if len(seg_motion) > 0 else 0
                if avg_motion < local_threshold and max_motion < local_threshold * 5:
                    seg_type = 'QUIET'
                elif end - start < min_quiet_frames:
                    seg_type = 'TRANSITION'
                else:
                    seg_type = 'ACTIVE'
                segments.append((start, end, seg_type))

        # Further split ACTIVE segments at quiet sub-regions
        refined = []
        for start, end, seg_type in segments:
            if seg_type == 'ACTIVE':
                local_threshold = self._compute_local_threshold(
                    motion_magnitudes[start:end]
                )
                sub_segs = self._split_active_at_quiet(
                    motion_magnitudes, start, end, local_threshold, min_quiet_frames
                )
                refined.extend(sub_segs)
            else:
                refined.append((start, end, seg_type))

        # Verify ACTIVE segments have rhythmic oscillation (actual action pattern)
        # Non-rhythmic motion (camera pans, character walking) → downgrade to TRANSITION
        verified = []
        for start, end, seg_type in refined:
            if seg_type == 'ACTIVE' and (end - start) >= int(self.fps):
                seg_motion = motion_magnitudes[start:end]
                if not self._is_rhythmic_motion(seg_motion):
                    seg_type = 'TRANSITION'
            verified.append((start, end, seg_type))

        return verified

    def _compute_local_threshold(self, seg_motion):
        """Compute per-scene adaptive threshold for quiet zone detection."""
        abs_motion = np.abs(seg_motion)
        nonzero = abs_motion[abs_motion > 0.01]
        if len(nonzero) < 10:
            return 0.05  # Very low fallback for sparse data
        return np.percentile(nonzero, 15)

    def _is_rhythmic_motion(self, seg_motion):
        """
        Check if motion has rhythmic vertical oscillation pattern (indicating actual action).
        Returns True if the segment likely contains repetitive up-down stroking motion.
        Non-rhythmic motion (camera pans, walking, etc.) returns False.
        """
        if len(seg_motion) < int(self.fps):
            return True  # Too short to analyze, assume active

        # Use velocity signal (signed) for oscillation detection
        sig = np.array(seg_motion, dtype=float)
        sig = sig - np.mean(sig)  # Remove DC component

        n = len(sig)
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)

        # Action frequency band: 0.5-8 Hz (typical stroking frequencies)
        action_band = (freqs >= 0.5) & (freqs <= 8.0)
        total_energy = np.sum(fft_vals ** 2)

        if total_energy < 1e-10:
            return False

        action_energy = np.sum(fft_vals[action_band] ** 2)
        ratio = action_energy / total_energy

        # Also check zero-crossing rate as oscillation indicator
        zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
        duration_sec = n / self.fps
        zc_rate = zero_crossings / duration_sec if duration_sec > 0 else 0

        # Rhythmic if either: good frequency energy ratio OR sufficient oscillation rate
        return ratio > 0.25 or zc_rate > 1.0

    def _split_active_at_quiet(self, motion_magnitudes, start, end, threshold, min_frames):
        """Split an ACTIVE segment if it contains quiet sub-regions."""
        seg_motion = np.abs(motion_magnitudes[start:end])
        # Running average
        window = max(3, min_frames)
        if len(seg_motion) < window:
            return [(start, end, 'ACTIVE')]

        smoothed = uniform_filter1d(seg_motion.astype(float), size=window)
        is_quiet = smoothed < threshold

        segments = []
        current_start = 0
        current_type = 'QUIET' if is_quiet[0] else 'ACTIVE'

        for i in range(1, len(is_quiet)):
            new_type = 'QUIET' if is_quiet[i] else 'ACTIVE'
            if new_type != current_type:
                abs_start = start + current_start
                abs_end = start + i
                if abs_end - abs_start >= 2:
                    segments.append((abs_start, abs_end, current_type))
                current_start = i
                current_type = new_type

        # Last segment
        abs_start = start + current_start
        abs_end = end
        if abs_end - abs_start >= 2:
            segments.append((abs_start, abs_end, current_type))

        # Merge very short segments into neighbors
        merged = []
        for seg in segments:
            s, e, t = seg
            if e - s < min_frames and t == 'QUIET' and merged:
                # Merge short quiet into previous
                prev_s, prev_e, prev_t = merged[-1]
                merged[-1] = (prev_s, e, prev_t)
            else:
                merged.append(seg)

        return merged if merged else [(start, end, 'ACTIVE')]


class PositionEstimator:
    """Converts velocity signal to position using adaptive Butterworth HPF to remove drift."""

    def __init__(self, fps):
        self.fps = fps

    def velocity_to_position(self, velocity_signal, segments=None, scene_boundaries=None):
        """
        Convert velocity signal to drift-free position signal.
        When scene_boundaries is provided, each scene is filtered independently
        to prevent cross-scene contamination of the Butterworth filter state.
        """
        if len(velocity_signal) < 10:
            return np.zeros(len(velocity_signal))

        # Adaptive cutoff based on detected stroke frequency
        if segments:
            dominant_freq = self._estimate_stroke_frequency(velocity_signal, segments)
            cutoff_hz = max(0.1, dominant_freq * 0.3)
        else:
            cutoff_hz = 0.3

        nyquist = self.fps / 2.0
        if cutoff_hz >= nyquist:
            cutoff_hz = nyquist * 0.5

        # Per-scene filtering: process each scene independently
        if scene_boundaries and len(scene_boundaries) > 1:
            filtered = np.zeros(len(velocity_signal))
            for scene_start, scene_end in scene_boundaries:
                scene_end = min(scene_end, len(velocity_signal))
                if scene_start >= scene_end:
                    continue
                scene_vel = velocity_signal[scene_start:scene_end]
                filtered[scene_start:scene_end] = self._filter_segment(
                    scene_vel, cutoff_hz, nyquist
                )
            return filtered

        # Single scene: filter the whole signal
        raw_position = np.cumsum(velocity_signal)
        return self._apply_hpf(raw_position, cutoff_hz, nyquist)

    def _filter_segment(self, velocity_segment, cutoff_hz, nyquist):
        """Filter a single scene's velocity to drift-free position."""
        if len(velocity_segment) < 10:
            return np.zeros(len(velocity_segment))

        # Apply stabilization: zero out first few frames after scene change
        stabilize_frames = min(int(self.fps * 0.3), len(velocity_segment) // 4)
        if stabilize_frames > 0:
            velocity_segment = velocity_segment.copy()
            velocity_segment[:stabilize_frames] = 0.0

        raw_position = np.cumsum(velocity_segment)
        return self._apply_hpf(raw_position, cutoff_hz, nyquist)

    def _apply_hpf(self, raw_position, cutoff_hz, nyquist):
        """Apply Butterworth HPF with mirror padding."""
        try:
            sos = butter(4, cutoff_hz / nyquist, btype='highpass', output='sos')
            pad_len = min(len(raw_position) // 4, int(self.fps * 2))
            if pad_len > 0:
                padded = np.pad(raw_position, pad_len, mode='reflect')
                filtered_padded = sosfiltfilt(sos, padded)
                filtered = filtered_padded[pad_len:-pad_len]
            else:
                filtered = sosfiltfilt(sos, raw_position)
        except Exception:
            x = np.arange(len(raw_position))
            coeffs = np.polyfit(x, raw_position, 2)
            trend = np.polyval(coeffs, x)
            filtered = raw_position - trend
        return filtered

    def _estimate_stroke_frequency(self, velocity_signal, segments):
        """Estimate dominant stroke frequency from the longest ACTIVE segment using FFT."""
        # Find the longest ACTIVE segment
        best_seg = None
        best_length = 0
        for start, end, seg_type in segments:
            if seg_type == 'ACTIVE' and (end - start) > best_length:
                best_length = end - start
                best_seg = (start, end)

        if best_seg is None or best_length < int(self.fps):
            return 1.0  # Default 1 Hz

        start, end = best_seg
        seg = velocity_signal[start:end]

        # FFT analysis
        n = len(seg)
        fft_vals = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)

        # Only consider frequencies in the stroke range (0.3 - 10 Hz)
        mask = (freqs >= 0.3) & (freqs <= 10.0)
        if not np.any(mask):
            return 1.0

        fft_masked = fft_vals[mask]
        freqs_masked = freqs[mask]

        dominant_freq = freqs_masked[np.argmax(fft_masked)]
        return float(dominant_freq)

    def normalize_per_segment(self, position_signal, segments):
        """
        Per-segment normalization with improved noise guard:
        ACTIVE → normalized to full 0-100 (with minimum range validation)
        TRANSITION → normalized to 10-90
        QUIET → hold last active value
        """
        normalized = np.full(len(position_signal), 50.0)
        last_active_val = 50.0

        # Compute median range of ACTIVE segments for noise guard
        # Only include segments longer than 1 second to avoid noise skewing the median
        min_segment_frames = int(self.fps)
        active_ranges = []
        for start, end, seg_type in segments:
            end = min(end, len(position_signal))
            if start >= end or seg_type != 'ACTIVE':
                continue
            if (end - start) < min_segment_frames:
                continue
            seg = position_signal[start:end]
            seg_range = np.max(seg) - np.min(seg)
            if seg_range > 1e-5:
                active_ranges.append(seg_range)

        median_range = np.median(active_ranges) if active_ranges else 1.0

        for start, end, seg_type in segments:
            end = min(end, len(position_signal))
            if start >= end:
                continue

            seg = position_signal[start:end]

            if seg_type == 'ACTIVE':
                seg_min = np.min(seg)
                seg_max = np.max(seg)
                seg_range = seg_max - seg_min

                if seg_range < 1e-5:
                    normalized[start:end] = last_active_val
                elif seg_range < median_range * 0.03:
                    # Noise guard: range extremely small → likely noise, hold value
                    normalized[start:end] = last_active_val
                else:
                    normalized[start:end] = ((seg - seg_min) / seg_range) * 100.0

                last_active_val = normalized[end - 1]

            elif seg_type == 'TRANSITION':
                seg_min = np.min(seg)
                seg_max = np.max(seg)
                seg_range = seg_max - seg_min

                if seg_range < 1e-5:
                    normalized[start:end] = last_active_val
                else:
                    normalized[start:end] = 10.0 + ((seg - seg_min) / seg_range) * 80.0

                last_active_val = normalized[end - 1]

            else:  # QUIET
                normalized[start:end] = last_active_val

        return normalized

    def expand_contrast(self, position_signal, strength=2.0, segments=None):
        """
        Adaptive contrast expansion per segment.
        Segments with already good range get mild expansion;
        segments with narrow range get stronger expansion.
        """
        if segments is None:
            # Fallback: uniform expansion
            centered = (position_signal - 50.0) / 50.0
            expanded = np.tanh(strength * centered) / np.tanh(strength)
            result = 50.0 + expanded * 50.0
            return np.clip(result, 0, 100)

        result = position_signal.copy()

        for start, end, seg_type in segments:
            end = min(end, len(position_signal))
            if start >= end:
                continue

            if seg_type == 'QUIET':
                continue

            seg = position_signal[start:end]

            # Determine strength based on current range utilization
            p5, p95 = np.percentile(seg, 5), np.percentile(seg, 95)
            range_used = p95 - p5

            if range_used > 70:
                local_strength = 1.0   # Already wide → mild push
            elif range_used > 40:
                local_strength = 2.0   # Medium → moderate push
            else:
                local_strength = 2.5   # Narrow → strong push

            centered = (seg - 50.0) / 50.0
            if abs(np.tanh(local_strength)) > 1e-10:
                expanded = np.tanh(local_strength * centered) / np.tanh(local_strength)
            else:
                expanded = centered
            result[start:end] = np.clip(50.0 + expanded * 50.0, 0, 100)

        return result


class ActionPointGenerator:
    """
    Peak/trough-primary action point generator.
    Human funscripts are primarily composed of local peaks and troughs,
    NOT interval-sampled intermediate positions. This approach:
    1. Detects ALL peaks and troughs in ACTIVE segments as primary actions
    2. Fills large time gaps with intermediate samples
    3. Skips QUIET zones entirely
    Result: naturally bimodal position distribution matching human scripts.
    """

    def __init__(self, fps):
        self.fps = fps

    def generate(self, position_signal, segments):
        """Generate funscript action points using peak/trough-primary strategy."""
        if len(position_signal) < 2:
            return []

        actions = []

        for start, end, seg_type in segments:
            end = min(end, len(position_signal))
            if start >= end:
                continue

            if seg_type == 'QUIET':
                continue  # No actions in quiet zones

            seg = position_signal[start:end]

            if seg_type == 'TRANSITION':
                # Sparse sampling for transitions
                trans_actions = self._sample_transition(seg, start)
                actions.extend(trans_actions)
            else:  # ACTIVE
                # Peak/trough-primary sampling
                active_actions = self._sample_active(seg, start)
                actions.extend(active_actions)

        if not actions:
            return []

        actions.sort(key=lambda x: x['at'])

        # Enforce minimum spacing (20ms allows up to 50 actions/sec for fast strokes)
        actions = self._enforce_min_spacing(actions, min_gap_ms=20)

        # Deduplicate
        actions = self._deduplicate(actions)

        return actions

    def _detect_stroke_frequency(self, seg):
        """Detect dominant stroke frequency in a segment using FFT."""
        if len(seg) < int(self.fps):
            return 1.0

        n = len(seg)
        fft_vals = np.abs(np.fft.rfft(seg - np.mean(seg)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)

        mask = (freqs >= 0.3) & (freqs <= 10.0)
        if not np.any(mask):
            return 1.0

        fft_masked = fft_vals[mask]
        freqs_masked = freqs[mask]

        return float(freqs_masked[np.argmax(fft_masked)])

    def _sample_active(self, seg, global_offset):
        """
        Frequency-adaptive strategy for ACTIVE segments:
        1. Analyze stroke frequency to determine detection parameters
        2. Find all peaks and troughs as PRIMARY actions
        3. Refine to true extrema
        4. Fill gaps with frequency-appropriate spacing
        """
        if len(seg) < 3:
            return [self._make_action(global_offset, seg[0])]

        # Detect stroke frequency for adaptive parameters
        stroke_freq = self._detect_stroke_frequency(seg)

        if stroke_freq > 3.0:      # Fast strokes
            min_dist = max(2, int(self.fps / 16))
            prominence = 5.0
            max_gap_ms = 100
        elif stroke_freq > 1.0:    # Medium strokes
            min_dist = max(2, int(self.fps / 10))
            prominence = 8.0
            max_gap_ms = 180
        else:                       # Slow strokes
            min_dist = max(2, int(self.fps / 8))
            prominence = 10.0
            max_gap_ms = 300

        # Adaptive smoothing window: wider for slow strokes, narrower for fast
        # Window = ~1/4 of one stroke period, ensures peaks aren't smoothed away
        period_frames = self.fps / max(stroke_freq, 0.5)
        smooth_window = max(5, int(period_frames / 4))
        smooth_window = smooth_window | 1  # Ensure odd
        if smooth_window >= len(seg):
            smooth_window = max(3, (len(seg) - 1) | 1)

        if len(seg) > smooth_window:
            seg_smooth = savgol_filter(seg, smooth_window, min(2, smooth_window - 1))
        else:
            seg_smooth = seg

        peaks, _ = find_peaks(seg_smooth, distance=min_dist, prominence=prominence)
        troughs, _ = find_peaks(-seg_smooth, distance=min_dist, prominence=prominence)

        # Collect all peak/trough indices as primary action points
        primary_indices = set()
        primary_indices.add(0)
        primary_indices.add(len(seg) - 1)

        for idx in peaks:
            primary_indices.add(idx)
        for idx in troughs:
            primary_indices.add(idx)

        # Refine to true extrema in original signal
        refined_indices = set()
        for idx in primary_indices:
            search_start = max(0, idx - 3)
            search_end = min(len(seg), idx + 4)
            local_seg = seg[search_start:search_end]

            if len(local_seg) == 0:
                refined_indices.add(idx)
                continue

            if seg[idx] >= np.median(seg):
                best = search_start + np.argmax(local_seg)
            else:
                best = search_start + np.argmin(local_seg)

            refined_indices.add(best)

        sorted_indices = sorted(refined_indices)

        actions = []
        for idx in sorted_indices:
            actions.append(self._make_action(global_offset + idx, seg[idx]))

        # Fill gaps with frequency-appropriate max gap
        actions = self._fill_gaps(actions, seg, global_offset, max_gap_ms)

        return actions

    def _fill_gaps(self, actions, seg, global_offset, max_gap_ms):
        """Fill time gaps larger than max_gap_ms with intermediate action points."""
        if len(actions) < 2:
            return actions

        filled = [actions[0]]
        for i in range(1, len(actions)):
            prev = filled[-1]
            curr = actions[i]
            gap_ms = curr['at'] - prev['at']

            if gap_ms > max_gap_ms:
                # Add intermediate points
                n_intermediates = int(gap_ms / max_gap_ms)
                prev_frame = int(round(prev['at'] * self.fps / 1000.0)) - global_offset
                curr_frame = int(round(curr['at'] * self.fps / 1000.0)) - global_offset

                for j in range(1, n_intermediates + 1):
                    frac = j / (n_intermediates + 1)
                    frame_idx = int(prev_frame + frac * (curr_frame - prev_frame))
                    frame_idx = np.clip(frame_idx, 0, len(seg) - 1)
                    filled.append(self._make_action(global_offset + frame_idx, seg[frame_idx]))

            filled.append(curr)

        return filled

    def _sample_transition(self, seg, global_offset):
        """Sparse sampling for TRANSITION segments."""
        if len(seg) < 2:
            return [self._make_action(global_offset, seg[0])] if len(seg) > 0 else []

        actions = [self._make_action(global_offset, seg[0])]

        # Sample every ~500ms
        step = max(1, int(self.fps / 2))
        for i in range(step, len(seg) - 1, step):
            actions.append(self._make_action(global_offset + i, seg[i]))

        actions.append(self._make_action(global_offset + len(seg) - 1, seg[-1]))
        return actions

    def _make_action(self, frame_idx, pos_value):
        """Create a single action point dict."""
        return {
            'at': int(round(frame_idx * 1000.0 / self.fps)),
            'pos': int(np.clip(np.round(pos_value), 0, 100))
        }

    def _enforce_min_spacing(self, actions, min_gap_ms=33):
        """Enforce minimum time between consecutive actions."""
        if len(actions) < 2:
            return actions

        actions.sort(key=lambda x: x['at'])
        filtered = [actions[0]]

        for a in actions[1:]:
            gap = a['at'] - filtered[-1]['at']
            if gap >= min_gap_ms:
                filtered.append(a)
            elif abs(a['pos'] - filtered[-1]['pos']) > 15:
                filtered.append(a)

        return filtered

    def _deduplicate(self, actions):
        """Remove duplicate timestamps, keeping the last occurrence."""
        if not actions:
            return actions

        seen = {}
        for a in actions:
            seen[a['at']] = a

        result = sorted(seen.values(), key=lambda x: x['at'])

        # Remove consecutive actions with identical position
        if len(result) < 3:
            return result

        filtered = [result[0]]
        for i in range(1, len(result) - 1):
            if result[i]['pos'] == filtered[-1]['pos'] == result[i + 1]['pos']:
                continue
            filtered.append(result[i])
        filtered.append(result[-1])

        return filtered


class ScriptPostProcessor:
    """Physics validation for generated scripts."""

    def validate_and_fix(self, actions, max_speed=500):
        """
        Apply physics constraints:
        - Position clamped to 0-100
        - Maximum speed limit
        - Interpolation for large jumps
        """
        if not actions:
            return actions

        fixed = []

        for i, action in enumerate(actions):
            action['pos'] = int(np.clip(action['pos'], 0, 100))

            if i == 0:
                fixed.append(action)
                continue

            prev = fixed[-1]
            dt = (action['at'] - prev['at']) / 1000.0

            if dt <= 0:
                continue

            dp = action['pos'] - prev['pos']
            speed = abs(dp / dt) if dt > 0 else 0

            if speed > max_speed:
                direction = 1 if dp > 0 else -1
                action['pos'] = int(np.clip(
                    prev['pos'] + direction * max_speed * dt,
                    0, 100
                ))

            fixed.append(action)

        return fixed


class MotionExtractor:
    """Extracts vertical motion signal from video using optical flow in ROI."""

    def __init__(self, flow_estimator):
        self.flow_estimator = flow_estimator

    def extract_velocity_signal(self, prev_frame, curr_frame, roi_fractions, precomputed_flow=None):
        """
        Extract signed vertical flow in ROI with global motion compensation
        and scale normalization.
        roi_fractions: (y_start_frac, y_end_frac, x_start_frac, x_end_frac)
        precomputed_flow: optional pre-computed optical flow tensor (2, H, W)
        Returns: (velocity, magnitude, roi_height_px)
          - velocity is scale-normalized and negated so positive = upward in funscript
          - magnitude is scale-normalized for quiet zone detection
          - roi_height_px is the ROI height in pixels (for diagnostics)
        """
        flow = precomputed_flow if precomputed_flow is not None else \
            self.flow_estimator.estimate_flow(prev_frame, curr_frame)

        h, w = prev_frame.shape[:2]
        y1 = int(roi_fractions[0] * h)
        y2 = int(roi_fractions[1] * h)
        x1 = int(roi_fractions[2] * w)
        x2 = int(roi_fractions[3] * w)

        roi_height_px = max(y2 - y1, 1)

        # Background-only Global Motion Compensation:
        # Estimate camera motion from pixels OUTSIDE the ROI only.
        # When ROI covers most of the frame (e.g. subject fills frame),
        # there's no background → skip GMC to preserve the signal.
        bg_mask = torch.ones(flow.shape[1], flow.shape[2], dtype=torch.bool, device=flow.device)
        bg_mask[y1:y2, x1:x2] = False

        if bg_mask.sum() > 100:
            global_flow_y = flow[1][bg_mask].median().item()
        else:
            global_flow_y = 0.0

        flow_roi = flow[:, y1:y2, x1:x2]
        # Subtract background-derived camera motion from ROI vertical flow
        flow_y = flow_roi[1] - global_flow_y

        # Compute magnitude for thresholding (using compensated flow)
        flow_x = flow_roi[0]
        flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2)

        # Only consider pixels with meaningful motion
        mag_threshold = 0.3
        mask = flow_mag > mag_threshold

        if mask.sum() < 10:
            return 0.0, 0.0, roi_height_px

        # Negate: RAFT positive flow_y = downward, but funscript pos 100 = up (insert)
        velocity_raw = -flow_y[mask].mean().item()
        magnitude_raw = flow_mag[mask].mean().item()

        # Scale normalization: divide by ROI height so that the same real-world
        # motion produces the same normalized value regardless of zoom level.
        # A reference height of 256px (half of 512 resize width) means no scaling
        # when the ROI covers ~half the frame height.
        reference_height = h * 0.5
        scale_factor = reference_height / roi_height_px

        velocity = velocity_raw * scale_factor
        magnitude = magnitude_raw * scale_factor

        return velocity, magnitude, roi_height_px

    def extract_frame_histogram(self, frame):
        """Extract grayscale histogram for scene change detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
