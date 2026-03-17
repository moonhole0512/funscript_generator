import math
import numpy as np
from scipy.signal import savgol_filter, butter, sosfiltfilt
from config_manager import config

class OneEuroFilter:
    """
    1€ Filter (Casiez et al. 2012): 적응형 저역통과 필터.
    저속 → 높은 스무딩 (jitter 억제)
    고속 → 낮은 스무딩 (빠른 움직임 지연 없이 추적)
    """
    def __init__(self, freq: float, min_cutoff: float = 1.5,
                 beta: float = 0.5, d_cutoff: float = 1.0):
        self._freq       = max(float(freq), 1.0)
        self._min_cutoff = min_cutoff  # Hz — 저속 스무딩 강도
        self._beta       = beta        # 고속 적응 계수
        self._d_cutoff   = d_cutoff    # 미분 필터 컷오프
        self._x_prev     = None
        self._dx_prev    = 0.0

    def _alpha(self, cutoff: float) -> float:
        te  = 1.0 / self._freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev  = x
            return x
        dx    = (x - self._x_prev) * self._freq
        a_d   = self._alpha(self._d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a      = self._alpha(cutoff)
        x_hat  = a * x + (1.0 - a) * self._x_prev
        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self):
        self._x_prev  = None
        self._dx_prev = 0.0

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

        # Single scene: filter the whole signal (with mean-subtraction to remove drift)
        vel_demeaned = velocity_signal - np.mean(velocity_signal)
        raw_position = np.cumsum(vel_demeaned)
        return self._apply_hpf(raw_position, cutoff_hz, nyquist)

    def _filter_segment(self, velocity_segment, cutoff_hz, nyquist):
        """Filter a single scene's velocity to drift-free position."""
        if len(velocity_segment) < 10:
            return np.zeros(len(velocity_segment))

        # Apply stabilization: zero out first few frames after scene change
        stabilize_frames = min(int(self.fps * 0.3), len(velocity_segment) // 4)
        velocity_segment = velocity_segment.copy()
        if stabilize_frames > 0:
            velocity_segment[:stabilize_frames] = 0.0

        # Mean-subtract velocity to remove net drift before integration.
        active = velocity_segment[stabilize_frames:]
        if len(active) > 0:
            velocity_segment[stabilize_frames:] = active - np.mean(active)

        raw_position = np.cumsum(velocity_segment)
        return self._apply_hpf(raw_position, cutoff_hz, nyquist)

    def _apply_hpf(self, raw_position, cutoff_hz, nyquist):
        """Apply Butterworth HPF with mirror padding."""
        try:
            sos = butter(4, cutoff_hz / nyquist, btype='highpass', output='sos')
            pad_len = min(len(raw_position) // 3, int(self.fps * 3))
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
        Per-segment normalization with improved noise guard.
        """
        normalized = np.full(len(position_signal), 50.0)
        last_active_val = 50.0

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

        p_low = config.get("normalization", "percentile_low", 2)
        p_high = config.get("normalization", "percentile_high", 98)

        for start, end, seg_type in segments:
            end = min(end, len(position_signal))
            if start >= end:
                continue

            seg = position_signal[start:end]

            if seg_type == 'ACTIVE':
                seg_range = np.max(seg) - np.min(seg)

                if seg_range < 1e-5:
                    normalized[start:end] = last_active_val
                elif seg_range < median_range * 0.03:
                    if last_active_val > 75 or last_active_val < 25:
                        n = end - start
                        normalized[start:end] = np.linspace(last_active_val, 50.0, n)
                    else:
                        normalized[start:end] = last_active_val
                else:
                    p2 = np.percentile(seg, p_low)
                    p98 = np.percentile(seg, p_high)
                    p_range = p98 - p2
                    if p_range < 1e-5:
                        p2 = np.min(seg)
                        p_range = seg_range

                    amplitude_ratio = seg_range / median_range if median_range > 1e-5 else 1.0

                    if amplitude_ratio < 0.4:
                        output_range = max(20.0, 100.0 * amplitude_ratio)
                        output_center = 50.0
                        stretched = (
                            (seg - p2) / p_range * output_range
                            + (output_center - output_range / 2.0)
                        )
                    else:
                        stretched = (seg - p2) / p_range * 100.0

                    normalized[start:end] = np.clip(stretched, 0, 100)

                last_active_val = normalized[end - 1]

            elif seg_type == 'TRANSITION':
                seg_range = np.max(seg) - np.min(seg)

                if seg_range < 1e-5:
                    normalized[start:end] = last_active_val
                else:
                    p5 = np.percentile(seg, 5)
                    p95 = np.percentile(seg, 95)
                    p_range = p95 - p5
                    if p_range < 1e-5:
                        p5 = np.min(seg)
                        p_range = seg_range
                    stretched = (seg - p5) / p_range * 80.0 + 10.0
                    normalized[start:end] = np.clip(stretched, 0, 100)

                last_active_val = normalized[end - 1]

            else:  # QUIET
                normalized[start:end] = last_active_val

        return normalized

    def expand_contrast(self, position_signal, strength=2.0, segments=None):
        """Adaptive contrast expansion per segment."""
        if segments is None:
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

            p5, p95 = np.percentile(seg, 5), np.percentile(seg, 95)
            range_used = p95 - p5

            if range_used < 20 and (p5 > 65 or p95 < 35):
                local_strength = 0.5
            elif range_used > 70:
                local_strength = 2.0
            elif range_used > 40:
                local_strength = 3.0
            else:
                local_strength = 4.0

            centered = (seg - 50.0) / 50.0
            if abs(np.tanh(local_strength)) > 1e-10:
                expanded = np.tanh(local_strength * centered) / np.tanh(local_strength)
            else:
                expanded = centered
            result[start:end] = np.clip(50.0 + expanded * 50.0, 0, 100)

        return result
