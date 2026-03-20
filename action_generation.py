import numpy as np
from scipy.signal import find_peaks, savgol_filter
from config_manager import config

class ActionPointGenerator:
    """
    Peak/trough-primary action point generator.
    """

    def __init__(self, fps):
        self.fps = fps

    def generate(self, position_signal, segments, velocity_signal=None):
        """Generate funscript action points using peak/trough-primary strategy."""
        if len(position_signal) < 2:
            return []

        actions = []

        for start, end, seg_type in segments:
            end = min(end, len(position_signal))
            if start >= end:
                continue

            if seg_type == 'QUIET':
                continue

            seg = position_signal[start:end]

            if seg_type == 'TRANSITION':
                trans_actions = self._sample_transition(seg, start)
                actions.extend(trans_actions)
            else:  # ACTIVE
                vel_seg = velocity_signal[start:end] if velocity_signal is not None else None
                active_actions = self._sample_active(seg, start, vel_seg)
                
                # P5: Rhythm Regularization (Disabled for personal fidelity)
                # stroke_freq = self._detect_stroke_frequency(seg)
                # active_actions = self._regularize_rhythm(active_actions, stroke_freq)
                
                actions.extend(active_actions)

        if not actions:
            return []

        actions.sort(key=lambda x: x['at'])
        actions = self._enforce_min_spacing(actions, min_gap_ms=20)
        actions = self._deduplicate(actions)
        
        snap_thresh = config.get("normalization", "snap_threshold", 3)
        actions = self._snap_extremes(actions, snap_threshold=snap_thresh)

        return actions

    def _detect_stroke_frequency(self, seg):
        if len(seg) < int(self.fps):
            return 1.0

        n = len(seg)
        sig = seg - np.mean(seg)

        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        mask = (freqs >= 0.3) & (freqs <= 10.0)
        if np.any(mask):
            fft_freq = float(freqs[mask][np.argmax(fft_vals[mask])])
        else:
            fft_freq = 1.0

        zc = np.sum(np.diff(np.sign(sig)) != 0)
        duration_s = n / self.fps
        zc_freq = float(np.clip(zc / (2.0 * duration_s), 0.3, 10.0))

        return float(max(fft_freq, zc_freq))

    def _sample_active(self, seg, global_offset, vel_seg=None):
        if len(seg) < 3:
            return [self._make_action(global_offset, seg[0])]

        pos_range = float(np.max(seg) - np.min(seg))
        pos_mean = float(np.mean(seg))
        if pos_range < 8 and 38 < pos_mean < 62:
            return []

        min_dist_divisor = config.get("sampling", "min_dist_divisor", 20)
        min_dist = max(1, int(self.fps / min_dist_divisor))

        signal_range = float(np.max(seg) - np.min(seg))
        p_scale = config.get("sampling", "prominence_scale", 0.02) # Ultra sensitive
        p_min = config.get("sampling", "min_prominence", 0.3)      # Ultra sensitive
        p_max = config.get("sampling", "max_prominence", 10.0)
        prominence = max(p_min, min(p_max, signal_range * p_scale))

        stroke_freq = self._detect_stroke_frequency(seg)
        if vel_seg is not None and len(vel_seg) >= int(self.fps):
            vel_sig = vel_seg - np.mean(vel_seg)
            n_vel = len(vel_seg)
            fft_vel = np.abs(np.fft.rfft(vel_sig))
            freqs_vel = np.fft.rfftfreq(n_vel, d=1.0 / self.fps)
            mask_vel = (freqs_vel >= 0.3) & (freqs_vel <= 10.0)
            if np.any(mask_vel):
                vel_fft_freq = float(freqs_vel[mask_vel][np.argmax(fft_vel[mask_vel])])
            else:
                vel_fft_freq = 1.0
            stroke_freq = max(stroke_freq, vel_fft_freq)

        period_frames = self.fps / max(stroke_freq, 0.5)
        divisor = 8 if stroke_freq > 3.0 else 4
        smooth_window = max(3, int(period_frames / divisor)) # Increased sensitivity
        smooth_window = smooth_window | 1
        if smooth_window >= len(seg):
            smooth_window = max(3, (len(seg) - 1) | 1)

        if len(seg) > smooth_window:
            seg_smooth = savgol_filter(seg, smooth_window, min(2, smooth_window - 1))
        else:
            seg_smooth = seg

        peaks, _ = find_peaks(seg_smooth, distance=min_dist, prominence=prominence)
        troughs, _ = find_peaks(-seg_smooth, distance=min_dist, prominence=prominence)

        all_extrema = sorted(set(peaks) | set(troughs))
        stroke_period_ms = 1000.0 / max(stroke_freq, 0.3)
        freq_based_gap = max(30, min(1000, int(stroke_period_ms * 1.2)))

        dense_thresh = config.get("sampling", "dense_mode_threshold", 5.0)
        if stroke_freq > dense_thresh:
            max_gap_ms = freq_based_gap
        elif len(all_extrema) >= 2:
            extrema_intervals_ms = np.diff(all_extrema) * 1000.0 / self.fps
            median_interval_ms = float(np.median(extrema_intervals_ms))
            max_gap_ms = max(50, min(1000, int(median_interval_ms * 1.5)))
        else:
            max_gap_ms = freq_based_gap

        primary_indices = set([0, len(seg) - 1])
        for idx in peaks: primary_indices.add(idx)
        for idx in troughs: primary_indices.add(idx)

        # Bounce/Rebound detection
        _BOUNCE_TROUGH_MAX = 12
        _BOUNCE_MIN_AMP    = 25
        _bounce_look       = max(2, int(self.fps * 0.20))
        bounce_extra = []
        for t_idx in troughs:
            if seg[t_idx] > _BOUNCE_TROUGH_MAX: continue
            look = min(t_idx, _bounce_look)
            if look < 2: continue
            preceding_max = float(np.max(seg[max(0, t_idx - look):t_idx + 1]))
            if preceding_max - seg[t_idx] < _BOUNCE_MIN_AMP: continue
            trough_pos = float(seg[t_idx])
            for offset, delta in [(1, 2.5), (2, 0.5)]:
                b_idx = t_idx + offset
                if 0 < b_idx < len(seg) - 1:
                    bounce_extra.append(self._make_action(global_offset + b_idx, trough_pos + delta))

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
        actions = [self._make_action(global_offset + idx, seg[idx]) for idx in sorted_indices]
        actions = self._fill_gaps(actions, seg, global_offset, max_gap_ms)

        if bounce_extra:
            actions = sorted(actions + bounce_extra, key=lambda x: x['at'])

        return actions

    def _regularize_rhythm(self, actions, stroke_freq):
        """P5: Rhythm Regularizer - Align actions to the ideal beat grid."""
        min_freq = config.get("rhythm", "min_stroke_freq", 0.5)
        if stroke_freq < min_freq or len(actions) < 4:
            return actions

        snap_range_pct = config.get("rhythm", "snap_range", 0.15)
        beat_ms = 1000.0 / stroke_freq
        half_beat = beat_ms / 2  # Interval between peak and trough

        start_ms = actions[0]['at']
        regularized = [actions[0]]

        for action in actions[1:]:
            # Find the nearest ideal beat from the start of the segment
            ideal_ms = start_ms + round((action['at'] - start_ms) / half_beat) * half_beat
            snap_range = half_beat * snap_range_pct

            if abs(action['at'] - ideal_ms) <= snap_range:
                # Snap to beat but keep it as a new dict to avoid modifying original if reused
                new_action = action.copy()
                new_action['at'] = int(ideal_ms)
                regularized.append(new_action)
            else:
                regularized.append(action)

        return regularized

    def _fill_gaps(self, actions, seg, global_offset, max_gap_ms):
        if len(actions) < 2: return actions
        filled = [actions[0]]
        for i in range(1, len(actions)):
            prev, curr = filled[-1], actions[i]
            gap_ms = curr['at'] - prev['at']
            if gap_ms > max_gap_ms:
                n = int(gap_ms / max_gap_ms)
                prev_f = int(round(prev['at'] * self.fps / 1000.0)) - global_offset
                curr_f = int(round(curr['at'] * self.fps / 1000.0)) - global_offset
                for j in range(1, n + 1):
                    f = int(prev_f + (j / (n + 1)) * (curr_f - prev_f))
                    f = np.clip(f, 0, len(seg) - 1)
                    filled.append(self._make_action(global_offset + f, seg[f]))
            filled.append(curr)
        return filled

    def _sample_transition(self, seg, global_offset):
        if len(seg) < 2: return [self._make_action(global_offset, seg[0])] if len(seg) > 0 else []
        actions = [self._make_action(global_offset, seg[0])]
        step = max(1, int(self.fps / 2))
        for i in range(step, len(seg) - 1, step):
            actions.append(self._make_action(global_offset + i, seg[i]))
        actions.append(self._make_action(global_offset + len(seg) - 1, seg[-1]))
        return actions

    def _make_action(self, frame_idx, pos_value):
        return {
            'at': int(round(frame_idx * 1000.0 / self.fps)),
            'pos': int(np.clip(np.round(pos_value), 0, 100))
        }

    def _enforce_min_spacing(self, actions, min_gap_ms=10): # Minimal gap
        if len(actions) < 2: return actions
        actions.sort(key=lambda x: x['at'])
        filtered = [actions[0]]
        for a in actions[1:]:
            if a['at'] - filtered[-1]['at'] >= min_gap_ms:
                filtered.append(a)
            elif abs(a['pos'] - filtered[-1]['pos']) > 15:
                filtered.append(a)
        return filtered

    def _snap_extremes(self, actions, snap_threshold=3):
        if len(actions) < 4: return actions
        positions = [a['pos'] for a in actions]
        top, bot = max(positions), min(positions)
        n = len(actions)
        max_snap = max(3, int(n * 0.05))
        if top >= 100 - snap_threshold:
            cands = [a for a in actions if a['pos'] >= top - snap_threshold]
            if len(cands) <= max_snap:
                for a in cands: a['pos'] = 100
        if bot <= snap_threshold:
            cands = [a for a in actions if a['pos'] <= bot + snap_threshold]
            if len(cands) <= max_snap:
                for a in cands: a['pos'] = 0
        return actions

    def _deduplicate(self, actions):
        if not actions: return actions
        seen = {}
        for a in actions: seen[a['at']] = a
        result = sorted(seen.values(), key=lambda x: x['at'])
        if len(result) < 3: return result
        filtered = [result[0]]
        for i in range(1, len(result) - 1):
            if result[i]['pos'] == filtered[-1]['pos'] == result[i + 1]['pos']: continue
            filtered.append(result[i])
        filtered.append(result[-1])
        return filtered

class ScriptPostProcessor:
    """Physics validation for generated scripts."""
    def validate_and_fix(self, actions, max_speed=500):
        if not actions: return actions
        fixed = []
        for i, action in enumerate(actions):
            action['pos'] = int(np.clip(action['pos'], 0, 100))
            if i == 0:
                fixed.append(action)
                continue
            prev = fixed[-1]
            dt = (action['at'] - prev['at']) / 1000.0
            if dt <= 0: continue
            dp = action['pos'] - prev['pos']
            speed = abs(dp / dt)
            if speed > max_speed:
                direction = 1 if dp > 0 else -1
                action['pos'] = int(np.clip(prev['pos'] + direction * max_speed * dt, 0, 100))
            fixed.append(action)
        return fixed
