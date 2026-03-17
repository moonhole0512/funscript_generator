import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d
from config_manager import config
try:
    from tracking import RESIZE_WIDTH
except ImportError:
    RESIZE_WIDTH = 512

class QuickSceneDetector:
    """
    Pass 1: Lightweight histogram-based scene boundary detection.
    """
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def detect(self, video_path, sample_interval=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            cap.release()
            return [(0, total_frames)]

        change_frames = [0]
        prev_hist = None
        prev_was_flash = False

        for frame_idx in range(0, total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = float(np.mean(gray))
            is_flash = avg_brightness < 20 or avg_brightness > 235

            if prev_was_flash and not is_flash:
                if frame_idx not in change_frames:
                    change_frames.append(frame_idx)
                prev_hist = None
            prev_was_flash = is_flash

            if not is_flash and frame_idx % sample_interval == 0:
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                if prev_hist is not None:
                    score = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                    if score < self.threshold:
                        if frame_idx not in change_frames:
                            change_frames.append(frame_idx)

                prev_hist = hist

        cap.release()

        scenes = []
        for i in range(len(change_frames)):
            start = change_frames[i]
            end = change_frames[i + 1] if i + 1 < len(change_frames) else total_frames
            if end - start >= 10:
                scenes.append((start, end))

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

class SceneBoundaryHandler:
    """
    장면 전환 경계의 velocity spike를 Hanning window로 제거한다.
    """
    def smooth_at_boundaries(self, velocity_signal, scene_boundaries, fps,
                             post_cut_suppress_frames=30):
        transition_frames = max(3, int(fps * 0.3))
        result = velocity_signal.copy()

        for scene_start, _ in scene_boundaries:
            if scene_start == 0:
                continue

            if scene_start < len(result):
                result[scene_start] = 0.0

            t_start = max(0, scene_start - transition_frames)
            t_end = scene_start
            n = t_end - t_start
            if n > 1:
                window = np.hanning(n * 2)[n:]
                result[t_start:t_end] *= window

            suppress_end = min(len(result), scene_start + post_cut_suppress_frames)
            fade_end = min(len(result), scene_start + transition_frames + 1)

            if post_cut_suppress_frames > transition_frames:
                t_start = scene_start + 1
                n = fade_end - t_start
                if n > 1:
                    window = np.hanning(n * 2)[:n]
                    result[t_start:fade_end] *= window
                if fade_end < suppress_end:
                    result[fade_end:suppress_end] = 0.0
            else:
                t_start = scene_start + 1
                n = fade_end - t_start
                if n > 1:
                    window = np.hanning(n * 2)[:n]
                    result[t_start:fade_end] *= window

        return result

class SceneSegmenter:
    """Detects scene changes and classifies segments as ACTIVE / QUIET / TRANSITION."""

    def __init__(self, fps):
        self.fps = fps
        self.prev_hist = None
        self.scene_change_threshold = 0.7

    def detect_scene_changes(self, frames_gray):
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

    def classify_segments(self, motion_magnitudes, scene_changes, total_frames,
                          velocity_signal=None, yolo_confs=None):
        boundaries = sorted(set(scene_changes + [total_frames]))
        if boundaries[0] != 0:
            boundaries = [0] + boundaries

        segments = []
        min_quiet_frames_s = config.get("silence", "min_quiet_frames_s", 2.0)
        min_quiet_frames = int(self.fps * min_quiet_frames_s)
        
        vel_std_thresh = config.get("silence", "vel_std_threshold", 0.012)
        vel_mean_thresh = config.get("silence", "vel_mean_threshold", 0.020)
        yolo_conf_thresh = config.get("silence", "yolo_conf_threshold", 0.3)
        det_ratio_min = config.get("silence", "det_ratio_min", 0.15)

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            if end - start < 2:
                continue

            seg_motion = motion_magnitudes[start:end]
            local_threshold = self._compute_local_threshold(seg_motion)

            avg_motion = np.mean(np.abs(seg_motion))
            max_motion = np.max(np.abs(seg_motion)) if len(seg_motion) > 0 else 0

            if avg_motion < local_threshold and max_motion < local_threshold * 5:
                seg_type = 'QUIET'
            elif (velocity_signal is not None and
                  np.std(velocity_signal[start:end]) < vel_std_thresh and
                  np.mean(np.abs(velocity_signal[start:end])) < vel_mean_thresh):
                seg_type = 'QUIET'
            elif end - start < min_quiet_frames:
                seg_type = 'TRANSITION'
            else:
                seg_type = 'ACTIVE'

            # 리드미컬하지 않은 움직임 걸러내기
            if seg_type == 'ACTIVE' and not self._check_frequency_consistency(seg_motion):
                seg_type = 'TRANSITION'

            if seg_type == 'ACTIVE' and yolo_confs is not None and end > start:
                seg_c = [c for c in yolo_confs[start:end] if c is not None]
                if len(seg_c) >= 5:
                    det_ratio = sum(1 for c in seg_c if c > yolo_conf_thresh) / len(seg_c)
                    if det_ratio < det_ratio_min:
                        seg_type = 'QUIET'

            segments.append((start, end, seg_type))

        total_duration = sum(e - s for s, e, _ in segments)
        quiet_duration = sum(e - s for s, e, t in segments if t == 'QUIET')
        vel_var_thresh = config.get("silence", "vel_variance_threshold", 0.005)
        vel_variance = np.var(motion_magnitudes) if velocity_signal is None else np.var(velocity_signal)
        if total_duration > 0 and quiet_duration / total_duration > 0.70 and vel_variance > vel_var_thresh:
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

        refined = []
        for start, end, seg_type in segments:
            if seg_type == 'ACTIVE':
                local_threshold = self._compute_local_threshold(motion_magnitudes[start:end])
                sub_segs = self._split_active_at_quiet(motion_magnitudes, start, end, local_threshold, min_quiet_frames)
                refined.extend(sub_segs)
            else:
                refined.append((start, end, seg_type))

        if velocity_signal is not None:
            vel_checked = []
            for s, e, t in refined:
                if t == 'ACTIVE' and (e - s) >= min_quiet_frames:
                    vseg = velocity_signal[s:e]
                    if (np.std(vseg) < vel_std_thresh and
                            np.mean(np.abs(vseg)) < vel_mean_thresh):
                        t = 'QUIET'
                vel_checked.append((s, e, t))
            refined = vel_checked

        intro_frames_s = config.get("silence", "intro_frames_s", 3.0)
        intro_frames = int(self.fps * intro_frames_s)
        verified = []
        for start, end, seg_type in refined:
            if seg_type == 'ACTIVE' and (end - start) >= int(self.fps):
                seg_motion = motion_magnitudes[start:end]
                seg_vel = velocity_signal[start:end] if velocity_signal is not None else None
                if not self._is_rhythmic_motion(seg_motion, seg_vel):
                    seg_type = 'TRANSITION'
                elif start < intro_frames and (end - start) < int(self.fps * 2):
                    seg_type = 'TRANSITION'
            verified.append((start, end, seg_type))

        return verified

    def _compute_local_threshold(self, seg_motion):
        abs_motion = np.abs(seg_motion)
        nonzero = abs_motion[abs_motion > 0.01]
        if len(nonzero) < 10:
            return 0.05
        return np.percentile(nonzero, 15)

    def _check_frequency_consistency(self, seg_motion):
        """리드미컬하지 않은 움직임을 걸러내기 위한 주파수 일관성 체크."""
        if len(seg_motion) < int(self.fps):
            return True
        sig = np.array(seg_motion, dtype=float)
        sig = sig - np.mean(sig)
        n = len(sig)
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        action_band = (freqs >= 0.5) & (freqs <= 8.0)
        total_energy = np.sum(fft_vals ** 2)
        if total_energy < 1e-10:
            return False
        action_energy = np.sum(fft_vals[action_band] ** 2)
        ratio = action_energy / total_energy
        return ratio > 0.20

    def _is_rhythmic_motion(self, seg_motion, seg_velocity=None):
        if len(seg_motion) < int(self.fps):
            return True
        seg_abs = np.abs(seg_motion)
        if np.mean(seg_abs) < 0.015 and np.max(seg_abs) < 0.05:
            return False
        if seg_velocity is not None and len(seg_velocity) >= int(self.fps):
            vel = np.array(seg_velocity, dtype=float)
            vel_centered = vel - np.mean(vel)
            sign_pos = np.sum(vel_centered > 0) / max(1, len(vel_centered))
            vel_zc = np.sum(np.diff(np.sign(vel_centered)) != 0)
            vel_zc_rate = vel_zc / (len(vel_centered) / self.fps)
            if (sign_pos > 0.85 or sign_pos < 0.15) and vel_zc_rate < 0.5:
                return False
        sig = np.array(seg_motion, dtype=float)
        sig = sig - np.mean(sig)
        n = len(sig)
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        action_band = (freqs >= 0.5) & (freqs <= 8.0)
        total_energy = np.sum(fft_vals ** 2)
        if total_energy < 1e-10:
            return False
        action_energy = np.sum(fft_vals[action_band] ** 2)
        ratio = action_energy / total_energy
        zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
        duration_sec = n / self.fps
        zc_rate = zero_crossings / duration_sec if duration_sec > 0 else 0
        return ratio > 0.25 or zc_rate > 1.0

    def _split_active_at_quiet(self, motion_magnitudes, start, end, threshold, min_frames):
        seg_motion = np.abs(motion_magnitudes[start:end])
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
        abs_start = start + current_start
        abs_end = end
        if abs_end - abs_start >= 2:
            segments.append((abs_start, abs_end, current_type))
        merged = []
        for seg in segments:
            s, e, t = seg
            if e - s < min_frames and t == 'QUIET' and merged:
                prev_s, prev_e, prev_t = merged[-1]
                merged[-1] = (prev_s, e, prev_t)
            else:
                merged.append(seg)
        return merged if merged else [(start, end, 'ACTIVE')]

class SceneTypeDetector:
    """P1/P2 키포인트 상대 거리를 분석하여 장면 유형을 자동 감지."""
    STABLE_SEC = 5.0
    BJ_RATIO   = 0.65
    HIP_CLOSE  = 0.35

    def __init__(self, fps: float):
        self._fps = max(fps, 1.0)
        self._current_type  = 'OPTICAL_FLOW'
        self._candidate     = 'OPTICAL_FLOW'
        self._cand_frames   = 0
        self._stable_frames = int(self._fps * self.STABLE_SEC)

    def update(self, p1_kp, p2_kp, p2_hip_y, frame_h: int) -> str:
        candidate = self._classify(p1_kp, p2_kp, p2_hip_y, frame_h)
        if candidate == self._candidate:
            self._cand_frames += 1
        else:
            self._candidate   = candidate
            self._cand_frames = 1
        if self._cand_frames >= self._stable_frames:
            self._current_type = self._candidate
        return self._current_type

    def _classify(self, p1_kp, p2_kp, p2_hip_y, frame_h) -> str:
        if p2_kp is None and p2_hip_y is None:
            return 'POV'
        if p1_kp is None or p2_kp is None:
            return 'OPTICAL_FLOW'

        def _hip_center(kp):
            lhx, lhy, lhc = kp[11]; rhx, rhy, rhc = kp[12]
            if lhc > 0.25 and rhc > 0.25:
                return ((lhx + rhx) / 2, (lhy + rhy) / 2)
            elif lhc > 0.25:
                return (lhx, lhy)
            elif rhc > 0.25:
                return (rhx, rhy)
            return None

        def _head(kp):
            for ki in [0, 1, 2, 3, 4]:
                if ki < kp.shape[0] and kp[ki][2] > 0.20:
                    return (kp[ki][0], kp[ki][1])
            return None

        def _dist(a, b):
            return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

        p1_hip  = _hip_center(p1_kp)
        p2_hip  = _hip_center(p2_kp)
        if p1_hip is None or p2_hip is None:
            return 'OPTICAL_FLOW'

        hh_dist = _dist(p1_hip, p2_hip) / max(frame_h, 1)
        p1_head = _head(p1_kp)
        p2_head = _head(p2_kp)
        if p1_head is not None:
            h1_dist = _dist(p1_head, p2_hip) / max(frame_h, 1)
            if h1_dist < hh_dist * self.BJ_RATIO:
                return 'BJ'
        if p2_head is not None:
            h2_dist = _dist(p2_head, p1_hip) / max(frame_h, 1)
            if h2_dist < hh_dist * self.BJ_RATIO:
                return 'BJ'

        if hh_dist < self.HIP_CLOSE:
            return 'HIP_HIP'
        return 'OPTICAL_FLOW'

    def reset(self):
        self._current_type = 'OPTICAL_FLOW'
        self._candidate    = 'OPTICAL_FLOW'
        self._cand_frames  = 0

class SceneAnchorSelector:
    """씬 시작 N프레임 YOLO 샘플링 → P1/P2 골반 중심 픽셀 앵커 확정."""
    def select(self, video_path: str, scene_start: int, scene_end: int,
               yolo_tracker, n_samples: int = 10,
               resize_w: int = RESIZE_WIDTH) -> dict:
        scene_len = max(scene_end - scene_start, 1)
        step = max(1, scene_len // n_samples)
        sample_frames = list(range(scene_start, scene_end, step))[:n_samples]

        cap = cv2.VideoCapture(video_path)
        frame_h, frame_w = 0, 0
        p1_xs, p1_ys, p2_xs, p2_ys = [], [], [], []

        for fi in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            h_orig, w_orig = frame.shape[:2]
            frame_resized = cv2.resize(frame, (resize_w, int(h_orig * resize_w / w_orig)))
            frame_h, frame_w = frame_resized.shape[:2]
            persons = yolo_tracker.detect_persons_stateless(frame_resized)
            if len(persons) >= 1 and persons[0]['hip_px'] is not None:
                hx, hy = persons[0]['hip_px']
                p1_xs.append(hx)
                p1_ys.append(hy)
            if len(persons) >= 2:
                p2 = persons[1]
                if p2['bbox_area'] >= persons[0]['bbox_area'] * 0.25:
                    p1_bbox = persons[0]['bbox']
                    bw, bh = p1_bbox[2] - p1_bbox[0], p1_bbox[3] - p1_bbox[1]
                    ex1, ex2 = p1_bbox[0] - bw, p1_bbox[2] + bw
                    ey1, ey2 = p1_bbox[1] - bh, p1_bbox[3] + bh
                    if p2['bbox'] is not None:
                        p2cx = (p2['bbox'][0] + p2['bbox'][2]) / 2.0
                        p2cy = (p2['bbox'][1] + p2['bbox'][3]) / 2.0
                        if ex1 <= p2cx <= ex2 and ey1 <= p2cy <= ey2:
                            if p2['hip_px'] is not None:
                                hx2, hy2 = p2['hip_px']
                                p2_xs.append(hx2)
                                p2_ys.append(hy2)
        cap.release()
        n_valid_p1, n_valid_p2 = len(p1_xs), len(p2_xs)
        base_conf = min(n_valid_p1, n_valid_p2) / max(len(sample_frames), 1)
        if n_valid_p1 >= 3 and n_valid_p2 >= 3 and frame_h > 0:
            consistency = 1.0 - min(1.0, max(float(np.std(p1_ys)), float(np.std(p2_ys))) / (frame_h * 0.15 + 1e-6))
        else:
            consistency = 1.0
        confidence = base_conf * max(consistency, 0.0)
        p1_hip_px = (self._robust_median(p1_xs), self._robust_median(p1_ys)) if n_valid_p1 >= 3 else None
        p2_hip_px = (self._robust_median(p2_xs), self._robust_median(p2_ys)) if n_valid_p2 >= 3 else None
        return {
            'p1_hip_px' : p1_hip_px,
            'p2_hip_px' : p2_hip_px,
            'frame_h'   : frame_h,
            'frame_w'   : frame_w,
            'is_dual'   : (p1_hip_px is not None and p2_hip_px is not None),
            'confidence': confidence,
        }

    @staticmethod
    def _robust_median(values):
        if len(values) < 3:
            return float(np.median(values))
        arr = np.array(values, dtype=np.float64)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        filtered = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
        return float(np.median(filtered)) if len(filtered) >= 2 else float(np.median(arr))
