import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from scipy.signal import find_peaks, savgol_filter, butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from collections import namedtuple

MEDIAPIPE_AVAILABLE = False  # MediaPipe 제거됨 — YOLO만 사용

try:
    from ultralytics import YOLO as _YOLO_CHECK
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Internal Defaults
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RAFT_SMALL = False
RESIZE_WIDTH = 512

# Named tuple for affine camera motion estimation result
AffineResult = namedtuple('AffineResult', ['tx', 'ty', 'scale', 'rotation', 'inlier_ratio', 'bg_std', 'valid'])


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
        prev_was_flash = False

        for frame_idx in range(0, total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Flash frame detection (black/white transition cuts)
            avg_brightness = float(np.mean(gray))
            is_flash = avg_brightness < 20 or avg_brightness > 235

            if prev_was_flash and not is_flash:
                # flash → non-flash: new scene starts here
                if frame_idx not in change_frames:
                    change_frames.append(frame_idx)
                prev_hist = None  # reset histogram baseline after cut
            prev_was_flash = is_flash

            # Histogram-based detection (skip flash frames and non-sample frames)
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


class CameraMotionCompensator:
    """
    RAFT 광학 흐름 기반 카메라 이동 추정.

    이미 계산된 RAFT flow 텐서에서 배경 픽셀의 흐름 통계를 추출하여
    translation(ty)와 scale(줌)을 추정한다.
    LK나 별도 특징점 추적 없이 GPU 연산만으로 동작 → 추가 처리 시간 없음.

    보정 임계값: 잡음 수준 이하의 작은 카메라 움직임은 무시하여
    정적 카메라 영상(bonfie 등)의 신호 품질을 보호한다.
    """

    MIN_BG_PIXELS = 16      # 배경 추정에 필요한 최소 픽셀 수
    MIN_TY_PX = 0.5         # 이 이하 translation은 잡음으로 간주, 보정 안 함
    MIN_SCALE_DEV = 0.03    # 이 이하 줌 변화는 보정 안 함 (3%)

    _INVALID = AffineResult(tx=0.0, ty=0.0, scale=1.0, rotation=0.0,
                            inlier_ratio=0.0, bg_std=0.0, valid=False)

    def estimate_from_flow(self, flow_tensor, y1, y2, x1, x2):
        """
        이미 계산된 RAFT flow에서 카메라 이동을 추정한다.
        추가 광학 흐름 계산 없음. 전적으로 GPU tensor 연산.

        flow_tensor: (2, H, W) RAFT flow, GPU 상의 fp32 텐서
        y1,y2,x1,x2: ROI 픽셀 경계 (피사체 영역 → 배경 추정에서 제외)

        Returns: AffineResult
          ty: 픽셀 단위 카메라 수직 이동 (translation)
          scale: 줌 비율 (1.0 = 변화 없음, >1 = 줌인)
          inlier_ratio: 배경 흐름의 일관성 (신뢰도 지표)
          valid: 추정 성공 여부
        """
        H, W = flow_tensor.shape[1], flow_tensor.shape[2]

        # 배경 마스크 (ROI 외부 픽셀만 사용)
        bg_mask = torch.ones(H, W, dtype=torch.bool, device=flow_tensor.device)
        bg_mask[y1:y2, x1:x2] = False

        if bg_mask.sum().item() < self.MIN_BG_PIXELS:
            return self._INVALID

        bg_dy = flow_tensor[1][bg_mask]  # 배경 수직 흐름
        bg_dx = flow_tensor[0][bg_mask]  # 배경 수평 흐름

        # Robust translation: median (GPU, O(n log n))
        ty = float(bg_dy.median().item())
        tx = float(bg_dx.median().item())

        # 줌 추정: 배경 흐름의 방사형 성분 분석
        # 줌인 시: 화면 중심에서 멀어지는 방향의 흐름이 증가
        ys_bg, xs_bg = torch.where(bg_mask)
        cy, cx = H / 2.0, W / 2.0

        # 방사형 단위 벡터 (정규화)
        rad_y = (ys_bg.float() - cy) / H
        rad_x = (xs_bg.float() - cx) / W

        # translation 제거 후 방사형 성분
        flow_y_c = bg_dy - ty
        flow_x_c = bg_dx - tx
        radial_dot = flow_y_c * rad_y + flow_x_c * rad_x

        mean_r2 = float((rad_y ** 2 + rad_x ** 2).mean().item())
        if mean_r2 > 1e-6:
            scale_dev = float(radial_dot.mean().item()) / mean_r2
            scale_dev = max(-0.2, min(0.5, scale_dev))  # 합리적 범위 클램프
            scale = 1.0 + scale_dev
        else:
            scale = 1.0

        # 신뢰도: 배경 흐름 std가 낮을수록 일관된 카메라 이동 → 높은 신뢰도
        ty_std = float(bg_dy.float().std().item())
        inlier_ratio = max(0.0, 1.0 - min(1.0, ty_std / (abs(ty) + 1.0)))

        return AffineResult(
            tx=tx, ty=ty, scale=scale, rotation=0.0,
            inlier_ratio=inlier_ratio, bg_std=ty_std, valid=True
        )


class SceneBoundaryHandler:
    """
    장면 전환 경계의 velocity spike를 Hanning window로 제거한다.

    장면 전환 시 이전 씬 마지막 프레임과 새 씬 첫 프레임 사이의
    광학 흐름은 피사체 동작과 무관한 노이즈다.
    이 spike가 velocity_signal에 남으면 HPF 필터가 일시적으로 왜곡된다.
    """

    def smooth_at_boundaries(self, velocity_signal, scene_boundaries, fps,
                             post_cut_suppress_frames=30):
        """
        velocity_signal: numpy array (N,)
        scene_boundaries: list of (start_frame, end_frame) tuples
        fps: 초당 프레임 수
        post_cut_suppress_frames: 장면 전환 후 억제할 프레임 수 (dual 모드에서 사용)

        Returns: smoothed velocity_signal (동일 크기)
        """
        transition_frames = max(3, int(fps * 0.3))
        result = velocity_signal.copy()

        for scene_start, _ in scene_boundaries:
            if scene_start == 0:
                continue  # 첫 씬은 경계 없음

            # 경계 프레임 자체 → 0 (cross-scene flow는 완전히 무의미)
            if scene_start < len(result):
                result[scene_start] = 0.0

            # 이전 씬 끝 → Hanning 하강 (fade out: 1.0 → 0.0)
            t_start = max(0, scene_start - transition_frames)
            t_end = scene_start
            n = t_end - t_start
            if n > 1:
                window = np.hanning(n * 2)[n:]
                result[t_start:t_end] *= window

            # 새 씬 시작 → Hanning 상승 + post-cut 억제
            suppress_end = min(len(result), scene_start + post_cut_suppress_frames)
            fade_end = min(len(result), scene_start + transition_frames + 1)

            if post_cut_suppress_frames > transition_frames:
                # 전환 구간: Hanning 상승 (기존)
                t_start = scene_start + 1
                n = fade_end - t_start
                if n > 1:
                    window = np.hanning(n * 2)[:n]
                    result[t_start:fade_end] *= window
                # 억제 구간: 0으로 유지 (Hanning 이후 ~ suppress_end)
                if fade_end < suppress_end:
                    result[fade_end:suppress_end] = 0.0
            else:
                # post_cut_suppress_frames <= transition_frames: 기존 로직
                t_start = scene_start + 1
                n = fade_end - t_start
                if n > 1:
                    window = np.hanning(n * 2)[:n]
                    result[t_start:fade_end] *= window

        return result


class OpticalFlowEstimator:
    def __init__(self, batch_size=8):
        self.device = DEVICE
        self.batch_size = batch_size

        self.weights = Raft_Small_Weights.DEFAULT if RAFT_SMALL else Raft_Large_Weights.DEFAULT
        self.model = raft_small(weights=self.weights) if RAFT_SMALL else raft_large(weights=self.weights)
        self.model = self.model.to(self.device).eval()
        # fp16 is NOT used: RAFT's internal ops (grid_sample, correlation) require fp32.
        # Batch processing alone gives the speedup without precision issues.
        self.transforms = self.weights.transforms()

    def estimate_flow(self, img1, img2):
        """
        단일 프레임 쌍 인터페이스 (하위 호환 유지).
        img1, img2: numpy arrays (H, W, 3) in BGR
        Returns: flow tensor (2, H, W) on GPU, fp32
        """
        flows = self.estimate_flow_batch([(img1, img2)])
        return flows[0]

    def estimate_flow_batch(self, frame_pairs):
        """
        배치 광학 흐름 추정 (fp32 배치 처리).
        frame_pairs: list of (prev_bgr, curr_bgr) numpy arrays
        Returns: list of flow tensors (2, H, W), fp32, on GPU

        배치 처리만으로도 Python 루프 오버헤드 제거 + GPU 파이프라인 최적화로
        순차 처리 대비 유의미한 속도 향상.
        """
        all_flows = []

        for i in range(0, len(frame_pairs), self.batch_size):
            batch = frame_pairs[i:i + self.batch_size]

            prev_tensors = []
            curr_tensors = []
            for prev_bgr, curr_bgr in batch:
                prev_tensors.append(self._bgr_to_tensor(prev_bgr))
                curr_tensors.append(self._bgr_to_tensor(curr_bgr))

            prev_batch = torch.stack(prev_tensors).to(self.device)
            curr_batch = torch.stack(curr_tensors).to(self.device)

            # RAFT transforms: uint8 (B,C,H,W) → float32 normalized
            prev_batch, curr_batch = self.transforms(prev_batch, curr_batch)

            with torch.no_grad():
                flows_list = self.model(prev_batch, curr_batch)
                flows = flows_list[-1]  # (B, 2, H, W) 최종 반복 결과

            for j in range(flows.shape[0]):
                all_flows.append(flows[j])  # (2, H, W)

        return all_flows

    def _bgr_to_tensor(self, img_bgr):
        """BGR numpy → uint8 RGB tensor (3, H, W)"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).permute(2, 0, 1)


class ROIDetector:
    """Detects the region of interest where the primary vertical motion occurs."""

    def __init__(self, flow_estimator):
        self.flow_estimator = flow_estimator

    def detect_roi(self, video_path, sample_count=80, frame_range=None,
                   yolo_tracker=None):
        """
        Sample frames from the video and find the region with the most
        consistent vertical oscillation.
        frame_range: optional (start_frame, end_frame) to limit detection to a scene.
        yolo_tracker: optional YoloPoseTracker — if provided, tries YOLO person bbox first.
        Returns: (y_start, y_end, x_start, x_end) as fractions of frame size (0-1).
        """
        # ── YOLO-first ROI detection ──
        if yolo_tracker is not None:
            yolo_roi = self._detect_roi_yolo(video_path, frame_range, yolo_tracker)
            if yolo_roi is not None:
                return yolo_roi
        # ── Fallback: optical flow variance ──
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

    def _detect_roi_yolo(self, video_path, frame_range, yolo_tracker, max_samples=20,
                         min_conf=0.4, min_hit_ratio=0.30):
        """
        YOLO 인물 감지로 ROI 추정.
        샘플 프레임에서 person bbox의 중앙값을 ROI로 변환.
        성공 시 (y1,y2,x1,x2) fraction, 실패 시 None 반환.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        r_start = 0 if frame_range is None else max(0, frame_range[0])
        r_end = total_frames if frame_range is None else min(total_frames, frame_range[1])
        span = r_end - r_start
        if span < 5:
            cap.release()
            return None

        n_samples = min(max_samples, span // 2)
        step = max(1, span // (n_samples + 1))
        indices = list(range(r_start + step, r_end - 1, step))[:n_samples]

        bboxes = []  # (x1,y1,x2,y2) pixel, in original frame coords before resize
        frame_hw = None

        for idx in indices:
            frame = self._read_frame(cap, idx)
            if frame is None:
                continue
            if frame_hw is None:
                frame_hw = frame.shape[:2]  # (h, w) of resized frame
            result = yolo_tracker.process_frame(frame)
            if result['bbox'] is not None and result['confidence'] >= min_conf:
                bboxes.append(result['bbox'])

        cap.release()

        if frame_hw is None or len(bboxes) < max(3, int(len(indices) * min_hit_ratio)):
            return None

        # bbox 중앙값으로 안정적 ROI 계산
        bboxes_arr = np.array(bboxes, dtype=np.float32)
        x1m = float(np.median(bboxes_arr[:, 0]))
        y1m = float(np.median(bboxes_arr[:, 1]))
        x2m = float(np.median(bboxes_arr[:, 2]))
        y2m = float(np.median(bboxes_arr[:, 3]))

        fh, fw = frame_hw
        margin = 0.12
        bw, bh = x2m - x1m, y2m - y1m
        x1r = max(0.0, (x1m - bw * margin) / fw)
        y1r = max(0.0, (y1m - bh * margin) / fh)
        x2r = min(1.0, (x2m + bw * margin) / fw)
        y2r = min(1.0, (y2m + bh * margin) / fh)

        # 너무 작거나 전체 화면에 가까우면 fallback
        if (x2r - x1r) > 0.95 and (y2r - y1r) > 0.95:
            return None

        return (y1r, y2r, x1r, x2r)

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
        # Smoothing factor to prevent jitter (lower = less ROI drift from camera movement)
        self._ema_alpha = 0.1
        # Maximum cumulative drift allowed (fraction of frame size)
        self._max_drift = 0.20
        self._base_roi = None

    def reset(self, initial_roi):
        """Reset tracker with a new base ROI (e.g., at scene boundary)."""
        self.current_roi = initial_roi
        self._base_roi = initial_roi
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

        # Apply shift to ROI (move ROI with camera), clamped to max drift from base
        roi_height = y2_frac - y1_frac
        roi_width = x2_frac - x1_frac

        # Clamp cumulative drift to prevent ROI from wandering too far
        if self._base_roi is not None:
            base_y1, base_y2, base_x1, base_x2 = self._base_roi
            self._shift_y = max(-self._max_drift, min(self._max_drift, self._shift_y))
            self._shift_x = max(-self._max_drift, min(self._max_drift, self._shift_x))

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

    def classify_segments(self, motion_magnitudes, scene_changes, total_frames,
                          velocity_signal=None, yolo_confs=None):
        """
        Classify each segment between scene changes using per-scene local thresholds.
        Returns list of (start_frame, end_frame, segment_type) tuples.
        """
        # Add final boundary
        boundaries = sorted(set(scene_changes + [total_frames]))
        if boundaries[0] != 0:
            boundaries = [0] + boundaries

        segments = []
        min_quiet_frames = int(self.fps * 2.0)  # Reverted Fix 5b: 1.5s caused bonfie over-splitting via smaller smoothing window

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
            elif (velocity_signal is not None and
                  np.std(velocity_signal[start:end]) < 0.012 and
                  np.mean(np.abs(velocity_signal[start:end])) < 0.020):
                seg_type = 'QUIET'
            elif end - start < min_quiet_frames:
                seg_type = 'TRANSITION'
            else:
                seg_type = 'ACTIVE'

            # ── YOLO 신뢰도 기반 비활성 구간 보정 ────────────────────────────
            # 인물이 지속적으로 감지되지 않는 구간 → 성행위 구간이 아닐 가능성 높음
            if seg_type == 'ACTIVE' and yolo_confs is not None and end > start:
                seg_c = [c for c in yolo_confs[start:end] if c is not None]
                if len(seg_c) >= 5:
                    # 탐지율: conf > 0.3 프레임 비율
                    det_ratio = sum(1 for c in seg_c if c > 0.3) / len(seg_c)
                    if det_ratio < 0.15:
                        # 15% 미만 감지 → 실제 성행위 구간이 아님 → QUIET 전환
                        seg_type = 'QUIET'

            segments.append((start, end, seg_type))

        # Safety check: if too many segments are QUIET, reclassify aggressively
        # But only if there's genuine high-variance motion (not a truly quiet video)
        total_duration = sum(e - s for s, e, _ in segments)
        quiet_duration = sum(e - s for s, e, t in segments if t == 'QUIET')
        vel_variance = np.var(motion_magnitudes) if velocity_signal is None else np.var(velocity_signal)
        if total_duration > 0 and quiet_duration / total_duration > 0.70 and vel_variance > 0.005:
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

        # vel_std 조용한 구간 재분류: _split_active_at_quiet 이후 서브 세그먼트에도 적용
        # (씬 레벨 vel_std 체크는 통과했더라도 서브 세그먼트 자체가 조용할 수 있음)
        if velocity_signal is not None:
            vel_checked = []
            for s, e, t in refined:
                if t == 'ACTIVE' and (e - s) >= min_quiet_frames:
                    vseg = velocity_signal[s:e]
                    if (np.std(vseg) < 0.012 and
                            np.mean(np.abs(vseg)) < 0.020):
                        t = 'QUIET'
                vel_checked.append((s, e, t))
            refined = vel_checked

        # Verify ACTIVE segments have rhythmic oscillation (actual action pattern)
        # Non-rhythmic motion (camera pans, character walking) → downgrade to TRANSITION
        # Static intro protection: short ACTIVE segments in first 3 seconds → TRANSITION
        intro_frames = int(self.fps * 3.0)
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
        """Compute per-scene adaptive threshold for quiet zone detection."""
        abs_motion = np.abs(seg_motion)
        nonzero = abs_motion[abs_motion > 0.01]
        if len(nonzero) < 10:
            return 0.05  # Very low fallback for sparse data
        return np.percentile(nonzero, 15)

    def _is_rhythmic_motion(self, seg_motion, seg_velocity=None):
        """
        Check if motion has rhythmic vertical oscillation pattern (indicating actual action).
        Returns True if the segment likely contains repetitive up-down stroking motion.
        Non-rhythmic motion (camera pans, zooms, etc.) returns False.

        seg_velocity: optional signed velocity signal for unidirectional camera-motion check.
        """
        if len(seg_motion) < int(self.fps):
            return True  # Too short to analyze, assume active

        # --- Absolute amplitude guard ---
        # Very low absolute motion means residual background noise, not actual stroking.
        # Librarian 130-150s: optical flow noise has std > 0.012 but tiny amplitude.
        seg_abs = np.abs(seg_motion)
        if np.mean(seg_abs) < 0.015 and np.max(seg_abs) < 0.05:
            return False

        # --- Unidirectional velocity check (camera pan/zoom detection) ---
        # Uses SIGNED velocity, not magnitude. Camera pans have mostly one-sign velocity.
        if seg_velocity is not None and len(seg_velocity) >= int(self.fps):
            vel = np.array(seg_velocity, dtype=float)
            vel_centered = vel - np.mean(vel)
            sign_pos = np.sum(vel_centered > 0) / max(1, len(vel_centered))
            vel_zc = np.sum(np.diff(np.sign(vel_centered)) != 0)
            vel_zc_rate = vel_zc / (len(vel_centered) / self.fps)
            if (sign_pos > 0.85 or sign_pos < 0.15) and vel_zc_rate < 0.5:
                return False  # Strongly unidirectional → camera movement, not stroking

        # --- FFT rhythmicity check on motion magnitude ---
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

        # Zero-crossing rate as oscillation indicator
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
        is_quiet = smoothed < threshold  # Reverted: threshold*0.7 caused bonfie over-splitting into <1s segments

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
        # Stroking is symmetric (up/down) so net mean velocity ≈ 0.
        # Any non-zero mean indicates camera drift or ROI error → remove it.
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
                seg_range = np.max(seg) - np.min(seg)

                if seg_range < 1e-5:
                    normalized[start:end] = last_active_val
                elif seg_range < median_range * 0.03:
                    # Noise guard: range extremely small → likely noise or detection failure.
                    # If last value is extreme (stuck at top/bottom), return to neutral (50)
                    if last_active_val > 75 or last_active_val < 25:
                        n = end - start
                        normalized[start:end] = np.linspace(last_active_val, 50.0, n)
                    else:
                        normalized[start:end] = last_active_val
                else:
                    # Percentile-based linear normalization
                    p2 = np.percentile(seg, 2)
                    p98 = np.percentile(seg, 98)
                    p_range = p98 - p2
                    if p_range < 1e-5:
                        p2 = np.min(seg)
                        p_range = seg_range

                    amplitude_ratio = seg_range / median_range if median_range > 1e-5 else 1.0

                    if amplitude_ratio < 0.4:
                        # G3: 소진폭 세그먼트 — 비례 스케일링 (전범위 0-100 stretch 없음)
                        # 기기가 소폭 동작(깔짝이는 동작)에 과도하게 반응하는 것 방지
                        output_range = max(20.0, 100.0 * amplitude_ratio)
                        output_center = 50.0
                        stretched = (
                            (seg - p2) / p_range * output_range
                            + (output_center - output_range / 2.0)
                        )
                    else:
                        # 정상 진폭: p2→0, p98→100 (기존 방식)
                        stretched = (seg - p2) / p_range * 100.0

                    normalized[start:end] = np.clip(stretched, 0, 100)

                last_active_val = normalized[end - 1]

            elif seg_type == 'TRANSITION':
                seg_range = np.max(seg) - np.min(seg)

                if seg_range < 1e-5:
                    normalized[start:end] = last_active_val
                else:
                    # Percentile-based for transitions (slightly tighter: p5→10, p95→90)
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

            # Detect stuck-at-extreme artifact: narrow range AND biased near top or bottom.
            if range_used < 20 and (p5 > 65 or p95 < 35):
                local_strength = 0.5   # Stuck artifact → suppress amplification
            elif range_used > 70:
                local_strength = 2.0   # Already wide — push harder toward extremes
            elif range_used > 40:
                local_strength = 3.0   # Medium
            else:
                local_strength = 4.0   # Narrow → aggressive expansion

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
                continue  # No actions in quiet zones

            seg = position_signal[start:end]

            if seg_type == 'TRANSITION':
                # Sparse sampling for transitions
                trans_actions = self._sample_transition(seg, start)
                actions.extend(trans_actions)
            else:  # ACTIVE
                # Peak/trough-primary sampling
                vel_seg = velocity_signal[start:end] if velocity_signal is not None else None
                active_actions = self._sample_active(seg, start, vel_seg)
                actions.extend(active_actions)

        if not actions:
            return []

        actions.sort(key=lambda x: x['at'])

        # Enforce minimum spacing (20ms allows up to 50 actions/sec for fast strokes)
        actions = self._enforce_min_spacing(actions, min_gap_ms=20)

        # Deduplicate
        actions = self._deduplicate(actions)

        # Snap segment extrema to 0/100 so full-stroke range is used
        actions = self._snap_extremes(actions)

        return actions

    def _detect_stroke_frequency(self, seg):
        """
        Detect dominant stroke frequency using MAX of FFT and ZC-rate estimates.

        FFT: accurate for clean sinusoidal signals, can underestimate for diffuse spectra.
        ZC-rate: accurate for fast oscillations, can overestimate for slow drifting signals.
        Taking the MAX ensures fast stroking content (bonfie) is never underestimated,
        while slow content (Librarian) returns the correct low value from both methods.
        """
        if len(seg) < int(self.fps):
            return 1.0

        n = len(seg)
        sig = seg - np.mean(seg)

        # --- FFT estimate (on position signal) ---
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        mask = (freqs >= 0.3) & (freqs <= 10.0)
        if np.any(mask):
            fft_freq = float(freqs[mask][np.argmax(fft_vals[mask])])
        else:
            fft_freq = 1.0

        # --- ZC-rate estimate (on position signal) ---
        zc = np.sum(np.diff(np.sign(sig)) != 0)
        duration_s = n / self.fps
        zc_freq = float(np.clip(zc / (2.0 * duration_s), 0.3, 10.0))

        # Take MAX: if either method detects fast stroking, honor it
        return float(max(fft_freq, zc_freq))

    def _sample_active(self, seg, global_offset, vel_seg=None):
        """
        Frequency-adaptive strategy for ACTIVE segments:
        1. Analyze stroke frequency to determine detection parameters
        2. Find all peaks and troughs as PRIMARY actions
        3. Refine to true extrema
        4. Fill gaps with frequency-appropriate spacing
        """
        if len(seg) < 3:
            return [self._make_action(global_offset, seg[0])]

        # Suppress near-center narrow-range segments (noise / camera movement artifact)
        # Tighter condition: only filter if range is very small AND strictly centered
        pos_range = float(np.max(seg) - np.min(seg))
        pos_mean = float(np.mean(seg))
        if pos_range < 8 and 38 < pos_mean < 62:
            return []

        # Fix 3: Universal fine-grained min_dist — detect more peaks, filter later
        min_dist = max(1, int(self.fps / 20))

        # Fix 2: Amplitude-relative prominence (8% of signal range)
        # Replaces fixed frequency-tier prominence (3/5/8)
        signal_range = float(np.max(seg) - np.min(seg))
        prominence = max(1.5, min(10.0, signal_range * 0.08))  # 2.0→1.5: 고속 소진폭 피크 누락 방지

        # Fix B: Use velocity FFT as additional stroke_freq estimate.
        # Position = integral(velocity) → high-freq (>5Hz) amplitude shrinks by 1/(2πf) relative to drift.
        # Slow drift (0.5-2Hz) dominates position signal → position-based stroke_freq << actual.
        # Velocity FFT is reliable for 3-10Hz content: SG(5,1) reduces amplitude but NOT frequency.
        # Diagnostic: bonfie vel_zc=75 (1.4Hz) → velocity also shows slow oscillation → bonfie unchanged.
        # Anna Anon (10Hz), Remilia (7.8Hz): velocity FFT correctly detects high freq → fast path.
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
            print(f"[DIAG] pos_freq={stroke_freq:.2f} vel_fft={vel_fft_freq:.2f} seg={len(seg)}f vel={len(vel_seg)}f")
            stroke_freq = max(stroke_freq, vel_fft_freq)

        # Adaptive smoothing window: narrower for fast strokes to preserve peaks
        period_frames = self.fps / max(stroke_freq, 0.5)
        divisor = 8 if stroke_freq > 3.0 else 4
        smooth_window = max(5, int(period_frames / divisor))
        smooth_window = smooth_window | 1  # Ensure odd
        if smooth_window >= len(seg):
            smooth_window = max(3, (len(seg) - 1) | 1)

        if len(seg) > smooth_window:
            seg_smooth = savgol_filter(seg, smooth_window, min(2, smooth_window - 1))
        else:
            seg_smooth = seg

        peaks, _ = find_peaks(seg_smooth, distance=min_dist, prominence=prominence)
        troughs, _ = find_peaks(-seg_smooth, distance=min_dist, prominence=prominence)

        # Fix A+B: stroke_freq 분기 기반 max_gap_ms
        # stroke_freq는 position-based MAX(FFT,ZC) + velocity-based FFT의 MAX
        #
        # 고속(>5Hz, Fix B가 감지): stroke_period * 1.2 → 실제 스트로크 주기에 맞는 fill
        #   - threshold 5Hz: Librarian(3Hz)를 slow path에 안전하게 유지
        #   - multiplier 1.2: 0.4(v4)보다 크게 → Anna Anon(10Hz)에서 ~100ms fill (인간 스크립트와 일치)
        # 저속(≤5Hz): extrema × 1.5 → 비정상적으로 큰 gap만 fill, 정상 gap은 보존
        #   - bonfie: vel_FFT도 1.4Hz → slow path → 81 (velocity가 실제 20Hz 스트로크 미감지)
        all_extrema = sorted(set(peaks) | set(troughs))
        stroke_period_ms = 1000.0 / max(stroke_freq, 0.3)
        freq_based_gap = max(30, min(1000, int(stroke_period_ms * 1.2)))

        if stroke_freq > 5.0:
            # Fast content: velocity FFT detected >5Hz → use freq-based gap
            max_gap_ms = freq_based_gap
        elif len(all_extrema) >= 2:
            # Slow content: use extrema interval × 1.5 (no fill for normal-spaced extrema)
            extrema_intervals_ms = np.diff(all_extrema) * 1000.0 / self.fps
            median_interval_ms = float(np.median(extrema_intervals_ms))
            max_gap_ms = max(50, min(1000, int(median_interval_ms * 1.5)))
        else:
            max_gap_ms = freq_based_gap

        # Collect all peak/trough indices as primary action points
        primary_indices = set()
        primary_indices.add(0)
        primary_indices.add(len(seg) - 1)

        for idx in peaks:
            primary_indices.add(idx)
        for idx in troughs:
            primary_indices.add(idx)

        # ── Bounce/Rebound 감지 ─────────────────────────────────────────────
        # 빠른 대진폭 하강 스트로크 후 반동 기교 포인트 추가
        # 패턴: 10→0→2→0→1→0 (Anna Anon, Librarian 스타일)
        # 조건: 트라우에서 포지션 ≤ 12 AND 직전 스트로크 진폭 ≥ 25
        _BOUNCE_TROUGH_MAX = 12
        _BOUNCE_MIN_AMP    = 25
        _bounce_look       = max(2, int(self.fps * 0.20))  # 200ms 이전 최대값 확인
        bounce_extra = []
        for t_idx in troughs:
            if seg[t_idx] > _BOUNCE_TROUGH_MAX:
                continue
            look = min(t_idx, _bounce_look)
            if look < 2:
                continue
            preceding_max = float(np.max(seg[max(0, t_idx - look):t_idx + 1]))
            if preceding_max - seg[t_idx] < _BOUNCE_MIN_AMP:
                continue
            # 반동 포인트: trough+1f(소폭 상승), trough+2f(저점 복귀)
            trough_pos = float(seg[t_idx])
            for offset, delta in [(1, 2.5), (2, 0.5)]:
                b_idx = t_idx + offset
                if 0 < b_idx < len(seg) - 1:
                    bounce_extra.append(
                        self._make_action(global_offset + b_idx, trough_pos + delta)
                    )

        # G2: velocity zero-crossing 비활성화
        # 이유: extra action points가 DTW를 악화시킴 (NIKKE -0.050, Anna -0.012, Remilia -0.008)
        # bonfie는 G2로 추가 actions=0 → G3만으로 충분함
        # Librarian은 G2로 310→462 과다 생성 (30th percentile 필터로도 부족)

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

        # Bounce 포인트 병합 (빠른 대진폭 스트로크에서만 추가됨)
        if bounce_extra:
            actions = sorted(actions + bounce_extra, key=lambda x: x['at'])

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

    def _snap_extremes(self, actions, snap_threshold=3):
        """
        Snap the global top/bottom of the script to pos=100/0.
        Only snaps when:
        1. The value is already very close to the extreme (within snap_threshold)
        2. The number of affected actions is <= 5% of total (avoids mass snapping)
        This ensures full 0–100 range like human scripts without over-snapping.
        """
        if len(actions) < 4:
            return actions
        positions = [a['pos'] for a in actions]
        top = max(positions)
        bot = min(positions)
        n = len(actions)
        max_snap_count = max(3, int(n * 0.05))  # At most 5% of actions snapped

        if top >= 100 - snap_threshold:
            candidates = [a for a in actions if a['pos'] >= top - snap_threshold]
            if len(candidates) <= max_snap_count:
                for a in candidates:
                    a['pos'] = 100

        if bot <= snap_threshold:
            candidates = [a for a in actions if a['pos'] <= bot + snap_threshold]
            if len(candidates) <= max_snap_count:
                for a in candidates:
                    a['pos'] = 0

        return actions

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
        self.camera_comp = CameraMotionCompensator()

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

        # ── RAFT Flow 기반 카메라 이동 보정 (추가 계산 없음) ──
        # 이미 계산된 RAFT flow의 배경 픽셀에서 translation/scale 추정.
        # LK 없이 GPU 연산만 사용 → 처리 시간 추가 없음.
        affine = self.camera_comp.estimate_from_flow(flow, y1, y2, x1, x2)

        flow_roi = flow[:, y1:y2, x1:x2]

        # Translation 보정: 임계값 이하(잡음)는 무시하여 정적 카메라 보호
        ty_correction = (affine.ty
                         if affine.valid and abs(affine.ty) >= CameraMotionCompensator.MIN_TY_PX
                         else 0.0)
        flow_y = flow_roi[1] - ty_correction
        flow_x = flow_roi[0]

        # 줌/회전 탐지 및 억제
        # bg_std가 높으면 배경 흐름이 비균일 → 줌/회전 (순수 translation이면 bg_std 낮음)
        # zoom_by_scale은 제거: affine.scale이 노이즈가 많아 정상 프레임에서도 오발동
        bg_abs_mean = abs(affine.ty) if affine.valid else 0.0
        zoom_detected = (affine.valid
                         and affine.bg_std > max(bg_abs_mean * 1.5, 0.5)
                         and affine.bg_std > 0.8)
        if zoom_detected:
            # 줌/회전 프레임: velocity/magnitude 억제 (카메라 움직임 신호 오염 방지)
            flow_y = flow_y * 0.1
            flow_x = flow_x * 0.1

        # Compute magnitude for thresholding (using compensated flow)
        flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2)

        # Low threshold to capture subtle animation motion (was 0.3, then 0.15)
        mag_threshold = 0.08
        mask = flow_mag > mag_threshold

        if mask.sum() < 10:
            return 0.0, 0.0, roi_height_px, zoom_detected

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

        return velocity, magnitude, roi_height_px, zoom_detected

    def extract_frame_histogram(self, frame):
        """Extract grayscale histogram for scene change detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray


class OneEuroFilter:
    """
    1€ Filter (Casiez et al. 2012): 적응형 저역통과 필터.

    저속 → 높은 스무딩 (jitter 억제)
    고속 → 낮은 스무딩 (빠른 움직임 지연 없이 추적)

    EMA 대비 장점: 저속 노이즈와 고속 실제 이동을 동시에 처리.
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


class YoloPoseTracker:
    """
    YOLOv8/v11 Pose 기반 인물 감지 + 골반 키포인트 추출.

    COCO 17-keypoint 포맷:
      5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip

    process_frame()은 PoseTracker와 동일한 (hip_center_y, reference_length, confidence)
    외에 bbox, keypoints를 추가로 반환한다.
    """

    LEFT_HIP       = 11
    RIGHT_HIP      = 12
    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6
    LEFT_KNEE      = 13
    RIGHT_KNEE     = 14

    KP_BODY_WEIGHTS = {11: 2.0, 12: 2.0, 13: 0.6, 14: 0.6}
    HIP_KNEE_RATIO  = 0.12   # 무릎→골반 정규화 거리 보정값

    MIN_KP_CONF    = 0.30   # 키포인트 최소 신뢰도
    MIN_DET_CONF   = 0.40   # 탐지 최소 신뢰도

    # 슬롯 기반 트래킹 파라미터
    MAX_SLOTS            = 2      # 최대 추적 인물 수 (P1=0, P2=1 고정)
    SLOT_MATCH_DIST      = 0.25   # 슬롯 매칭 최대 거리 (정규화 frame 비율)
    HIP_HISTORY_LEN      = 40     # hip_y 이력 길이 (프레임)
    MAX_MISS_FRAMES      = 20     # 이 프레임 이상 미탐지 시 last_result 클리어
    DUAL_WARMUP_FRAMES   = 30     # dual 모드 확정까지 유예 프레임
    DUAL_STABLE_FRAMES   = 15     # dual 확정 후 rel_dist 안정화 대기 프레임
    IOU_DUPLICATE_THRESH = 0.30   # P1-P2 bbox IoU > 이 값 → ghost P2로 제거
    EMA_HIP_ALPHA        = 0.4    # hip_y EMA 평활화 (낮을수록 안정)

    # 인물 인식 정확도 필터
    MIN_ROI_OVERLAP   = 0.05   # bbox-ROI 최소 오버랩 비율 (완전 배경 인물 제거)
    P2_PROXIMITY_MAR  = 0.60   # P2: P1 bbox 크기의 60% 여백 내에만 슬롯 생성 허용
    MIN_P2_SIZE_RATIO = 0.35   # P2 bbox height ≥ P1 bbox height × 0.35

    def __init__(self, model_path: str, device: str = None, fps: float = 30.0):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        from ultralytics import YOLO
        from collections import deque
        self._deque_cls = deque
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = YOLO(model_path)
        self._fps = fps
        self._last_result = None  # 마지막 process_frame() 결과 캐시

        # 슬롯 기반 트래킹 상태
        self._slots = []            # list of slot dicts
        self._primary_slot = 0      # 현재 주인공 슬롯 인덱스
        self._secondary_slot = -1   # secondary 슬롯 인덱스 (-1 = 없음)
        self._dual_confirmed = False
        self._dual_confirmed_at = 0  # dual 확정 시점 프레임
        self._frame_count = 0        # 처리 프레임 수

        # ID 기반 트래킹 (model.track persist=True)
        self._use_tracking = True   # False → detection-only fallback
        self._track_id_map = {}     # track_id(int) → slot_index(int)

    def _make_slot(self, cx, cy):
        return {
            'cx': cx,               # bbox 중심 x (정규화)
            'cy': cy,               # bbox 중심 y (정규화)
            'hip_history': self._deque_cls(maxlen=self.HIP_HISTORY_LEN),
            'last_result': None,
            'miss': 0,              # 연속 미탐지 프레임 수
            'smoothed_hip_y': None, # 평활화된 hip_y
            'hip_oef': None,        # OneEuroFilter (lazy init)
        }

    @staticmethod
    def _iou(bbox_a: tuple, bbox_b: tuple) -> float:
        """두 bbox의 IoU (Intersection over Union) 계산."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _bbox_roi_overlap_ratio(bbox: tuple, roi: tuple, fw: float, fh: float) -> float:
        """bbox(pixel abs)와 roi(정규화 y1,y2,x1,x2)의 교집합 / bbox면적 비율."""
        ry1, ry2, rx1, rx2 = roi
        roi_x1, roi_y1 = rx1 * fw, ry1 * fh
        roi_x2, roi_y2 = rx2 * fw, ry2 * fh
        ix1 = max(bbox[0], roi_x1);  iy1 = max(bbox[1], roi_y1)
        ix2 = min(bbox[2], roi_x2);  iy2 = min(bbox[3], roi_y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return float(inter / bbox_area) if bbox_area > 0 else 0.0

    @staticmethod
    def _bbox_proximity(bbox_cand: tuple, bbox_ref: tuple, margin_ratio: float) -> bool:
        """bbox_cand 중심이 bbox_ref를 margin_ratio 비율만큼 확장한 영역 내에 있는지 확인.
        성행위 영상에서 P1-P2는 항상 신체 접촉 → P2 center는 P1 expanded bbox 안에 위치."""
        w = bbox_ref[2] - bbox_ref[0]
        h = bbox_ref[3] - bbox_ref[1]
        ex1 = bbox_ref[0] - w * margin_ratio
        ex2 = bbox_ref[2] + w * margin_ratio
        ey1 = bbox_ref[1] - h * margin_ratio
        ey2 = bbox_ref[3] + h * margin_ratio
        cx = (bbox_cand[0] + bbox_cand[2]) / 2.0
        cy = (bbox_cand[1] + bbox_cand[3]) / 2.0
        return ex1 <= cx <= ex2 and ey1 <= cy <= ey2

    def process_frame(self, frame_bgr: np.ndarray,
                      roi_fractions: tuple = None) -> dict:
        """
        frame_bgr: BGR uint8 (H, W, 3)
        Returns dict:
          'hip_center_y'     : float (0~1, frame_h 정규화) or None
          'reference_length' : float (shoulder-hip 거리 정규화) or None
          'confidence'       : float 0~1
          'bbox'             : (x1,y1,x2,y2) pixel or None
          'keypoints'        : np.ndarray shape(17,3) [x,y,conf] or None
          'secondary_hip_y'  : float or None  (두 번째 인물 hip_y)
          'rel_dist'         : float or None  (두 골반 상대 거리)
          'is_dual'          : bool           (dual 모드 활성 여부)
        """
        empty = {'hip_center_y': None, 'reference_length': None,
                 'confidence': 0.0, 'bbox': None, 'keypoints': None,
                 'secondary_hip_y': None, 'rel_dist': None, 'is_dual': False}
        self._frame_count += 1

        try:
            if self._use_tracking:
                results = self._model.track(
                    frame_bgr, persist=True, verbose=False,
                    device=self._device,
                )
            else:
                results = self._model(
                    frame_bgr, verbose=False, device=self._device,
                )
        except Exception as _e:
            # track() 미지원 시 detection-only fallback
            if self._use_tracking:
                self._use_tracking = False
                try:
                    results = self._model(frame_bgr, verbose=False, device=self._device)
                except Exception:
                    self._last_result = empty
                    return empty
            else:
                self._last_result = empty
                return empty

        fh, fw = frame_bgr.shape[:2]
        persons = self._parse_all_persons(results, (fh, fw), roi_fractions=roi_fractions)
        result = self._update_slots(persons, fw, fh)
        self._last_result = result
        return result

    def _estimate_hip_y_robust(self, kp, fh: float):
        """골반 키포인트 불완전 시 무릎으로 보완한 body center y (정규화)."""
        weighted_sum = 0.0
        weight_total = 0.0

        for ki in [self.LEFT_HIP, self.RIGHT_HIP]:
            kx, ky, kc = kp[ki]
            if kc >= 0.25:
                w = self.KP_BODY_WEIGHTS[ki] * kc
                weighted_sum += (ky / fh) * w
                weight_total += w

        hip_visible_weight = sum(
            kp[ki][2] for ki in [self.LEFT_HIP, self.RIGHT_HIP]
            if kp[ki][2] >= 0.25
        )
        if hip_visible_weight < 0.5 and kp.shape[0] > self.RIGHT_KNEE:
            for ki in [self.LEFT_KNEE, self.RIGHT_KNEE]:
                kx, ky, kc = kp[ki]
                if kc >= 0.30:
                    est_hip_y = (ky / fh) - self.HIP_KNEE_RATIO
                    w = self.KP_BODY_WEIGHTS[ki] * kc
                    weighted_sum += est_hip_y * w
                    weight_total += w

        return weighted_sum / weight_total if weight_total > 0 else None

    def _estimate_ref_len(self, kp, bbox, fh: float) -> float:
        """3단계 fallback: 어깨+골반 → 골반+무릎 → bbox 비율."""
        lhx, lhy, lhc = kp[self.LEFT_HIP]
        rhx, rhy, rhc = kp[self.RIGHT_HIP]
        lsx, lsy, lsc = kp[self.LEFT_SHOULDER]
        rsx, rsy, rsc = kp[self.RIGHT_SHOULDER]

        # Level 1: 어깨 + 골반
        if lsc >= 0.25 and rsc >= 0.25 and (lhc >= 0.25 or rhc >= 0.25):
            hy = ((lhy if lhc >= 0.25 else rhy) + (rhy if rhc >= 0.25 else lhy)) / 2.0
            shoulder_y = (lsy + rsy) / 2.0
            ref = abs(hy - shoulder_y) / fh
            if ref >= 0.04:
                return ref

        # Level 2: 골반 + 무릎
        if kp.shape[0] > self.RIGHT_KNEE:
            lkx, lky, lkc = kp[self.LEFT_KNEE]
            rkx, rky, rkc = kp[self.RIGHT_KNEE]
            hip_ok = lhc >= 0.25 or rhc >= 0.25
            knee_ok = lkc >= 0.25 or rkc >= 0.25
            if hip_ok and knee_ok:
                hy = lhy if lhc >= 0.25 else rhy
                ky_val = lky if lkc >= 0.25 else rky
                ref = abs(hy - ky_val) * 0.9 / fh
                if ref >= 0.04:
                    return ref

        # Level 3: bbox 비율
        return (bbox[3] - bbox[1]) / fh * 0.35

    def _kp_quality_score(self, kp) -> float:
        """가시 키포인트 수 기반 품질 점수 0~1."""
        visible = sum(
            1 for ki in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                         self.LEFT_HIP, self.RIGHT_HIP,
                         self.LEFT_KNEE, self.RIGHT_KNEE]
            if ki < kp.shape[0] and kp[ki][2] >= 0.25
        )
        return min(1.0, visible / 4.0)

    def _adaptive_ema_alpha(self, slot) -> float:
        """hip_history ZC rate로 stroke 주파수 추정 → EMA alpha 동적 결정."""
        h = list(slot['hip_history'])
        if len(h) < 10:
            return self.EMA_HIP_ALPHA
        sig = np.array(h) - np.mean(h)
        zc = np.sum(np.diff(np.sign(sig)) != 0)
        freq_est = zc / (2.0 * len(h) / self._fps)
        if freq_est > 3.5:   # 5.0→3.5: Anna 10Hz가 더 빨리 fast path 진입
            return 0.85      # 0.75→0.85: 10Hz 진폭 65%→82% 보존
        return self.EMA_HIP_ALPHA

    def _parse_all_persons(self, results, frame_hw,
                           roi_fractions=None) -> list:
        """탐지된 모든 person을 파싱하여 list[dict] 반환."""
        empty = {'hip_center_y': None, 'reference_length': None,
                 'confidence': 0.0, 'bbox': None, 'keypoints': None, 'cx': 0.5, 'cy': 0.5}
        out = []

        if not results or results[0].boxes is None:
            return out

        boxes = results[0].boxes
        kps_data = results[0].keypoints
        fh, fw = frame_hw

        # track() 사용 시 persist 트래킹 ID 추출
        track_ids_raw = None
        if hasattr(boxes, 'id') and boxes.id is not None:
            track_ids_raw = boxes.id.int().cpu().numpy()

        for i, (cls, conf) in enumerate(zip(boxes.cls.cpu().numpy(),
                                            boxes.conf.cpu().numpy())):
            if int(cls) != 0 or float(conf) < self.MIN_DET_CONF:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy()
            bbox = tuple(float(v) for v in xyxy)
            cx = (bbox[0] + bbox[2]) / 2.0 / fw
            cy = (bbox[1] + bbox[3]) / 2.0 / fh

            # 트래킹 ID (None = 탐지만 된 경우)
            track_id = int(track_ids_raw[i]) if track_ids_raw is not None else None

            # ── ROI 오버랩 필터: motion ROI 밖 완전 배경 인물 제거 ──
            if roi_fractions is not None:
                overlap = self._bbox_roi_overlap_ratio(bbox, roi_fractions, fw, fh)
                if overlap < self.MIN_ROI_OVERLAP:
                    continue

            base = {'confidence': float(conf), 'bbox': bbox, 'track_id': track_id,
                    'keypoints': None, 'hip_center_y': None,
                    'reference_length': None, 'cx': cx, 'cy': cy,
                    'track_id': track_id}

            if kps_data is None or kps_data.data is None:
                out.append(base)
                continue

            kp = kps_data.data[i].cpu().numpy()
            if kp.shape[0] < 13:
                out.append(base)
                continue

            base['keypoints'] = kp
            hip_y = self._estimate_hip_y_robust(kp, fh)

            if hip_y is None:
                base['confidence'] *= 0.5
                out.append(base)
                continue

            ref_len = self._estimate_ref_len(kp, bbox, fh)
            if ref_len < 0.04:
                base['confidence'] *= 0.3
                out.append(base)
                continue

            kp_quality = self._kp_quality_score(kp)
            base['confidence'] = float(conf) * max(kp_quality, 0.3)
            base['hip_center_y'] = float(hip_y)
            base['reference_length'] = float(ref_len)
            out.append(base)

        return out

    def _update_slots(self, persons: list, fw: int, fh: int) -> dict:
        """고정 인덱스 슬롯 매칭: slots[0]=P1, slots[1]=P2 영구 고정, 재정렬 없음."""
        empty = {'hip_center_y': None, 'reference_length': None,
                 'confidence': 0.0, 'bbox': None, 'keypoints': None,
                 'secondary_hip_y': None, 'secondary_bbox': None,
                 'rel_dist': None, 'is_dual': False}

        # 슬롯 미탐지 카운터 증가
        for s in self._slots:
            s['miss'] += 1

        # 신뢰도 높은 순서로 greedy 매칭 (좋은 탐지 먼저 고정 슬롯 선점)
        sorted_persons = sorted(persons, key=lambda p: p['confidence'], reverse=True)

        matched_slots = set()
        for p in sorted_persons:
            tid = p.get('track_id')

            # ── 1단계: Track ID 기반 슬롯 매칭 (최우선) ──────────────────
            # YOLO persist 트래킹 ID가 있으면 슬롯을 고정 → P1/P2 스와핑 방지
            best_si = None
            if tid is not None:
                # 등록된 ID → 해당 슬롯 직접 매칭
                if tid in self._track_id_map:
                    si = self._track_id_map[tid]
                    if si not in matched_slots and si < len(self._slots):
                        best_si = si

            # ── 2단계: 근접 거리 기반 매칭 (track ID 없는 경우 fallback) ──
            if best_si is None:
                best_dist = self.SLOT_MATCH_DIST
                for si, s in enumerate(self._slots):
                    if si in matched_slots:
                        continue
                    dist = ((p['cx'] - s['cx']) ** 2 + (p['cy'] - s['cy']) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_si = si

            if best_si is not None:
                s = self._slots[best_si]
                # 새 track ID → 슬롯 등록
                if tid is not None and tid not in self._track_id_map:
                    # 같은 슬롯에 이미 다른 ID가 있으면 덮어쓰기 (재초기화)
                    old_id = next((k for k, v in self._track_id_map.items() if v == best_si), None)
                    if old_id is not None:
                        del self._track_id_map[old_id]
                    self._track_id_map[tid] = best_si
            elif len(self._slots) < self.MAX_SLOTS:
                # P2 슬롯 신규 생성 시 — P1 근접 제약 + 크기 필터
                if len(self._slots) == 1:
                    p1_res = self._slots[0].get('last_result')
                    if p1_res and p1_res.get('bbox') and p.get('bbox'):
                        p1_bbox = p1_res['bbox']
                        p1_h = p1_bbox[3] - p1_bbox[1]
                        p2_h = p['bbox'][3] - p['bbox'][1]
                        # 크기 필터: P2가 P1 대비 너무 작으면 거부
                        if p1_h > 0 and p2_h < p1_h * self.MIN_P2_SIZE_RATIO:
                            continue
                        # 근접 필터: P2 center가 P1 expanded bbox 밖이면 거부
                        if not self._bbox_proximity(p['bbox'], p1_bbox, self.P2_PROXIMITY_MAR):
                            continue
                # 새 슬롯 생성 (0=P1, 1=P2 순서 고정)
                self._slots.append(self._make_slot(p['cx'], p['cy']))
                best_si = len(self._slots) - 1
                # 새 슬롯 track ID 등록
                if tid is not None:
                    self._track_id_map[tid] = best_si
                s = self._slots[best_si]
            else:
                continue  # 슬롯 가득 참 → 이 인물 무시

            # 슬롯 EMA 위치 갱신
            s['cx'] = p['cx'] * 0.3 + s['cx'] * 0.7
            s['cy'] = p['cy'] * 0.3 + s['cy'] * 0.7
            s['miss'] = 0
            # OneEuroFilter hip_y 스무딩 — 저속 jitter 억제 + 고속 반응성 보존
            if p['hip_center_y'] is not None:
                s['hip_history'].append(p['hip_center_y'])  # raw값: std 계산용
                if s['hip_oef'] is None:
                    # 파라미터 설계:
                    # min_cutoff=3.0 → 정지시 alpha≈0.52 (EMA 0.4보다 살짝 강한 스무딩)
                    # beta=8.0, d_cutoff=5.0 → 고속 스트로크(dx_hat≈3/s)시 cutoff≈27Hz → alpha≈0.85
                    # EMA adaptive (저속 0.4 / >3.5Hz 0.85) 동등한 동적 특성
                    s['hip_oef'] = OneEuroFilter(
                        self._fps, min_cutoff=3.0, beta=8.0, d_cutoff=5.0
                    )
                smoothed = s['hip_oef'](p['hip_center_y'])
                s['smoothed_hip_y'] = smoothed
                p = dict(p)  # 복사하여 반환값용 hip_center_y만 스무딩 적용
                p['hip_center_y'] = smoothed
            s['last_result'] = p
            matched_slots.add(best_si)

        # miss > MAX_MISS_FRAMES: last_result 클리어 (슬롯 인덱스는 절대 변경 안함)
        for s in self._slots:
            if s['miss'] > self.MAX_MISS_FRAMES:
                s['last_result'] = None

        if not self._slots:
            return empty

        # P1은 항상 slots[0] — 교체 없음
        primary = self._slots[0]
        if primary['last_result'] is None:
            # P1 미탐지 시: P2가 있으면 single 모드로 P2 사용 (dual 비활성)
            if len(self._slots) >= 2 and self._slots[1]['last_result'] is not None:
                r = self._slots[1]['last_result']
                return {
                    'hip_center_y'    : r['hip_center_y'],
                    'reference_length': r['reference_length'],
                    'confidence'      : r['confidence'],
                    'bbox'            : r['bbox'],
                    'keypoints'       : r['keypoints'],
                    'secondary_hip_y' : None,
                    'secondary_bbox'  : None,
                    'rel_dist'        : None,
                    'is_dual'         : False,
                }
            return empty

        # ── P1-P2 IoU 중복 가드: 동일 인물 ghost 탐지 제거 ─────────────
        if len(self._slots) >= 2:
            p1_r = self._slots[0]['last_result']
            p2_r = self._slots[1]['last_result']
            if p1_r is not None and p2_r is not None:
                b1, b2 = p1_r.get('bbox'), p2_r.get('bbox')
                if b1 and b2 and self._iou(b1, b2) > self.IOU_DUPLICATE_THRESH:
                    self._slots[1]['last_result'] = None   # ghost P2 제거
                    self._slots[1]['smoothed_hip_y'] = None

        # ── P2 (secondary) 슬롯 확인 ─────────────────────────────────
        p2_active = (
            self._frame_count >= self.DUAL_WARMUP_FRAMES and
            len(self._slots) >= 2 and
            self._slots[1]['last_result'] is not None
        )

        if p2_active:
            self._secondary_slot = 1
            prev_confirmed = self._dual_confirmed
            self._dual_confirmed = True
            if not prev_confirmed:
                self._dual_confirmed_at = self._frame_count
        else:
            self._secondary_slot = -1
            if self._dual_confirmed:
                # P2가 사라졌을 때 확정 상태 유지 (miss 기간 내 재등장 대기)
                # MAX_MISS_FRAMES 초과 시 last_result=None 으로 이미 처리됨
                p2_slot_exists = len(self._slots) >= 2
                if not p2_slot_exists:
                    self._dual_confirmed = False

        # ── rel_dist 계산 ─────────────────────────────────────────────
        stable = (self._dual_confirmed and
                  self._frame_count >= self._dual_confirmed_at + self.DUAL_STABLE_FRAMES)

        sec_hip_y = None
        sec_bbox  = None
        rel_dist  = None
        if self._dual_confirmed and len(self._slots) >= 2:
            sec = self._slots[1]
            if sec['last_result'] is not None:
                sec_hip_y = sec['last_result'].get('hip_center_y')
                sec_bbox  = sec['last_result'].get('bbox')
                if stable:
                    pri_hip_y = primary['last_result'].get('hip_center_y')
                    if sec_hip_y is not None and pri_hip_y is not None:
                        rel_dist = abs(pri_hip_y - sec_hip_y)

        r = primary['last_result']
        return {
            'hip_center_y'    : r['hip_center_y'],
            'reference_length': r['reference_length'],
            'confidence'      : r['confidence'],
            'bbox'            : r['bbox'],
            'keypoints'       : r['keypoints'],
            'secondary_hip_y' : sec_hip_y,
            'secondary_bbox'  : sec_bbox,
            'rel_dist'        : rel_dist,
            'is_dual'         : (rel_dist is not None),
        }

    def get_person_roi(self, frame_h: int, frame_w: int,
                       margin: float = 0.12) -> tuple:
        """주인공 슬롯 bbox → ROI fraction (y1,y2,x1,x2). 없으면 None."""
        if self._last_result is None or self._last_result['bbox'] is None:
            return None
        x1, y1, x2, y2 = self._last_result['bbox']
        bw, bh = x2 - x1, y2 - y1
        x1r = max(0.0, (x1 - bw * margin) / frame_w)
        y1r = max(0.0, (y1 - bh * margin) / frame_h)
        x2r = min(1.0, (x2 + bw * margin) / frame_w)
        y2r = min(1.0, (y2 + bh * margin) / frame_h)
        return (y1r, y2r, x1r, x2r)

    def reset_tracking(self):
        """씬 전환 시 슬롯 상태 클리어 — 새 씬에서 처음부터 인물 재매칭."""
        for s in self._slots:
            s['last_result'] = None
            s['smoothed_hip_y'] = None
            s['hip_history'].clear()
            s['miss'] = self.MAX_MISS_FRAMES
            if s.get('hip_oef') is not None:
                s['hip_oef'].reset()
        self._track_id_map.clear()  # 씬 전환 시 ID 재매핑 허용
        self._dual_confirmed = False
        self._dual_confirmed_at = 0
        self._frame_count = 0

    def close(self):
        pass  # ultralytics YOLO는 명시적 close 불필요


class ContactPointTracker:
    """
    MTFG 개념: P1+P2 골반 중간 접촉점을 Lucas-Kanade 픽셀 추적으로 안정화.

    YOLO 탐지 오류에 독립적 — 픽셀 패턴 자체를 추적하므로 겹침 노이즈 극복.

    접촉점 정의:
      contact_px = (P1.hip_x_pixel + P2.hip_x_pixel) / 2
      contact_py = (P1.hip_y_pixel + P2.hip_y_pixel) / 2

    추적 포인트: contact_py 주변 PATCH_SIZE 영역의 goodFeaturesToTrack 결과
    """

    PATCH_SIZE = 32
    MIN_VALID_POINTS = 3
    MAX_Y_JUMP = 0.15  # frame_h 기준 contact_y 최대 프레임간 변화

    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    GFT_PARAMS = dict(
        maxCorners=15,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5
    )

    def __init__(self):
        self._prev_gray   = None
        self._prev_pts    = None
        self._contact_y   = None
        self._initialized = False

    def reset(self):
        """씬 전환 or 추적 실패 시 초기화."""
        self._prev_gray   = None
        self._prev_pts    = None
        self._contact_y   = None
        self._initialized = False

    def update(self, frame_gray: np.ndarray,
               contact_px, contact_py,
               frame_h: int, frame_w: int):
        """
        frame_gray: 현재 프레임 (H×W uint8 grayscale)
        contact_px, contact_py: YOLO 기반 접촉점 픽셀 좌표 (None if not dual)
        frame_h, frame_w: 프레임 크기
        반환: contact_y 정규화 (0~1) or None
        """
        result_y = self._track(frame_gray, frame_h)

        if result_y is not None:
            self._prev_gray = frame_gray
            return result_y

        # 추적 실패 or 미초기화 → YOLO 접촉점으로 재초기화
        if contact_px is not None and contact_py is not None:
            self._initialize(frame_gray, contact_px, contact_py, frame_h, frame_w)

        self._prev_gray = frame_gray
        return None

    def _initialize(self, frame_gray, cx, cy, frame_h, frame_w):
        half = self.PATCH_SIZE // 2
        x1 = max(0, int(cx) - half)
        x2 = min(frame_w, int(cx) + half)
        y1 = max(0, int(cy) - half)
        y2 = min(frame_h, int(cy) + half)

        patch = frame_gray[y1:y2, x1:x2]
        if patch.size == 0:
            return

        pts = cv2.goodFeaturesToTrack(patch, **self.GFT_PARAMS)
        if pts is None or len(pts) < self.MIN_VALID_POINTS:
            return

        # 패치 좌표 → 프레임 절대 좌표
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1

        self._prev_pts    = pts
        self._contact_y   = float(cy) / frame_h
        self._initialized = True

    def _track(self, frame_gray, frame_h):
        if not self._initialized or self._prev_gray is None or self._prev_pts is None:
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, frame_gray,
            self._prev_pts, None,
            **self.LK_PARAMS
        )
        if next_pts is None:
            self.reset()
            return None

        good = status.ravel() == 1
        if good.sum() < self.MIN_VALID_POINTS:
            self.reset()
            return None

        new_y = float(np.median(next_pts[good, 0, 1])) / frame_h

        if self._contact_y is not None and abs(new_y - self._contact_y) > self.MAX_Y_JUMP:
            self.reset()
            return None

        self._prev_pts  = next_pts[good].reshape(-1, 1, 2)
        self._contact_y = new_y
        return new_y
