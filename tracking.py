import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from collections import namedtuple
from config_manager import config
from signal_processing import OneEuroFilter

# Internal Defaults
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RAFT_SMALL = False
RESIZE_WIDTH = 512

try:
    from ultralytics import YOLO as _YOLO_CHECK
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Named tuple for affine camera motion estimation result
AffineResult = namedtuple('AffineResult', ['tx', 'ty', 'scale', 'rotation', 'inlier_ratio', 'bg_std', 'valid'])

class FrameCadenceDetector:
    """Detects if the video is animated 'on-twos' or 'on-threes' (repeated frames)."""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.history = []
        self.max_history = 10
        
    def is_duplicate(self, img1, img2):
        """Checks if two frames are identical or near-identical using MSE."""
        if img1 is None or img2 is None: return False
        # Calculate MSE on a downsampled version for speed
        h, w = img1.shape[:2]
        ds = 4
        m1 = cv2.resize(img1, (w//ds, h//ds), interpolation=cv2.INTER_NEAREST)
        m2 = cv2.resize(img2, (w//ds, h//ds), interpolation=cv2.INTER_NEAREST)
        mse = np.mean((m1.astype(np.float32) - m2.astype(np.float32)) ** 2)
        return mse < self.threshold

class TemporalInterpolator:
    """Interpolates motion signals for repeated animation frames."""
    def __init__(self):
        self.last_valid_velocity = 0.0
        self.duplicate_count = 0
        
    def update(self, velocity, is_duplicate):
        if is_duplicate:
            self.duplicate_count += 1
            # Maintain previous velocity but slightly decay it to avoid infinite loops
            # In animations, motion is 'held' or 'stepped'
            return self.last_valid_velocity * 0.95
        else:
            self.duplicate_count = 0
            self.last_valid_velocity = velocity
            return velocity

class CameraMotionCompensator:
    """RAFT flow-based camera motion estimation."""
    MIN_BG_PIXELS = 16
    MIN_TY_PX = 0.5
    MIN_SCALE_DEV = 0.03
    _INVALID = AffineResult(tx=0.0, ty=0.0, scale=1.0, rotation=0.0,
                            inlier_ratio=0.0, bg_std=0.0, valid=False)

    def estimate_from_flow(self, flow_tensor, y1, y2, x1, x2):
        H, W = flow_tensor.shape[1], flow_tensor.shape[2]
        bg_mask = torch.ones(H, W, dtype=torch.bool, device=flow_tensor.device)
        bg_mask[y1:y2, x1:x2] = False
        if bg_mask.sum().item() < self.MIN_BG_PIXELS:
            return self._INVALID
        
        bg_flow = flow_tensor[:, bg_mask]
        ty = bg_flow[1].median().item()
        tx = bg_flow[0].median().item()
        bg_std = bg_flow.std().item()
        return AffineResult(tx=tx, ty=ty, scale=1.0, rotation=0.0, inlier_ratio=1.0, bg_std=bg_std, valid=True)

class OpticalFlowEstimator:
    def __init__(self, batch_size=8):
        self.device = DEVICE
        self.batch_size = batch_size
        self.weights = Raft_Small_Weights.DEFAULT if RAFT_SMALL else Raft_Large_Weights.DEFAULT
        self.model = raft_small(weights=self.weights) if RAFT_SMALL else raft_large(weights=self.weights)
        self.model = self.model.to(self.device).eval()
        self.transforms = self.weights.transforms()

    def estimate_flow(self, img1, img2):
        flows = self.estimate_flow_batch([(img1, img2)])
        return flows[0]

    def estimate_flow_batch(self, frame_pairs):
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
            prev_batch, curr_batch = self.transforms(prev_batch, curr_batch)
            with torch.no_grad():
                flows_list = self.model(prev_batch, curr_batch)
                flows = flows_list[-1]
            for j in range(flows.shape[0]):
                all_flows.append(flows[j])
        return all_flows

    def _bgr_to_tensor(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).permute(2, 0, 1)

class ROIDetector:
    def __init__(self, flow_estimator):
        self.flow_estimator = flow_estimator

    def detect_roi(self, video_path, sample_count=80, frame_range=None, yolo_tracker=None):
        if yolo_tracker is not None:
            yolo_roi = self._detect_roi_yolo(video_path, frame_range, yolo_tracker)
            if yolo_roi is not None: return yolo_roi
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return (0.15, 0.85, 0.15, 0.85)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        r_start, r_end = (0, total_frames) if frame_range is None else (max(0, frame_range[0]), min(total_frames, frame_range[1]))
        if r_end - r_start < 10:
            cap.release(); return (0.15, 0.85, 0.15, 0.85)
        effective_count = min(sample_count, (r_end - r_start) // 2)
        step = max(1, (r_end - r_start) // (effective_count + 1))
        indices = list(range(r_start + step, r_end - 1, step))[:effective_count]
        if len(indices) < 5:
            cap.release(); return (0.15, 0.85, 0.15, 0.85)
        first_frame = self._read_frame(cap, indices[0])
        if first_frame is None:
            cap.release(); return (0.15, 0.85, 0.15, 0.85)
        h, w = first_frame.shape[:2]
        grid_h, grid_w = 8, 8
        cell_h, cell_w = h // grid_h, w // grid_w
        grid_flows = np.zeros((grid_h, grid_w, len(indices) - 1))
        prev_frame = first_frame
        for i in range(1, len(indices)):
            curr_frame = self._read_frame(cap, indices[i])
            if curr_frame is None: break
            flow = self.flow_estimator.estimate_flow(prev_frame, curr_frame)
            flow_y = flow[1].cpu().numpy()
            for gy in range(grid_h):
                for gx in range(grid_w):
                    grid_flows[gy, gx, i-1] = np.mean(flow_y[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w])
            prev_frame = curr_frame
        cap.release()
        grid_variance = np.var(grid_flows, axis=2)
        best_score, best_region = 0, (0, grid_h, 0, grid_w)
        for sy in range(3, grid_h + 1):
            for sx in range(3, grid_w + 1):
                for gy in range(grid_h - sy + 1):
                    for gx in range(grid_w - sx + 1):
                        score = np.sum(grid_variance[gy:gy+sy, gx:gx+sx])
                        penalty = (sy * sx) / (grid_h * grid_w)
                        adj_score = score / (penalty ** 0.3)
                        if adj_score > best_score:
                            best_score, best_region = adj_score, (gy, gy+sy, gx, gx+sx)
        gy1, gy2, gx1, gx2 = best_region
        return (gy1/grid_h, gy2/grid_h, gx1/grid_w, gx2/grid_w)

    def _detect_roi_yolo(self, video_path, frame_range, yolo_tracker, max_samples=20, min_conf=0.4, min_hit_ratio=0.30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rs, re = (0, total) if frame_range is None else (max(0, frame_range[0]), min(total, frame_range[1]))
        if re - rs < 5: cap.release(); return None
        n = min(max_samples, (re - rs) // 2)
        step = max(1, (re - rs) // (n + 1))
        indices = list(range(rs + step, re - 1, step))[:n]
        bboxes, hw = [], None
        for idx in indices:
            frame = self._read_frame(cap, idx)
            if frame is None: continue
            if hw is None: hw = frame.shape[:2]
            res = yolo_tracker.detect_persons_stateless(frame)
            
            frame_bboxes = []
            for p in res:
                if p['bbox'] is not None and p['confidence'] >= min_conf:
                    frame_bboxes.append(p['bbox'])
            
            if frame_bboxes:
                ba_f = np.array(frame_bboxes, dtype=np.float32)
                bboxes.append((np.min(ba_f[:, 0]), np.min(ba_f[:, 1]), np.max(ba_f[:, 2]), np.max(ba_f[:, 3])))
                
        cap.release()
        if hw is None or len(bboxes) < max(3, int(len(indices) * min_hit_ratio)): return None
        
        ba = np.array(bboxes, dtype=np.float32)
        # Use percentiles to ensure robustness against frames where some persons were missing
        x1m, y1m = np.percentile(ba[:, 0], 25), np.percentile(ba[:, 1], 25)
        x2m, y2m = np.percentile(ba[:, 2], 75), np.percentile(ba[:, 3], 75)
        fh, fw = hw
        bw, bh = x2m - x1m, y2m - y1m
        margin = 0.12
        x1r, y1r, x2r, y2r = max(0, (x1m-bw*margin)/fw), max(0, (y1m-bh*margin)/fh), min(1, (x2m+bw*margin)/fw), min(1, (y2m+bh*margin)/fh)
        if (x2r - x1r) > 0.95 and (y2r - y1r) > 0.95: return None
        return (y1r, y2r, x1r, x2r)

    def _read_frame(self, cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: return None
        return cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1]))))

class DynamicROITracker:
    def __init__(self, fps, revalidate_interval_sec=2.0):
        self.fps = fps
        self.revalidate_interval = int(fps * revalidate_interval_sec)
        self.current_roi = None
        self.frame_count_since_reset = 0
        self._shift_y = 0.0
        self._shift_x = 0.0
        self._ema_alpha = 0.1
        self._max_drift = 0.20
        self._base_roi = None

    def reset(self, initial_roi):
        self.current_roi = initial_roi
        self._base_roi = initial_roi
        self.frame_count_since_reset = 0
        self._shift_y = 0.0
        self._shift_x = 0.0

    def update(self, flow_tensor, frame_h, frame_w):
        if self.current_roi is None: return (0.15, 0.85, 0.15, 0.85)
        y1f, y2f, x1f, x2f = self.current_roi
        y1, y2, x1, x2 = int(y1f * frame_h), int(y2f * frame_h), int(x1f * frame_w), int(x2f * frame_w)
        bg_mask = torch.ones(flow_tensor.shape[1], flow_tensor.shape[2], dtype=torch.bool, device=flow_tensor.device)
        bg_mask[y1:y2, x1:x2] = False
        cam_dy, cam_dx = (flow_tensor[1][bg_mask].median().item(), flow_tensor[0][bg_mask].median().item()) if bg_mask.sum() > 100 else (0.0, 0.0)
        dyf, dxf = cam_dy / frame_h, cam_dx / frame_w
        self._shift_y = self._ema_alpha * dyf + (1 - self._ema_alpha) * self._shift_y
        self._shift_x = self._ema_alpha * dxf + (1 - self._ema_alpha) * self._shift_x
        ry, rx = y2f - y1f, x2f - x1f
        if self._base_roi is not None:
            self._shift_y = max(-self._max_drift, min(self._max_drift, self._shift_y))
            self._shift_x = max(-self._max_drift, min(self._max_drift, self._shift_x))
        ny1, nx1 = y1f + self._shift_y, x1f + self._shift_x
        ny2, nx2 = ny1 + ry, nx1 + rx
        if ny1 < 0: ny1, ny2 = 0.0, ry
        if ny2 > 1: ny1, ny2 = 1.0 - ry, 1.0
        if nx1 < 0: nx1, nx2 = 0.0, rx
        if nx2 > 1: nx1, nx2 = 1.0 - rx, 1.0
        self.current_roi = (max(0, ny1), min(1, ny2), max(0, nx1), min(1, nx2))
        self.frame_count_since_reset += 1
        return self.current_roi

    def needs_revalidation(self):
        return self.frame_count_since_reset >= self.revalidate_interval

class YoloPoseTracker:
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_KNEE, RIGHT_KNEE = 11, 12, 5, 6, 13, 14
    KP_BODY_WEIGHTS = {11: 2.0, 12: 2.0, 13: 0.6, 14: 0.6}
    HIP_KNEE_RATIO, MIN_KP_CONF, MIN_DET_CONF = 0.12, 0.30, 0.40
    MAX_SLOTS, SLOT_MATCH_DIST, HIP_HISTORY_LEN, MAX_MISS_FRAMES = 5, 0.25, 40, 20
    DUAL_WARMUP_FRAMES, DUAL_STABLE_FRAMES, IOU_DUPLICATE_THRESH, EMA_HIP_ALPHA = 30, 15, 0.60, 0.4
    MIN_ROI_OVERLAP, P2_PROXIMITY_MAR, MIN_P2_SIZE_RATIO = 0.05, 0.60, 0.35

    def __init__(self, model_path: str, device: str = None, fps: float = 30.0):
        from ultralytics import YOLO
        from collections import deque
        self._deque_cls = deque
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = YOLO(model_path)
        self._fps, self._last_result = fps, None
        self._slots, self._primary_slot, self._secondary_slot = [], 0, -1
        self._dual_confirmed, self._dual_confirmed_at, self._frame_count = False, 0, 0
        self._use_tracking, self._track_id_map = True, {}
        self._p1_centroid_ema, self._p2_centroid_ema, self._p1_ema_alpha = None, None, 0.15
        self._p1_hint_bbox, self._p2_hint_bbox = None, None
        self._hint_applied = False
        
        # PSP: Adaptive Skeleton Proportions
        self._psp = {
            'p1': {'ref_lens': deque(maxlen=60), 'avg_ref': 0.15},
            'p2': {'ref_lens': deque(maxlen=60), 'avg_ref': 0.15}
        }

    def _make_slot(self, cx, cy, bbox=None):
        return {
            'cx': cx, 'cy': cy, 'hip_history': self._deque_cls(maxlen=self.HIP_HISTORY_LEN),
            'last_result': None, 'miss': 0, 'smoothed_hip_y': None,
            'hip_oef': None, 'shoulder_oef': None, 'knee_oef': None,
            'smoothed_shoulder_y': None, 'smoothed_knee_y': None,
            'bbox': bbox, 'dx': 0.0, 'dy': 0.0,
        }

    def _iou(self, b1, b2):
        if not b1 or not b2: return 0.0
        ix1, iy1, ix2, iy2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return float(inter / u) if u > 0 else 0.0

    def process_frame(self, frame_bgr, roi_fractions=None):
        empty = {'hip_center_y': None, 'reference_length': None, 'confidence': 0.0, 'bbox': None, 'keypoints': None, 'secondary_hip_y': None, 'rel_dist': None, 'is_dual': False}
        self._frame_count += 1
        try:
            res = self._model.track(frame_bgr, persist=True, verbose=False, device=self._device, tracker="botsort.yaml") if self._use_tracking else self._model(frame_bgr, verbose=False, device=self._device)
        except:
            if self._use_tracking: self._use_tracking = False; res = self._model(frame_bgr, verbose=False, device=self._device)
            else: return empty
        fh, fw = frame_bgr.shape[:2]
        persons = self._parse_all_persons(res, (fh, fw), roi_fractions)
        self._last_result = self._update_slots(persons, fw, fh)
        return self._last_result
    def _parse_all_persons(self, results, hw, roi=None):
        out = []
        if not results or results[0].boxes is None: return out
        boxes, kps, fh, fw = results[0].boxes, results[0].keypoints, hw[0], hw[1]
        ids = boxes.id.int().cpu().numpy() if hasattr(boxes, 'id') and boxes.id is not None else None
        
        # NMS deduplication to prevent YOLO from detecting the same person twice
        # Sort by confidence first
        raw_preds = []
        for i, (cls, conf) in enumerate(zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy())):
            if int(cls) != 0 or float(conf) < self.MIN_DET_CONF: continue
            raw_preds.append((i, float(conf)))
            
        raw_preds.sort(key=lambda x: x[1], reverse=True)
        
        kept_indices = []
        for i, conf in raw_preds:
            bbox_i = tuple(float(v) for v in boxes.xyxy[i].cpu().numpy())
            is_dup = False
            for j in kept_indices:
                bbox_j = tuple(float(v) for v in boxes.xyxy[j].cpu().numpy())
                if self._iou(bbox_i, bbox_j) > self.IOU_DUPLICATE_THRESH:
                    is_dup = True
                    break
            if not is_dup:
                kept_indices.append(i)

        for i in kept_indices:
            cls, conf = int(boxes.cls[i].item()), float(boxes.conf[i].item())
            bbox = tuple(float(v) for v in boxes.xyxy[i].cpu().numpy())
            if roi:
                ry1, ry2, rx1, rx2 = roi
                ix1, iy1, ix2, iy2 = max(bbox[0], rx1*fw), max(bbox[1], ry1*fh), min(bbox[2], rx2*fw), min(bbox[3], ry2*fh)
                if (max(0,ix2-ix1)*max(0,iy2-iy1))/((bbox[2]-bbox[0])*(bbox[3]-bbox[1])) < self.MIN_ROI_OVERLAP: continue
            p = {'confidence': float(conf), 'bbox': bbox, 'track_id': ids[i] if ids is not None else None, 'keypoints': None, 'hip_center_y': None, 'reference_length': None, 'cx': (bbox[0]+bbox[2])/2/fw, 'cy': (bbox[1]+bbox[3])/2/fh}
            if kps is not None and kps.data is not None:
                kp = kps.data[i].cpu().numpy()
                if kp.shape[0] >= 13:
                    p['keypoints'] = kp
                    hy = self._estimate_hip_y_robust(kp, fh)
                    if hy: p['hip_center_y'], p['reference_length'] = hy, self._estimate_ref_len(kp, bbox, fh)
            out.append(p)
        return out

    def _estimate_hip_y_robust(self, kp, fh):
        ws, wt = 0, 0
        for ki in [self.LEFT_HIP, self.RIGHT_HIP]:
            if kp[ki][2] >= 0.25: w = self.KP_BODY_WEIGHTS[ki]*kp[ki][2]; ws += (kp[ki][1]/fh)*w; wt += w
        if wt == 0 and kp.shape[0] > self.RIGHT_KNEE:
            for ki in [self.LEFT_KNEE, self.RIGHT_KNEE]:
                if kp[ki][2] >= 0.3: est = (kp[ki][1]/fh)-self.HIP_KNEE_RATIO; w = self.KP_BODY_WEIGHTS[ki]*kp[ki][2]; ws += est*w; wt += w
        return ws/wt if wt > 0 else None

    def _estimate_ref_len(self, kp, bbox, fh):
        if kp[5][2]>=0.25 and kp[6][2]>=0.25 and (kp[11][2]>=0.25 or kp[12][2]>=0.25):
            hy = ((kp[11][1] if kp[11][2]>=0.25 else kp[12][1])+(kp[12][1] if kp[12][2]>=0.25 else kp[11][1]))/2
            ref = abs(hy-(kp[5][1]+kp[6][1])/2)/fh
            if ref >= 0.04: return ref
        return (bbox[3]-bbox[1])/fh*0.35

    def _assign_person_to_slot(self, s, p):
        if s['bbox'] is not None and p['bbox'] is not None:
            cx_old, cy_old = (s['bbox'][0]+s['bbox'][2])/2, (s['bbox'][1]+s['bbox'][3])/2
            cx_new, cy_new = (p['bbox'][0]+p['bbox'][2])/2, (p['bbox'][1]+p['bbox'][3])/2
            s['dx'] = s['dx'] * 0.5 + (cx_new - cx_old) * 0.5
            s['dy'] = s['dy'] * 0.5 + (cy_new - cy_old) * 0.5
        s['bbox'] = p['bbox']
        s['cx'] = p['cx']*0.3 + s['cx']*0.7
        s['cy'] = p['cy']*0.3 + s['cy']*0.7
        s['miss'] = 0
        if p['hip_center_y'] is not None:
            if s['hip_oef'] is None: s['hip_oef'] = OneEuroFilter(self._fps, 3.0, 8.0, 5.0)
            p['hip_center_y'] = s['hip_oef'](p['hip_center_y'])
        s['last_result'] = p

    def _update_slots(self, persons, fw, fh):
        for s in self._slots: s['miss'] += 1
        unmatched_ps = list(range(len(persons)))
        matched_slots = set()

        if not self._hint_applied and getattr(self, '_p1_hint_bbox', None) is not None:
            self._hint_applied = True
            if not self._slots:
                def _match_hint(hint_bbox):
                    best_iou, best_pi = 0.0, -1
                    if hint_bbox is None: return -1
                    for pi in unmatched_ps:
                        iou = self._iou(hint_bbox, persons[pi]['bbox'])
                        if iou > best_iou: best_iou, best_pi = iou, pi
                    return best_pi if best_iou > 0.1 else -1

                pi1 = _match_hint(self._p1_hint_bbox)
                if pi1 != -1:
                    unmatched_ps.remove(pi1)
                    s0 = self._make_slot(persons[pi1]['cx'], persons[pi1]['cy'], persons[pi1]['bbox'])
                    self._slots.append(s0)
                    self._assign_person_to_slot(s0, persons[pi1])
                    matched_slots.add(0)
                    
                if getattr(self, '_p2_hint_bbox', None) is not None:
                    pi2 = _match_hint(self._p2_hint_bbox)
                    if pi2 != -1:
                        unmatched_ps.remove(pi2)
                        if not self._slots:
                            empty_s = self._make_slot(0, 0, None)
                            empty_s['miss'] = self.MAX_MISS_FRAMES
                            self._slots.append(empty_s)
                        s1 = self._make_slot(persons[pi2]['cx'], persons[pi2]['cy'], persons[pi2]['bbox'])
                        self._slots.append(s1)
                        self._assign_person_to_slot(s1, persons[pi2])
                        matched_slots.add(1)

        for s in self._slots:
            if s['bbox'] is not None and s['miss'] < self.MAX_MISS_FRAMES:
                s['pred_bbox'] = (s['bbox'][0] + s['dx'], s['bbox'][1] + s['dy'], s['bbox'][2] + s['dx'], s['bbox'][3] + s['dy'])
            else:
                s['pred_bbox'] = s['bbox']

        track_matches = []
        for pi in unmatched_ps:
            tid = persons[pi].get('track_id')
            if tid is not None and tid in self._track_id_map:
                si = self._track_id_map[tid]
                if si not in matched_slots and si < len(self._slots) and self._slots[si]['miss'] <= self.MAX_MISS_FRAMES:
                    track_matches.append((si, pi))
                    
        for si, pi in track_matches:
            if si not in matched_slots and pi in unmatched_ps:
                matched_slots.add(si); unmatched_ps.remove(pi)
                self._assign_person_to_slot(self._slots[si], persons[pi])

        matches = []
        for si, s in enumerate(self._slots):
            if si in matched_slots or s['miss'] > self.MAX_MISS_FRAMES: continue
            for pi in unmatched_ps:
                iou = self._iou(s['pred_bbox'], persons[pi]['bbox'])
                if iou > 0.25: matches.append((iou, si, pi))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        for iou, si, pi in matches:
            if si not in matched_slots and pi in unmatched_ps:
                matched_slots.add(si); unmatched_ps.remove(pi)
                self._assign_person_to_slot(self._slots[si], persons[pi])

        dist_matches = []
        for si, s in enumerate(self._slots):
            if si in matched_slots or s['miss'] > self.MAX_MISS_FRAMES + 10: continue
            px, py = s['cx'] + (s['dx'] / fw), s['cy'] + (s['dy'] / fh)
            for pi in unmatched_ps:
                d = ((persons[pi]['cx'] - px)**2 + (persons[pi]['cy'] - py)**2)**0.5
                if d < self.SLOT_MATCH_DIST: dist_matches.append((d, si, pi))
        
        dist_matches.sort(key=lambda x: x[0])
        for d, si, pi in dist_matches:
            if si not in matched_slots and pi in unmatched_ps:
                matched_slots.add(si); unmatched_ps.remove(pi)
                self._assign_person_to_slot(self._slots[si], persons[pi])
                
        ps_sorted = sorted([persons[pi] for pi in unmatched_ps], key=lambda p: p['confidence'], reverse=True)
        for p in ps_sorted:
            if len(self._slots) < self.MAX_SLOTS:
                s = self._make_slot(p['cx'], p['cy'], p['bbox']); self._slots.append(s)
                self._assign_person_to_slot(s, p)
                matched_slots.add(len(self._slots)-1)
                
        for s in self._slots:
            if s['miss'] > self.MAX_MISS_FRAMES: s['last_result'], s['bbox'] = None, None
            if s['last_result'] and s['last_result'].get('track_id') is not None:
                self._track_id_map[s['last_result']['track_id']] = self._slots.index(s)

        # ── PSP Update ──
        if self._slots and self._slots[0]['last_result']:
            ref = self._slots[0]['last_result'].get('reference_length')
            if ref:
                self._psp['p1']['ref_lens'].append(ref)
                self._psp['p1']['avg_ref'] = np.median(self._psp['p1']['ref_lens'])
        if len(self._slots) >= 2 and self._slots[1]['last_result']:
            ref2 = self._slots[1]['last_result'].get('reference_length')
            if ref2:
                self._psp['p2']['ref_lens'].append(ref2)
                self._psp['p2']['avg_ref'] = np.median(self._psp['p2']['ref_lens'])

        if not self._slots or self._slots[0]['last_result'] is None:
            return {'hip_center_y': None, 'reference_length': None, 'confidence': 0.0, 'bbox': None, 'keypoints': None, 'secondary_hip_y': None, 'secondary_bbox': None, 'secondary_keypoints': None, 'rel_dist': None, 'is_dual': False}
        p1 = self._slots[0]['last_result']
        res = {'hip_center_y': p1['hip_center_y'], 'reference_length': p1['reference_length'], 'confidence': p1['confidence'], 'bbox': p1['bbox'], 'keypoints': p1['keypoints'], 'secondary_hip_y': None, 'secondary_bbox': None, 'secondary_keypoints': None, 'rel_dist': None, 'is_dual': False}
        if len(self._slots) >= 2 and self._slots[1]['last_result']:
            p2 = self._slots[1]['last_result']
            res.update({
                'secondary_hip_y': p2['hip_center_y'], 
                'secondary_bbox': p2['bbox'], 
                'secondary_keypoints': p2.get('keypoints'),
                'secondary_confidence': p2.get('confidence', 0.0),
                'secondary_reference_length': p2.get('reference_length'),
                'rel_dist': abs(p1['hip_center_y'] - p2['hip_center_y']) if p1['hip_center_y'] is not None and p2['hip_center_y'] is not None else None, 
                'is_dual': True
            })
        return res

    def detect_persons_stateless(self, frame_bgr):
        try: res = self._model(frame_bgr, verbose=False, device=self._device)
        except: return []
        fh, fw = frame_bgr.shape[:2]
        ps = self._parse_all_persons(res, (fh, fw), None)
        out = []
        for p in ps:
            kp, bbox = p.get('keypoints'), p.get('bbox')
            if not bbox: continue
            hip_px = None
            if kp is not None:
                v = [v for v in [kp[11], kp[12]] if v[2]>=self.MIN_KP_CONF]
                if v: hip_px = (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
            out.append({'bbox':bbox, 'keypoints':kp, 'hip_px':hip_px, 'confidence':p['confidence'], 'bbox_area':(bbox[2]-bbox[0])*(bbox[3]-bbox[1])})
        return sorted(out, key=lambda x: x['bbox_area'], reverse=True)

    def reset_tracking(self):
        self._slots.clear()
        self._track_id_map.clear(); self._p1_centroid_ema = self._p2_centroid_ema = None
        self._dual_confirmed = False; self._frame_count = 0
        self._hint_applied = False

    def force_slot_reorder_by_hint(self, p1_hint_bbox, p2_hint_bbox=None):
        """Reorder existing slots so that the ones matching hints are at index 0 and 1."""
        if not self._slots:
            return None

        def _find_best_slot(hint_bbox):
            if hint_bbox is None: return -1
            best_iou, best_si = 0.0, -1
            for si, s in enumerate(self._slots):
                if s['bbox'] is None: continue
                iou = self._iou(hint_bbox, s['bbox'])
                if iou > best_iou: best_iou, best_si = iou, si
            return best_si if best_iou > 0.1 else -1

        si1 = _find_best_slot(p1_hint_bbox)
        si2 = _find_best_slot(p2_hint_bbox)
        
        # Identity Map: original_index -> new_intended_index
        swaps = {}
        if si1 != -1:
            swaps[si1] = 0
            if si2 != -1 and si2 != si1:
                swaps[si2] = 1
        elif si2 != -1:
            swaps[si2] = 1

        if not swaps:
            return None

        # Reconstruct slots list
        new_slots = [None] * len(self._slots)
        used_orig = set()
        
        # Place the matched ones
        for orig_si, new_si in swaps.items():
            new_slots[new_si] = self._slots[orig_si]
            used_orig.add(orig_si)
            
        # Fill the rest
        next_free = 0
        for orig_si in range(len(self._slots)):
            if orig_si in used_orig: continue
            while next_free < len(new_slots) and new_slots[next_free] is not None:
                next_free += 1
            if next_free < len(new_slots):
                new_slots[next_free] = self._slots[orig_si]
                next_free += 1
        
        # Clean up None (if any, though shouldn't happen)
        self._slots = [s for s in new_slots if s is not None]
        
        # Rebuild track_id_map
        self._track_id_map.clear()
        for i, s in enumerate(self._slots):
            if s['last_result'] and s['last_result'].get('track_id') is not None:
                self._track_id_map[s['last_result']['track_id']] = i
                
        return swaps # Return map for retrospective data swapping in main.py

    def get_psp(self):
        """Returns the Personalized Skeleton Profile (PSP) for P1 and P2."""
        return {
            'p1_avg_ref': self._psp['p1']['avg_ref'],
            'p2_avg_ref': self._psp['p2']['avg_ref']
        }

    def close(self):
        """No explicit close needed for Ultralytics YOLO."""
        pass

class ContactPointTracker:
    """
    두 인물의 접촉점(골반 중간) 픽셀 패턴을 Lucas-Kanade로 추적.
    YOLO 탐지 오류에 독립적인 보조 신호 제공.
    """
    PATCH_SIZE = 32
    MIN_VALID_POINTS = 3
    MAX_Y_JUMP = 0.15

    def __init__(self):
        self.reset()
        self.LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def reset(self):
        self._initialized = False
        self._prev_gray = None
        self._prev_pts = None
        self._contact_y = None

    def update(self, frame_gray, contact_px, contact_py, fh, fw, is_duplicate=False):
        """
        반환: 추적된 contact_y (정규화 0~1) 또는 None
        """
        if is_duplicate:
            return self._contact_y

        if not self._initialized:
            if contact_px is not None and contact_py is not None:
                self._init_tracker(frame_gray, contact_px, contact_py, fh, fw)
            return None

        # LK 추적
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, frame_gray, self._prev_pts, None, **self.LK_PARAMS)
        
        good = status.ravel() == 1 if status is not None else np.array([])
        if status is None or good.sum() < self.MIN_VALID_POINTS:
            self._initialized = False
            return None

        new_pts = next_pts[good]
        new_y_px = np.median(new_pts[:, 0, 1])
        new_y_norm = new_y_px / fh

        # 급격한 점프 체크
        if self._contact_y is not None and abs(new_y_norm - self._contact_y) > self.MAX_Y_JUMP:
            self._initialized = False
            return None

        self._contact_y = new_y_norm
        self._prev_pts = new_pts.reshape(-1, 1, 2)
        self._prev_gray = frame_gray
        return self._contact_y

    def _init_tracker(self, frame_gray, px, py, fh, fw):
        half = self.PATCH_SIZE // 2
        x1, x2 = max(0, int(px - half)), min(fw, int(px + half))
        y1, y2 = max(0, int(py - half)), min(fh, int(py + half))
        patch = frame_gray[y1:y2, x1:x2]
        if patch.size == 0: return

        pts = cv2.goodFeaturesToTrack(patch, maxCorners=20, qualityLevel=0.01, minDistance=3)
        if pts is not None and len(pts) >= self.MIN_VALID_POINTS:
            pts[:, 0, 0] += x1
            pts[:, 0, 1] += y1
            self._prev_pts = pts
            self._prev_gray = frame_gray
            self._initialized = True
            self._contact_y = py / fh

class DualAnchorTracker:
    """
    P1 hip 앵커 + P2 hip 앵커를 LK 광학흐름으로 독립 추적.
    ORB fallback 추가하여 특징점이 적은 애니메이션/3D 대응 강화.
    """

    PATCH_SIZE     = 48
    MIN_VALID_PTS  = 3
    MAX_Y_JUMP     = 0.20
    MAX_DRIFT      = 0.30
    REINIT_FRAMES  = 45

    LK_PARAMS = dict(
        winSize=(19, 19),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
    )
    GFT_PARAMS = dict(
        maxCorners=12,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5
    )

    def __init__(self):
        self._prev_gray    = None
        self._pts1         = None
        self._pts2         = None
        self._anchor1      = None
        self._anchor2      = None
        self._anchor1_init = None
        self._anchor2_init = None
        self._frame_count  = 0
        self._initialized  = False
        self._is_manual    = False
        self._frame_diag   = 1.0
        self._orb = cv2.ORB_create(nfeatures=100)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._patch1       = None
        self._patch2       = None
        self._last_ncc     = 0.0

    def reset(self, p1_hip_px, p2_hip_px, frame_gray, frame_h, frame_w, is_manual=False):
        self._is_manual = is_manual
        self._frame_diag = float(np.sqrt(frame_h**2 + frame_w**2))
        self._frame_count = 0
        self._initialized = False
        self._frame_h = frame_h

        ret1 = self._init_anchor(frame_gray, p1_hip_px, frame_h, frame_w) if p1_hip_px else None
        ret2 = self._init_anchor(frame_gray, p2_hip_px, frame_h, frame_w) if p2_hip_px else None

        if ret1 is not None:
            self._pts1, self._patch1 = ret1
            self._anchor1 = self._anchor1_init = (
                float(np.median(self._pts1[:, 0, 0])), float(np.median(self._pts1[:, 0, 1]))
            )
            self._prev_gray = frame_gray
            self._initialized = True
            
        if ret2 is not None:
            self._pts2, self._patch2 = ret2
            self._anchor2 = self._anchor2_init = (
                float(np.median(self._pts2[:, 0, 0])), float(np.median(self._pts2[:, 0, 1]))
            )

    def get_ncc_score(self):
        """Returns the latest Normalized Cross-Correlation metadata for manual tracking stability."""
        return self._last_ncc

    def update(self, frame_gray, yolo_result=None, frame_h=None, frame_w=None, is_duplicate=False):
        if not self._initialized or self._prev_gray is None:
            return None

        # 중복 프레임(애니메이션)일 경우 추적 스킵하고 이전 상태 유지
        if is_duplicate:
            return self._current_dist()

        self._frame_count += 1

        # REINIT_FRAMES 마다 YOLO로 앵커 재확인 (단, Manual 모드에서는 LK와 템플릿만 믿고 YOLO 의존 제거)
        if (not self._is_manual and self._frame_count % self.REINIT_FRAMES == 0
                and yolo_result is not None and frame_h and frame_w):
            p1_px, p2_px = self._extract_hip_px(yolo_result)
            
            # 둘 다 유효하거나, 하나라도 유효하면 해당 지점들로 reset (None인 쪽은 기존 추적점 유지)
            if p1_px or p2_px:
                self.reset(p1_px or self._anchor1, p2_px or self._anchor2, frame_gray, frame_h, frame_w, is_manual=self._is_manual)
                return self._current_dist()

        # LK 추적
        pts1_new = self._track_pts(self._prev_gray, frame_gray, self._pts1)
        pts2_new = self._track_pts(self._prev_gray, frame_gray, self._pts2)

        if pts1_new is None:
            self._initialized = False
            return None

        a1 = (float(np.median(pts1_new[:, 0, 0])), float(np.median(pts1_new[:, 0, 1])))
        if pts2_new is not None:
            a2 = (float(np.median(pts2_new[:, 0, 0])), float(np.median(pts2_new[:, 0, 1])))
        else:
            a2 = None

        # 과도한 drift 감지 → 재초기화 필요 신호
        if self._anchor1_init:
            drift1 = np.sqrt((a1[0]-self._anchor1_init[0])**2 + (a1[1]-self._anchor1_init[1])**2)
            if drift1 / self._frame_diag > self.MAX_DRIFT:
                self._initialized = False
                return None
                
        if self._anchor2_init and a2:
            drift2 = np.sqrt((a2[0]-self._anchor2_init[0])**2 + (a2[1]-self._anchor2_init[1])**2)
            if drift2 / self._frame_diag > self.MAX_DRIFT:
                a2 = None # P2 유실 처리, P1만 유지

        self._pts1    = pts1_new
        self._pts2    = pts2_new
        self._anchor1 = a1
        self._anchor2 = a2
        self._prev_gray = frame_gray
        self._last_ncc = self._calculate_ncc(frame_gray, a1, self._patch1) if self._is_manual and a1 and self._patch1 is not None else 1.0

        return self._current_dist()

    def get_anchor_pixels(self):
        """디버그 오버레이용 현재 앵커 픽셀 반환."""
        return self._anchor1, self._anchor2

    def is_manual(self):
        """현재 앵커가 수동으로 설정되었는지 반환."""
        return self._is_manual
        
    def get_p1_y_norm(self):
        """YOLO fallback용 y좌표 정규화 반환"""
        if self._anchor1 and hasattr(self, '_frame_h'):
            return self._anchor1[1] / self._frame_h
        return None

    # ── private ────────────────────────────────────────────────────────────

    def _init_anchor(self, frame_gray, hip_px, frame_h, frame_w):
        cx, cy = int(hip_px[0]), int(hip_px[1])
        half = self.PATCH_SIZE // 2
        x1, x2 = max(0, cx - half), min(frame_w, cx + half)
        y1, y2 = max(0, cy - half), min(frame_h, cy + half)
        patch = frame_gray[y1:y2, x1:x2]
        if patch.size == 0 or x2 <= x1 or y2 <= y1:
            return None
        pts = cv2.goodFeaturesToTrack(patch, **self.GFT_PARAMS)
        if pts is None or len(pts) < self.MIN_VALID_PTS:
            return None
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        return pts, patch.copy()

    def _calculate_ncc(self, frame_gray, px, patch):
        if patch.size == 0 or len(patch.shape) != 2: return 0.0
        h, w = patch.shape
        cx, cy = int(px[0]), int(px[1])
        x1, y1 = max(0, cx - w//2), max(0, cy - h//2)
        x2, y2 = min(frame_gray.shape[1], x1 + w), min(frame_gray.shape[0], y1 + h)
        cur_patch = frame_gray[y1:y2, x1:x2]
        if cur_patch.shape != patch.shape: return 0.0
        res = cv2.matchTemplate(cur_patch, patch, cv2.TM_CCOEFF_NORMED)
        return float(np.max(res))

    def _track_pts(self, prev_gray, cur_gray, pts):
        if pts is None or len(pts) < self.MIN_VALID_PTS:
            return None
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, pts, None, **self.LK_PARAMS
        )
        
        good = status.ravel() == 1 if status is not None else np.array([])
        if status is None or good.sum() < self.MIN_VALID_PTS:
            # LK 실패 시 ORB Fallback
            return self._track_pts_orb(prev_gray, cur_gray, pts)
            
        return next_pts[good].reshape(-1, 1, 2)

    def _track_pts_orb(self, prev_gray, cur_gray, pts):
        """ORB 특징점 매칭을 이용한 백업 추적 (LK 실패 시)."""
        kp1, des1 = self._orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self._orb.detectAndCompute(cur_gray, None)
        if des1 is None or des2 is None: return None
        
        matches = self._matcher.match(des1, des2)
        if len(matches) < 4: return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None: return None
        
        # 이전 포인트들에 변환 행렬 적용
        next_pts = cv2.perspectiveTransform(pts, M)
        return next_pts

    def _current_dist(self):
        if self._anchor1 is None or self._anchor2 is None:
            return None
        dx = self._anchor1[0] - self._anchor2[0]
        dy = self._anchor1[1] - self._anchor2[1]
        return float(np.sqrt(dx*dx + dy*dy)) / self._frame_diag

    def _extract_hip_px(self, yolo_result):
        kp1 = yolo_result.get('keypoints')
        kp2 = yolo_result.get('secondary_keypoints')
        p1 = p2 = None
        if kp1 is not None and len(kp1) > 12:
            lh, rh = kp1[11], kp1[12]
            if lh[2] > 0.30 and rh[2] > 0.30:
                p1 = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        if kp2 is not None and len(kp2) > 12:
            lh2, rh2 = kp2[11], kp2[12]
            if lh2[2] > 0.30 and rh2[2] > 0.30:
                p2 = ((lh2[0]+rh2[0])/2, (lh2[1]+rh2[1])/2)
        return p1, p2

class MotionExtractor:
    def __init__(self, flow_estimator):
        self.flow_estimator = flow_estimator
        self.camera_comp = CameraMotionCompensator()

    def extract_velocity_signal(self, prev_frame, curr_frame, roi_fractions, precomputed_flow=None):
        flow = precomputed_flow if precomputed_flow is not None else self.flow_estimator.estimate_flow(prev_frame, curr_frame)
        h, w = prev_frame.shape[:2]
        y1, y2 = int(roi_fractions[0] * h), int(roi_fractions[1] * h)
        x1, x2 = int(roi_fractions[2] * w), int(roi_fractions[3] * w)
        roi_height_px = max(y2 - y1, 1)
        affine = self.camera_comp.estimate_from_flow(flow, y1, y2, x1, x2)
        flow_roi = flow[:, y1:y2, x1:x2]
        ty_corr = affine.ty if affine.valid and abs(affine.ty) >= 0.5 else 0.0
        flow_y = flow_roi[1] - ty_corr
        flow_x = flow_roi[0]
        flow_mag = torch.sqrt(flow_x ** 2 + flow_y ** 2)
        mask = flow_mag > 0.08
        if mask.sum() < 10: return 0.0, 0.0, roi_height_px, False
        velocity = -flow_y[mask].mean().item()
        magnitude = flow_mag[mask].mean().item()
        return velocity, magnitude, roi_height_px, False
