import os
import sys
import base64
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
    SceneBoundaryHandler,
    YoloPoseTracker,
    SceneTypeDetector,
    ContactPointTracker,
    SceneAnchorSelector,
    DualAnchorTracker,
    YOLO_AVAILABLE,
    DEVICE,
    RESIZE_WIDTH
)
from scipy.signal import savgol_filter

# Scene count threshold for UI classification (single vs multi-scene display).
# ROI strategy is always per-scene regardless of this value.
MULTI_SCENE_THRESHOLD = 3


# ── Dataclasses for interactive preprocessing ─────────────────────────────

from dataclasses import dataclass as _dataclass


@_dataclass
class SceneConfig:
    """씬별 사용자 설정 (인터랙티브 전처리 다이얼로그)."""
    enabled       : bool  = True    # F3: 스트로크 생성 여부
    mode          : str   = 'auto'  # 'auto' | 'manual' | 'optical_flow'
    p1_person_idx : int   = 0       # F1: P1 인물 인덱스 (per_scene_persons 기준)
    p2_person_idx : int   = 1       # F1: P2 인물 인덱스 (-1=없음)
    p1_hip_px     : tuple = None    # F2: 수동 P1 골반 픽셀 (x,y)
    p2_hip_px     : tuple = None    # F2: 수동 P2 골반 픽셀 (x,y)


@_dataclass
class Pass1Result:
    """Pass 1 분석 결과 컨테이너."""
    scene_boundaries   : list   # [(start, end), ...]
    per_scene_rois     : list   # [roi_fraction, ...]
    per_scene_anchors  : list   # [anchor_dict or None, ...]
    per_scene_persons  : list   # [[person_dict, ...], ...]  씬별 탐지 인물
    per_scene_previews : list   # [base64_jpeg_str, ...]  씬별 대표 프레임
    fps                : float
    total_frames       : int
    video_duration_ms  : int
    video_type         : str
    yolo_tracker       : object  # YoloPoseTracker (or None)
    flow_estimator     : object  # OpticalFlowEstimator


@_dataclass
class UserConfig:
    """유저 인터랙션 설정."""
    scene_configs : list  # [SceneConfig, ...]

    @classmethod
    def auto_from_pass1(cls, r: 'Pass1Result') -> 'UserConfig':
        """자동 기본 설정 — 모든 씬 활성화, person 인덱스 기본값."""
        return cls(scene_configs=[SceneConfig() for _ in r.scene_boundaries])


def _extract_scene_frame(video_path: str, frame_idx: int):
    """지정 프레임 추출 후 RESIZE_WIDTH로 리사이즈. 실패 시 None."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1])))


def _draw_scene_preview(frame_bgr, persons: list) -> str:
    """씬 대표 프레임에 인물 번호 bbox 오버레이 → base64 JPEG 반환."""
    if frame_bgr is None:
        return ""
    vis = frame_bgr.copy()
    # P1=노랑, P2=하늘, 나머지=회색
    COLORS = [(0, 215, 255), (255, 200, 0), (150, 150, 150), (150, 150, 150)]
    for idx, person in enumerate(persons[:4]):
        color = COLORS[min(idx, len(COLORS) - 1)]
        if person.get('bbox'):
            x1, y1, x2, y2 = (int(v) for v in person['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"P{idx+1}", (x1 + 4, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if person.get('hip_px'):
            hx, hy = int(person['hip_px'][0]), int(person['hip_px'][1])
            cv2.circle(vis, (hx, hy), 5, color, -1)
    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ── Debug overlay helpers ──────────────────────────────────────────────────

def _draw_debug_overlay(frame_bgr, current_roi, velocity, magnitude, zoom_detected,
                        frame_idx, fps, bbox=None, keypoints=None,
                        secondary_hip_y=None, secondary_bbox=None, rel_dist=None,
                        secondary_keypoints=None, scene_type=None,
                        anchor_p1=None, anchor_p2=None, anchor_dist=None):
    """
    ROI 박스 + velocity 방향 바 + zoom 상태를 프레임 위에 그려 JPEG base64로 반환.
    - ROI: 밝은 라임 그린 (#00FF80)
    - Velocity 바: 위=초록, 아래=주황 (프레임 오른쪽 끝)
    - ZOOM 배지: 빨강 (zoom 감지 시)
    - bbox: 인물 bbox (노란색, YOLO 탐지 시)
    - keypoints: COCO 17 keypoints ndarray (P1 — 선명한 색상)
    - secondary_keypoints: COCO 17 keypoints ndarray (P2 — 어두운 색상)
    - scene_type: 장면 유형 문자열 (HIP_HIP/BJ/POV/OPTICAL_FLOW)
    """
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    # ── ROI 박스 ──
    y1 = int(current_roi[0] * h)
    y2 = int(current_roi[1] * h)
    x1 = int(current_roi[2] * w)
    x2 = int(current_roi[3] * w)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 128), 2)
    cv2.putText(vis, "ROI", (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)

    # ── Velocity 세로 바 (우측 끝) ──
    bx, bw = w - 18, 10
    bar_top, bar_bot = 8, h - 8
    bar_mid = (bar_top + bar_bot) // 2
    cv2.rectangle(vis, (bx, bar_top), (bx + bw, bar_bot), (35, 35, 35), -1)
    # 중앙선
    cv2.line(vis, (bx, bar_mid), (bx + bw, bar_mid), (80, 80, 80), 1)
    vel_clamped = max(-1.0, min(1.0, velocity * 3.0))
    bar_len = int(abs(vel_clamped) * (bar_bot - bar_mid))
    if vel_clamped > 0:    # 위 (insert)
        cv2.rectangle(vis, (bx, bar_mid - bar_len), (bx + bw, bar_mid),
                      (0, 230, 80), -1)
    elif vel_clamped < 0:  # 아래 (withdraw)
        cv2.rectangle(vis, (bx, bar_mid), (bx + bw, bar_mid + bar_len),
                      (60, 120, 255), -1)

    # ── Magnitude 점 ──
    mag_v = min(1.0, magnitude * 2.0)
    mag_color = (0, int(255 * mag_v), int(180 * (1.0 - mag_v)))
    cv2.circle(vis, (bx + bw // 2, bar_top - 6 if bar_top > 10 else bar_top + 10),
               5, mag_color, -1)

    # ── ZOOM 배지 ──
    if zoom_detected:
        cv2.rectangle(vis, (2, 2), (62, 24), (0, 0, 200), -1)
        cv2.putText(vis, "ZOOM", (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── YOLO 인물 bbox (노란색) ──
    if bbox is not None:
        bx1, by1, bx2, by2 = (int(v) for v in bbox)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 215, 255), 1)
        cv2.putText(vis, "P1", (bx1 + 2, max(by1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 215, 255), 1, cv2.LINE_AA)

    # ── 전신 키포인트 오버레이 ──────────────────────────────────────────────
    if keypoints is not None and len(keypoints) >= 13:
        from algorithms import YoloPoseTracker as _YPT
        # 키포인트별 색상: 얼굴(흰/시안/황), 어깨(녹), 골반(오렌지), 무릎(파랑), 발목(시안)
        _KP_STYLE = {
            0:  ((220, 220, 220), 'N',  3),   # nose (연회색)
            1:  ((0,   220, 220), 'LE', 3),   # left_eye (시안)
            2:  ((0,   220, 220), 'RE', 3),   # right_eye
            3:  ((0,   180, 255), 'LA', 3),   # left_ear (하늘색)
            4:  ((0,   180, 255), 'RA', 3),   # right_ear
            5:  ((80, 230, 80),  'LS', 4),   # left_shoulder
            6:  ((80, 230, 80),  'RS', 4),   # right_shoulder
            11: ((30, 120, 255), 'LH', 5),   # left_hip
            12: ((30, 120, 255), 'RH', 5),   # right_hip
            13: ((200, 80, 40),  'LK', 4),   # left_knee
            14: ((200, 80, 40),  'RK', 4),   # right_knee
            15: ((0,  200, 200), 'LA', 3),   # left_ankle
            16: ((0,  200, 200), 'RA', 3),   # right_ankle
        }
        for ki, (color, label, radius) in _KP_STYLE.items():
            if ki < len(keypoints):
                kx, ky, kc = keypoints[ki]
                if kc > 0.25:
                    cv2.circle(vis, (int(kx), int(ky)), radius, color, -1)
                    cv2.circle(vis, (int(kx), int(ky)), radius + 1, color, 1)

        # 골반 중심선 (수평 점선)
        lhx, lhy, lhc = keypoints[_YPT.LEFT_HIP]
        rhx, rhy, rhc = keypoints[_YPT.RIGHT_HIP]
        if lhc > 0.25 and rhc > 0.25:
            hip_y_px = int((lhy + rhy) / 2)
            for sx in range(0, w - 4, 8):
                cv2.line(vis, (sx, hip_y_px), (min(sx + 4, w), hip_y_px),
                         (190, 190, 190), 1)

        # 스켈레톤 연결선 (얼굴-어깨 포함)
        _SKELETON = [(0, 5), (0, 6), (5, 11), (6, 12), (5, 6), (11, 12)]
        for ka, kb in _SKELETON:
            if ka < len(keypoints) and kb < len(keypoints):
                ax, ay, ac = keypoints[ka]
                bx, by, bc = keypoints[kb]
                if ac > 0.25 and bc > 0.25:
                    cv2.line(vis, (int(ax), int(ay)), (int(bx), int(by)),
                             (120, 120, 120), 1)

    # ── P2 키포인트 오버레이 (채도 낮은 색상) ──────────────────────────────
    if secondary_keypoints is not None and len(secondary_keypoints) >= 13:
        _KP_STYLE_P2 = {
            0:  ((130, 130, 130), 2),  # nose (어두운 회)
            1:  ((0,  120, 120),  2),  # left_eye (어두운 시안)
            2:  ((0,  120, 120),  2),  # right_eye
            3:  ((0,   90, 150),  2),  # left_ear
            4:  ((0,   90, 150),  2),  # right_ear
            5:  ((40, 140, 40),  3),   # left_shoulder (어두운 녹)
            6:  ((40, 140, 40),  3),   # right_shoulder
            11: ((20, 60, 180),  4),   # left_hip (어두운 파랑)
            12: ((20, 60, 180),  4),   # right_hip
            13: ((110, 45, 20),  3),   # left_knee
            14: ((110, 45, 20),  3),   # right_knee
        }
        for ki, (color, radius) in _KP_STYLE_P2.items():
            if ki < len(secondary_keypoints):
                kx, ky, kc = secondary_keypoints[ki]
                if kc > 0.25:
                    cv2.circle(vis, (int(kx), int(ky)), radius, color, -1)
        # P2 어깨-골반 연결선
        for ka, kb in [(5, 11), (6, 12)]:
            if ka < len(secondary_keypoints) and kb < len(secondary_keypoints):
                ax, ay, ac = secondary_keypoints[ka]
                bx_, by_, bc = secondary_keypoints[kb]
                if ac > 0.25 and bc > 0.25:
                    cv2.line(vis, (int(ax), int(ay)), (int(bx_), int(by_)),
                             (70, 70, 70), 1)

    # ── Secondary 인물 bbox + hip_y 수평선 ──
    if secondary_bbox is not None:
        sbx1, sby1, sbx2, sby2 = (int(v) for v in secondary_bbox)
        cv2.rectangle(vis, (sbx1, sby1), (sbx2, sby2), (255, 200, 0), 1)
        cv2.putText(vis, "P2", (sbx1 + 2, max(sby1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 0), 1, cv2.LINE_AA)
    if secondary_hip_y is not None:
        sy_px = int(secondary_hip_y * h)
        for sx in range(0, w - 4, 8):
            cv2.line(vis, (sx, sy_px), (min(sx + 4, w), sy_px),
                     (255, 200, 0), 1)
    if rel_dist is not None:
        cv2.putText(vis, f"DUAL D:{rel_dist:.3f}", (4, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1, cv2.LINE_AA)

    # ── 장면 유형 표시 ──
    if scene_type and scene_type != 'OPTICAL_FLOW':
        _st_colors = {'HIP_HIP': (60, 200, 60), 'BJ': (60, 180, 255), 'POV': (200, 100, 255)}
        _st_color = _st_colors.get(scene_type, (200, 200, 50))
        cv2.putText(vis, f"MODE:{scene_type}", (4, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _st_color, 1, cv2.LINE_AA)

    # ── DualAnchor 시각화 (Manual Hip) ──
    if anchor_p1 is not None:
        ax1, ay1 = int(anchor_p1[0]), int(anchor_p1[1])
        # 빨간색 크로스헤어
        cv2.line(vis, (ax1 - 12, ay1), (ax1 + 12, ay1), (0, 0, 255), 2)
        cv2.line(vis, (ax1, ay1 - 12), (ax1, ay1 + 12), (0, 0, 255), 2)
        cv2.circle(vis, (ax1, ay1), 3, (255, 255, 255), -1)
        # 상단에 강조 텍스트
        cv2.putText(vis, "MANUAL HIP ACTIVE", (w // 2 - 75, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    if anchor_p2 is not None:
        ax2, ay2 = int(anchor_p2[0]), int(anchor_p2[1])
        # 파란색 크로스헤어
        cv2.line(vis, (ax2 - 12, ay2), (ax2 + 12, ay2), (255, 0, 0), 2)
        cv2.line(vis, (ax2, ay2 - 12), (ax2, ay2 + 12), (255, 0, 0), 2)
        cv2.circle(vis, (ax2, ay2), 3, (255, 255, 255), -1)

    if anchor_p1 is not None and anchor_p2 is not None:
        cv2.line(vis, (int(anchor_p1[0]), int(anchor_p1[1])),
                 (int(anchor_p2[0]), int(anchor_p2[1])), (0, 255, 255), 1)

    if anchor_dist is not None:
        cv2.putText(vis, f"ANC D:{anchor_dist:.3f}", (4, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    # ── 시간 레이블 ──
    t_sec = frame_idx / max(fps, 1.0)
    cv2.putText(vis, f"{t_sec:.1f}s", (4, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


def _draw_result_graph(position_normalized, actions, segments, fps, total_frames):
    """
    처리 완료 후 position 신호 + action 포인트 그래프를 PNG base64로 반환.
    - 배경: 세그먼트 타입별 색상 (ACTIVE=녹, QUIET=회, TRANSITION=황)
    - 흰 선: 정규화된 position 신호
    - 파란 원: action 포인트
    """
    W, H = 512, 130
    vis = np.full((H, W, 3), (18, 18, 22), dtype=np.uint8)
    n = max(len(position_normalized), 1)

    seg_colors = {
        'ACTIVE':     (0,  50,  0),
        'QUIET':      (30, 30, 30),
        'TRANSITION': (45, 40,  0),
    }
    for start, end, stype in segments:
        x1 = int(start / n * W)
        x2 = int(end / n * W)
        cv2.rectangle(vis, (x1, 0), (x2, H), seg_colors.get(stype, (20, 20, 20)), -1)

    # 격자선 (0, 50, 100)
    for pct in (0, 50, 100):
        gy = int((1.0 - pct / 100.0) * (H - 4)) + 2
        cv2.line(vis, (0, gy), (W, gy), (50, 50, 50), 1)

    # Position 신호
    pts = []
    for xi in range(W):
        fi = min(int(xi / W * n), n - 1)
        pos = float(np.clip(position_normalized[fi], 0, 100))
        y = int((1.0 - pos / 100.0) * (H - 6)) + 3
        pts.append((xi, y))
    for i in range(1, len(pts)):
        cv2.line(vis, pts[i - 1], pts[i], (210, 210, 210), 1)

    # Action 포인트
    for action in actions:
        fi = int(action['at'] / 1000.0 * fps)
        xi = int(fi / n * W)
        if 0 <= xi < W:
            pos = float(np.clip(action['pos'], 0, 100))
            y = int((1.0 - pos / 100.0) * (H - 6)) + 3
            cv2.circle(vis, (xi, y), 3, (255, 80, 30), -1)

    # 범례 텍스트
    cv2.putText(vis, "100", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    cv2.putText(vis, "  0", (2, H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    _, buf = cv2.imencode('.png', vis)
    return base64.b64encode(buf).decode('utf-8')


def _build_quality_mask(quality_records: list,
                        min_conf: float = 0.35,
                        low_conf_run: int = 15) -> np.ndarray:
    """
    저품질 프레임 마스크 (True = velocity 억제 대상).
    조건:
      - near_boundary = True (씬 경계 ±30프레임)
      - conf < min_conf 가 low_conf_run 프레임 이상 연속
    """
    n = len(quality_records)
    mask = np.zeros(n, dtype=bool)

    for i, qr in enumerate(quality_records):
        if qr.get('near_boundary', False):
            mask[i] = True

    run = 0
    run_start = 0
    for i, qr in enumerate(quality_records):
        if qr.get('conf', 0.0) < min_conf:
            if run == 0:
                run_start = i
            run += 1
        else:
            if run >= low_conf_run:
                mask[run_start:i] = True
            run = 0
    if run >= low_conf_run:
        mask[run_start:n] = True

    return mask


def _postprocess_velocity(velocity_signal: np.ndarray,
                          quality_mask: np.ndarray,
                          taper_frames: int = 8) -> np.ndarray:
    """
    저품질 구간의 velocity를 0으로 fade.
    전체의 60% 이상 억제 대상이면 원본 반환 (보수적 fallback).
    """
    if quality_mask.sum() > len(quality_mask) * 0.60:
        return velocity_signal

    result = velocity_signal.copy()
    n = len(result)
    i = 0
    while i < n:
        if quality_mask[i]:
            j = i
            while j < n and quality_mask[j]:
                j += 1
            seg_len = j - i
            taper = min(taper_frames, seg_len // 2)
            for k in range(taper):
                result[i + k] *= (k / taper_frames)
            mid_s = i + taper
            mid_e = j - taper
            if mid_s < mid_e:
                result[mid_s:mid_e] = 0.0
            for k in range(taper):
                idx = j - taper + k
                if idx < n:
                    result[idx] *= (k / taper_frames)
            i = j
        else:
            i += 1
    return result


def _normalize_stroke_amplitude(actions: list,
                                segments: list,
                                fps: float,
                                min_range: int = 45) -> list:
    """
    ACTIVE 씬별 stroke pos_range < min_range 이면 중앙(50) 기준 stretch.
    씬에 액션 < 3개이거나 range < 5이면 스킵.
    """
    if not actions:
        return actions

    result = [dict(a) for a in actions]

    for seg_start, seg_end, seg_type in segments:
        if seg_type != 'ACTIVE':
            continue
        t_start = seg_start / fps * 1000.0
        t_end   = seg_end   / fps * 1000.0
        idxs = [i for i, a in enumerate(result) if t_start <= a['at'] <= t_end]
        if len(idxs) < 3:
            continue
        positions = [result[i]['pos'] for i in idxs]
        pos_min, pos_max = min(positions), max(positions)
        pos_range = pos_max - pos_min
        if pos_range >= min_range or pos_range < 5:
            continue
        center = (pos_max + pos_min) / 2.0
        if center < 25 or center > 75:
            continue  # 의도적 하단/상단 집중 패턴 (Remilia 등) → stretch 스킵
        scale  = min_range / pos_range
        for i in idxs:
            old = result[i]['pos']
            result[i]['pos'] = int(np.clip(center + (old - center) * scale, 0, 100))

    return result


def _blend_yolo_pose(velocity_flow, yolo_hip_ys, yolo_ref_lens, yolo_confs,
                     min_conf=0.5):
    """
    YOLO hip_y 미분 → velocity로 변환하여 optical flow velocity와 혼합.

    신뢰도 >= min_conf 프레임: YOLO velocity 우선 (신뢰도 가중 혼합)
    신뢰도 < min_conf 프레임: optical flow 그대로 유지

    scale 자동 맞춤: YOLO 신호를 flow 진폭에 맞춰 정규화.
    """
    n = len(velocity_flow)
    conf_arr = np.array([c if c is not None else 0.0 for c in yolo_confs],
                        dtype=np.float64)

    yolo_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hy_c, hy_p = yolo_hip_ys[i], yolo_hip_ys[i - 1]
        rl = yolo_ref_lens[i]
        if hy_c is not None and hy_p is not None and rl and rl > 0.04:
            yolo_vel[i] = -(hy_c - hy_p) / rl  # 위로 이동 = 양수 (stroke in)

    confident = conf_arr > min_conf
    if confident.sum() < 15:
        return velocity_flow  # YOLO 탐지 부족 → flow 그대로

    flow_abs = np.abs(velocity_flow[confident])
    yolo_abs = np.abs(yolo_vel[confident])
    nonzero = yolo_abs > 1e-6
    if nonzero.sum() < 8:
        return velocity_flow

    scale = float(np.median(flow_abs[nonzero] / yolo_abs[nonzero]))
    if scale <= 0:
        return velocity_flow

    blended = velocity_flow.copy()
    for i in range(n):
        c = conf_arr[i]
        if c >= min_conf:
            blended[i] = c * (yolo_vel[i] * scale) + (1.0 - c) * velocity_flow[i]

    return blended


def _build_direct_hip_position(yolo_hip_ys: list, yolo_confs: list,
                               segments: list, fps: float,
                               min_conf: float = 0.5) -> tuple:
    """
    G1: YOLO hip_y를 직접 위치 신호로 변환 (적분 없음).

    hip_y: 0=화면 위, 1=화면 아래
    funscript: 0=스트로크 in (아래), 100=스트로크 out (위)
    → raw_pos[i] = (1 - hip_y[i]) * 100

    반환: (position_array, coverage_ratio) 또는 (None, 0.0)
    coverage_ratio: 고신뢰도 프레임 비율 (0~1)
    """
    n = len(yolo_hip_ys)
    if n < 10:
        return None, 0.0

    conf_arr = np.array([c if c is not None else 0.0 for c in yolo_confs],
                        dtype=np.float64)
    high_conf = conf_arr >= min_conf
    coverage = float(high_conf.sum()) / max(n, 1)

    if coverage < 0.30:
        return None, coverage  # 탐지 부족 → fallback

    # 고신뢰도 프레임에서 직접 위치 추출 (절대 위치)
    raw_pos = np.full(n, np.nan)
    for i in range(n):
        if high_conf[i] and yolo_hip_ys[i] is not None:
            raw_pos[i] = (1.0 - float(yolo_hip_ys[i])) * 100.0

    valid_mask = ~np.isnan(raw_pos)
    if valid_mask.sum() < 2:
        return None, coverage

    # NaN 선형 보간
    x = np.arange(n, dtype=float)
    raw_pos = np.interp(x, x[valid_mask], raw_pos[valid_mask])

    # 가벼운 스무딩: 고주파 노이즈 제거, 저주파 절대 위치 보존
    smooth_win = max(5, int(fps * 0.1)) | 1  # ~3 프레임 @ 30fps
    if n > smooth_win:
        from scipy.signal import savgol_filter as _sgf
        raw_pos = _sgf(raw_pos, smooth_win, 1)

    # ACTIVE 세그먼트별 — 자연 중심(위치 바이어스) 보존 정규화
    # 기존 normalize_per_segment와 달리: 세그먼트 중앙값을 50으로 강제 이동하지 않음
    position = raw_pos.copy()
    for start, end, seg_type in segments:
        end = min(end, n)
        if start >= end or seg_type != 'ACTIVE':
            continue
        seg = raw_pos[start:end]

        # 고신뢰도 프레임만으로 통계 계산
        seg_conf = high_conf[start:end]
        if seg_conf.sum() < 5:
            continue
        seg_valid = seg[seg_conf]

        p2 = np.percentile(seg_valid, 2)
        p98 = np.percentile(seg_valid, 98)
        p_range = p98 - p2
        if p_range < 3.0:
            continue  # 움직임 거의 없음 → 보간값 그대로 유지

        # 자연 중심 보존: raw hip_y 중앙값이 편향되어 있으면 그대로 반영
        # (예: Remilia hip_y median = 0.83 → natural_center = 17)
        natural_center = float(np.median(seg_valid))  # 0~100 funscript 공간

        # 스케일링: range를 확장하되 center는 natural_center 유지
        # 클리핑 방지: 중심에서 0과 100 중 가까운 쪽 거리로 반진폭 제한
        max_half = min(natural_center, 100.0 - natural_center)
        max_half = max(max_half, 15.0)  # 최소 진폭 보장
        half_out = min(max_half, p_range * 1.5)  # 1.5x 확장 허용

        normalized_seg = (
            (seg - np.median(seg_valid)) / (p_range / 2.0) * half_out
            + natural_center
        )
        position[start:end] = np.clip(normalized_seg, 0, 100)

    return position, coverage


def _estimate_dual_scale(flow_vel: np.ndarray, dual_vel: np.ndarray,
                         mask: np.ndarray):
    """dual velocity를 optical flow scale로 맞추는 비율 추정."""
    valid = mask & (np.abs(dual_vel) > 1e-6)
    if valid.sum() < 10:
        return None
    flow_abs = np.abs(flow_vel[valid])
    dual_abs = np.abs(dual_vel[valid])
    return float(np.median(flow_abs / dual_abs))


def _compute_bj_dist(yolo_result: dict, frame_h: int):
    """
    P1 head(nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4) ↔ P2 hip 정규화 거리.
    BJ 신호 계산용: 거리 감소 → 더 깊게 → velocity 양수.
    반환: 0~1 정규화 거리 (frame_h 기준) or None
    """
    kp1 = yolo_result.get('keypoints')
    p2_hip_y = yolo_result.get('secondary_hip_y')

    if kp1 is None or p2_hip_y is None:
        return None

    # P1 head: nose → left_eye → right_eye → left_ear → right_ear 순으로 첫 유효 키포인트
    head_y = None
    for ki in [0, 1, 2, 3, 4]:
        if ki < len(kp1) and kp1[ki][2] > 0.20:
            head_y = kp1[ki][1] / frame_h   # 정규화 (0~1)
            break

    if head_y is None:
        return None

    return abs(head_y - p2_hip_y)   # 정규화 거리 (0~1)


def _blend_dual_pose(velocity_flow: np.ndarray,
                     rel_dists: list,
                     yolo_is_duals: list,
                     rolling_window: int = 120) -> np.ndarray:
    """
    두 골반 상대 거리(rel_dist) 미분 → velocity로 변환하여 velocity_flow 대체.

    rel_dist 감소(두 사람이 가까워짐) → 음수 delta → velocity 양수(스트로크 in)
    rolling_window 프레임 기준 min/max 정규화로 씬 전환 후 독립 진폭 유지.
    dual coverage < 20% 이면 flow 그대로 반환.
    """
    n = len(velocity_flow)
    dual_mask = np.array([bool(b) for b in yolo_is_duals], dtype=bool)

    if dual_mask.sum() < n * 0.20:
        return velocity_flow

    dist_arr = np.array([d if d is not None else np.nan for d in rel_dists],
                        dtype=np.float64)

    # rolling min/max 정규화 (씬 별 독립 진폭)
    roll_min = np.full(n, np.nan)
    roll_max = np.full(n, np.nan)
    for i in range(n):
        w_start = max(0, i - rolling_window + 1)
        window = dist_arr[w_start:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 5:
            roll_min[i] = valid.min()
            roll_max[i] = valid.max()

    roll_range = roll_max - roll_min
    roll_range = np.where(
        np.isnan(roll_range) | (roll_range < 0.02), 0.02, roll_range
    )

    # 미분: rel_dist 감소 → 양수 velocity
    dual_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(dist_arr[i]) or np.isnan(dist_arr[i - 1]):
            continue
        delta = dist_arr[i] - dist_arr[i - 1]
        rr = roll_range[i] if not np.isnan(roll_range[i]) else 0.02
        dual_vel[i] = -delta / rr

    # flow 스케일 맞추기
    scale = _estimate_dual_scale(velocity_flow, dual_vel, dual_mask)
    if scale is None or scale <= 0:
        return velocity_flow

    blended = velocity_flow.copy()
    for i in range(n):
        if dual_mask[i] and not np.isnan(dual_vel[i]):
            blended[i] = dual_vel[i] * scale

    return blended


def _blend_bj_pose(velocity_flow: np.ndarray,
                   head_hip_dists: list,
                   bj_mask: np.ndarray,
                   rolling_window: int = 120) -> np.ndarray:
    """
    P1 head ↔ P2 hip 거리 미분 → velocity.
    거리 감소(가까워짐) → 음수 delta → velocity 양수(stroke in).
    BJ 구간 15% 미만이면 velocity_flow 그대로 반환.
    """
    n = len(velocity_flow)
    if bj_mask.sum() < n * 0.15:
        return velocity_flow

    dist_arr = np.array([d if d is not None else np.nan for d in head_hip_dists],
                        dtype=np.float64)

    # rolling min/max 정규화 (씬별 독립 진폭)
    roll_min = np.full(n, np.nan)
    roll_max = np.full(n, np.nan)
    for i in range(n):
        w_start = max(0, i - rolling_window + 1)
        window = dist_arr[w_start:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 5:
            roll_min[i] = valid.min()
            roll_max[i] = valid.max()

    roll_range = roll_max - roll_min
    roll_range = np.where(
        np.isnan(roll_range) | (roll_range < 0.01), 0.01, roll_range
    )

    # 미분: 거리 감소 → 양수 velocity
    bj_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(dist_arr[i]) or np.isnan(dist_arr[i - 1]):
            continue
        bj_vel[i] = -(dist_arr[i] - dist_arr[i - 1]) / roll_range[i]

    # flow 스케일 맞추기 (_estimate_dual_scale 재사용)
    scale = _estimate_dual_scale(velocity_flow, bj_vel, bj_mask)
    if scale is None or scale <= 0:
        return velocity_flow

    blended = velocity_flow.copy()
    for i in range(n):
        if bj_mask[i] and not np.isnan(bj_vel[i]):
            blended[i] = bj_vel[i] * scale

    return blended


def _blend_anchor_dist(velocity_flow: np.ndarray,
                       anchor_dists: list,
                       rolling_window: int = 120) -> np.ndarray:
    """
    DualAnchorTracker 거리 미분 → velocity.
    거리 감소(가까워짐) → 양수 velocity(stroke in).
    유효 프레임 30% 미만이면 velocity_flow 반환.
    """
    n = len(velocity_flow)
    valid_mask = np.array([d is not None for d in anchor_dists], dtype=bool)

    if valid_mask.sum() < n * 0.30:
        return velocity_flow

    dist_arr = np.array([d if d is not None else np.nan for d in anchor_dists],
                        dtype=np.float64)

    # rolling min/max 정규화
    roll_min = np.full(n, np.nan)
    roll_max = np.full(n, np.nan)
    for i in range(n):
        w_start = max(0, i - rolling_window + 1)
        window = dist_arr[w_start:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 5:
            roll_min[i] = valid.min()
            roll_max[i] = valid.max()

    roll_range = roll_max - roll_min
    roll_range = np.where(
        np.isnan(roll_range) | (roll_range < 0.005), 0.005, roll_range
    )

    # 미분: 거리 감소 → 양수 velocity
    anchor_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(dist_arr[i]) or np.isnan(dist_arr[i - 1]):
            continue
        anchor_vel[i] = -(dist_arr[i] - dist_arr[i - 1]) / roll_range[i]

    scale = _estimate_dual_scale(velocity_flow, anchor_vel, valid_mask)
    if scale is None or scale <= 0:
        return velocity_flow

    blended = velocity_flow.copy()
    for i in range(n):
        if valid_mask[i] and not np.isnan(anchor_vel[i]):
            blended[i] = anchor_vel[i] * scale

    return blended


def _compute_contact_pixel(yolo_result: dict, frame_h: int, frame_w: int):
    """P1 hip + P2 secondary_hip_y에서 접촉점 픽셀 좌표 계산."""
    p1_hip_y = yolo_result.get('hip_center_y')
    p2_hip_y = yolo_result.get('secondary_hip_y')
    p1_bbox  = yolo_result.get('bbox')
    p2_bbox  = yolo_result.get('secondary_bbox')

    if p1_hip_y is None or p2_hip_y is None:
        return None, None

    contact_py = (p1_hip_y + p2_hip_y) / 2.0 * frame_h
    x1 = (p1_bbox[0] + p1_bbox[2]) / 2.0 if p1_bbox else frame_w / 2.0
    x2 = (p2_bbox[0] + p2_bbox[2]) / 2.0 if p2_bbox else frame_w / 2.0
    contact_px = (x1 + x2) / 2.0
    return float(contact_px), float(contact_py)


def _blend_contact_tracking(velocity_flow: np.ndarray,
                            contact_ys: list,
                            yolo_is_duals: list) -> np.ndarray:
    """
    LK 접촉점 추적 y 시계열 → 미분 → velocity.
    LK 추적 성공 + dual 모드 프레임에서 optical flow 대체.
    """
    n = len(velocity_flow)
    cy_arr    = np.array([c if c is not None else np.nan for c in contact_ys])
    dual_mask = np.array([bool(b) for b in yolo_is_duals], dtype=bool)

    valid = ~np.isnan(cy_arr) & dual_mask
    if valid.sum() < 15:
        return velocity_flow

    contact_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(cy_arr[i]) and not np.isnan(cy_arr[i - 1]):
            contact_vel[i] = -(cy_arr[i] - cy_arr[i - 1])  # 위=양수(stroke in)

    valid_flow = np.abs(velocity_flow[valid])
    valid_cv   = np.abs(contact_vel[valid])
    nonzero    = valid_cv > 1e-6
    if nonzero.sum() < 8:
        return velocity_flow

    scale = float(np.median(valid_flow[nonzero] / valid_cv[nonzero]))
    if scale <= 0:
        return velocity_flow

    blended = velocity_flow.copy()
    for i in range(n):
        if valid[i]:
            blended[i] = contact_vel[i] * scale

    return blended


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


def _get_scene_index(frame_idx, scene_boundaries):
    """frame_idx가 속한 씬의 인덱스 반환. 해당 없으면 마지막 씬 인덱스."""
    for i, (start, end) in enumerate(scene_boundaries):
        if start <= frame_idx < end:
            return i
    return len(scene_boundaries) - 1


def pass1_analyze(video_path: str, progress_callback=None):
    """Pass 1: 씬 탐지 + ROI + 앵커 + 씬별 인물 탐지 + 프리뷰 생성.

    Returns Pass1Result or None on failure.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    video_duration_ms = int((total_frames / fps) * 1000)
    cap.release()

    if total_frames < 10:
        print("Error: Video too short")
        return None

    if progress_callback:
        progress_callback(0, 100, "Detecting scenes...")

    scene_det = QuickSceneDetector(threshold=0.7)
    scene_boundaries = scene_det.detect(video_path, sample_interval=5)
    print(f"Detected {len(scene_boundaries)} scene(s): {[(s, e) for s, e in scene_boundaries]}")

    video_type = _classify_video(scene_boundaries)
    print(f"Video classification: {video_type} ({len(scene_boundaries)} scenes)")

    if progress_callback:
        progress_callback(3, 100, f"classified:{video_type}")

    # YoloPoseTracker 초기화
    yolo_tracker = None
    if YOLO_AVAILABLE:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'models', 'yolo26x-pose.pt')
        try:
            yolo_tracker = YoloPoseTracker(model_path)
            print(f"YoloPoseTracker enabled (device: {DEVICE})")
        except Exception as e:
            yolo_tracker = None
            print(f"YoloPoseTracker init failed (skipping): {e}")

    if progress_callback:
        progress_callback(5, 100, "Detecting ROI...")

    flow_estimator = OpticalFlowEstimator()
    roi_detector = ROIDetector(flow_estimator)

    per_scene_rois     = []
    per_scene_anchors  = []
    per_scene_persons  = []
    per_scene_previews = []
    anchor_selector    = SceneAnchorSelector() if yolo_tracker is not None else None
    n_scenes           = max(len(scene_boundaries), 1)

    for i, (scene_start, scene_end) in enumerate(scene_boundaries):
        scene_length = scene_end - scene_start
        scene_sample_count = min(40, max(10, scene_length // 5))
        roi = roi_detector.detect_roi(video_path, sample_count=scene_sample_count,
                                      frame_range=(scene_start, scene_end),
                                      yolo_tracker=yolo_tracker)
        per_scene_rois.append(roi)
        print(f"  Scene {i+1} [{scene_start}-{scene_end}]: "
              f"ROI y=[{roi[0]:.2f}-{roi[1]:.2f}], x=[{roi[2]:.2f}-{roi[3]:.2f}]")

        if anchor_selector is not None:
            anchor = anchor_selector.select(
                video_path, scene_start, scene_end, yolo_tracker, n_samples=10
            )
            per_scene_anchors.append(anchor)
            print(f"  Scene {i+1} anchor: dual={anchor['is_dual']}, "
                  f"conf={anchor['confidence']:.2f}")
        else:
            per_scene_anchors.append(None)

        # 씬 대표 프레임 + 인물 탐지 (UI 프리뷰용)
        # 5개 프레임을 샘플링하여 가장 많은 인물이 탐지되는 프레임 선택
        # (단일 중간 프레임만 사용하면 특정 프레임에서 인물이 가려질 수 있음)
        scene_len = max(scene_end - scene_start, 1)
        sample_offsets = [0.20, 0.35, 0.50, 0.65, 0.80]
        best_frame, best_persons = None, []
        for t in sample_offsets:
            fi = scene_start + int(scene_len * t)
            f = _extract_scene_frame(video_path, fi)
            if f is None:
                continue

            # 이 씬에서 처음으로 유효하게 추출된 프레임을 기본 fallback으로 저장
            if best_frame is None:
                best_frame, best_persons = f, []

            ps = yolo_tracker.detect_persons_stateless(f) if yolo_tracker is not None else []
            # 더 많은 인원이 탐지되거나, 같은 수라면 총 신뢰도가 더 높은 프레임 채택
            if ps and (len(ps) > len(best_persons) or
                    (len(ps) == len(best_persons) and
                     sum(p['confidence'] for p in ps) > sum(p['confidence'] for p in best_persons))):
                best_frame, best_persons = f, ps
        frame_bgr = best_frame
        persons   = best_persons
        per_scene_persons.append(persons)
        per_scene_previews.append(_draw_scene_preview(frame_bgr, persons))

        if progress_callback:
            pct = 5 + int((i + 1) / n_scenes * 10)
            progress_callback(pct, 100, f"ROI detection ({i+1}/{n_scenes})...")

    if progress_callback:
        progress_callback(15, 100, "ROI detection complete.")

    return Pass1Result(
        scene_boundaries   = scene_boundaries,
        per_scene_rois     = per_scene_rois,
        per_scene_anchors  = per_scene_anchors,
        per_scene_persons  = per_scene_persons,
        per_scene_previews = per_scene_previews,
        fps                = fps,
        total_frames       = total_frames,
        video_duration_ms  = video_duration_ms,
        video_type         = video_type,
        yolo_tracker       = yolo_tracker,
        flow_estimator     = flow_estimator,
    )


def pass2_extract(video_path: str, r: Pass1Result, cfg: UserConfig,
                  progress_callback=None, frame_callback=None) -> bool:
    """Pass 2: 모션 추출 + funscript 생성. user_config 반영.

    r   : pass1_analyze() 결과
    cfg : UserConfig (유저 설정, None이면 자동)
    """
    if cfg is None:
        cfg = UserConfig.auto_from_pass1(r)

    scene_boundaries  = r.scene_boundaries
    per_scene_rois    = r.per_scene_rois
    per_scene_anchors = r.per_scene_anchors
    per_scene_persons = r.per_scene_persons
    yolo_tracker      = r.yolo_tracker
    flow_estimator    = r.flow_estimator
    fps               = r.fps
    total_frames      = r.total_frames
    video_type        = r.video_type
    roi_strategy      = "per_scene"

    # ── F3: 비활성 씬 프레임 집합 ──
    disabled_frame_set = set()
    for si, sc in enumerate(cfg.scene_configs):
        if not sc.enabled and si < len(scene_boundaries):
            s, e = scene_boundaries[si]
            disabled_frame_set.update(range(s, e))

    # SceneTypeDetector — yolo_tracker 유무와 무관하게 초기화
    scene_type_detector = SceneTypeDetector(fps=fps)

    # ── Phase 3: Sequential motion extraction with dynamic ROI tracking ──
    motion_extractor = MotionExtractor(flow_estimator)
    roi_tracker = DynamicROITracker(fps)

    initial_roi = _get_roi_for_frame(0, scene_boundaries, per_scene_rois)
    roi_tracker.reset(initial_roi)

    scene_change_set = {s for s, _ in scene_boundaries if s > 0}

    boundary_frame_set = set()
    for s_frame, _ in scene_boundaries:
        if s_frame > 30:
            for fi in range(max(0, s_frame - 5), min(total_frames, s_frame + 30)):
                boundary_frame_set.add(fi)

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
    yolo_hip_ys    = []
    yolo_ref_lens  = []
    yolo_confs     = []
    yolo_rel_dists = []
    yolo_is_duals  = []
    quality_records = []
    contact_ys     = []
    scene_types    = []
    head_hip_dists = []
    anchor_dists   = []
    contact_tracker    = ContactPointTracker()
    anchor_tracker     = DualAnchorTracker()
    anchor_initialized = False

    _empty_yolo = {'hip_center_y': None, 'reference_length': None,
                   'confidence': 0.0, 'bbox': None, 'keypoints': None,
                   'secondary_hip_y': None, 'secondary_bbox': None,
                   'secondary_keypoints': None,
                   'rel_dist': None, 'is_dual': False}

    # ── F1: 첫 번째 씬(Frame 0)에 대한 추적기 초기화 및 힌트 적용 ──
    if yolo_tracker is not None:
        yolo_tracker.reset_tracking()
        si = _get_scene_index(0, scene_boundaries)
        if si < len(cfg.scene_configs) and si < len(per_scene_persons):
            sc = cfg.scene_configs[si]
            persons = per_scene_persons[si]
            p1_bbox = persons[sc.p1_person_idx]['bbox'] if sc.p1_person_idx < len(persons) else None
            p2_bbox = (persons[sc.p2_person_idx]['bbox']
                       if sc.p2_person_idx >= 0 and sc.p2_person_idx < len(persons) else None)
            if p1_bbox is not None:
                yolo_tracker.set_slot_hint(p1_bbox, p2_bbox)

    for i in range(total_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(
            frame,
            (RESIZE_WIDTH, int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1])))
        )
        frame_h, frame_w = frame_resized.shape[:2]

        # ── F3: 비활성 씬 스킵 ──
        if i in disabled_frame_set:
            velocity_signal.append(0.0)
            magnitude_signal.append(0.0)
            yolo_hip_ys.append(None)
            yolo_ref_lens.append(None)
            yolo_confs.append(0.0)
            yolo_rel_dists.append(None)
            yolo_is_duals.append(False)
            quality_records.append({'conf': 0.0, 'is_dual': False, 'near_boundary': False})
            contact_ys.append(None)
            scene_types.append('QUIET')
            head_hip_dists.append(None)
            anchor_dists.append(None)
            prev_frame = frame_resized
            continue

        if i in scene_change_set:
            new_scene_roi = _get_roi_for_frame(i, scene_boundaries, per_scene_rois)
            roi_tracker.reset(new_scene_roi)
            if yolo_tracker is not None:
                yolo_tracker.reset_tracking()
                # ── F1: 인물 선택 힌트 적용 ──
                si = _get_scene_index(i, scene_boundaries)
                if si < len(cfg.scene_configs) and si < len(per_scene_persons):
                    sc = cfg.scene_configs[si]
                    persons = per_scene_persons[si]
                    p1_bbox = persons[sc.p1_person_idx]['bbox'] if sc.p1_person_idx < len(persons) else None
                    p2_bbox = (persons[sc.p2_person_idx]['bbox']
                               if sc.p2_person_idx >= 0 and sc.p2_person_idx < len(persons) else None)
                    if p1_bbox is not None:
                        yolo_tracker.set_slot_hint(p1_bbox, p2_bbox)
            scene_type_detector.reset()
            contact_tracker.reset()
            anchor_initialized = False
            # ── F2: 수동 골반 앵커 ──
            si = _get_scene_index(i, scene_boundaries)
            if si < len(cfg.scene_configs):
                sc2 = cfg.scene_configs[si]
                if sc2.p1_hip_px is not None:
                    frame_gray_tmp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    anchor_tracker.reset(sc2.p1_hip_px, sc2.p2_hip_px,
                                         frame_gray_tmp, frame_h, frame_w)
                    anchor_initialized = True

        flow = flow_estimator.estimate_flow(prev_frame, frame_resized)
        current_roi = roi_tracker.update(flow, frame_h, frame_w)

        velocity, magnitude, _, zoom_detected = motion_extractor.extract_velocity_signal(
            prev_frame, frame_resized, current_roi, precomputed_flow=flow
        )

        velocity_signal.append(velocity)
        magnitude_signal.append(magnitude)

        yolo_result = dict(_empty_yolo)
        if yolo_tracker is not None:
            yolo_result = yolo_tracker.process_frame(frame_resized,
                                                      roi_fractions=current_roi)

        scene_type = scene_type_detector.update(
            p1_kp=yolo_result.get('keypoints'),
            p2_kp=yolo_result.get('secondary_keypoints'),
            p2_hip_y=yolo_result.get('secondary_hip_y'),
            frame_h=frame_h,
        )

        yolo_hip_ys.append(yolo_result['hip_center_y'])
        yolo_ref_lens.append(yolo_result['reference_length'])
        yolo_confs.append(yolo_result['confidence'])
        yolo_rel_dists.append(yolo_result.get('rel_dist'))
        yolo_is_duals.append(yolo_result.get('is_dual', False))
        scene_types.append(scene_type)
        head_hip_dists.append(_compute_bj_dist(yolo_result, frame_h))
        quality_records.append({
            'conf'         : yolo_result.get('confidence', 0.0),
            'is_dual'      : yolo_result.get('is_dual', False),
            'near_boundary': i in boundary_frame_set,
        })

        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        if yolo_result.get('is_dual'):
            cpx, cpy = _compute_contact_pixel(yolo_result, frame_h, frame_w)
        else:
            cpx, cpy = None, None
        contact_y = contact_tracker.update(frame_gray, cpx, cpy, frame_h, frame_w)
        contact_ys.append(contact_y)

        if not anchor_initialized:
            si = _get_scene_index(i, scene_boundaries)
            # F2 수동 앵커 확인
            if si < len(cfg.scene_configs) and cfg.scene_configs[si].p1_hip_px is not None:
                sc2 = cfg.scene_configs[si]
                anchor_tracker.reset(sc2.p1_hip_px, sc2.p2_hip_px,
                                     frame_gray, frame_h, frame_w)
                anchor_initialized = True
            else:
                anchor_info = per_scene_anchors[si] if si < len(per_scene_anchors) else None
                if (anchor_info and anchor_info['is_dual']
                        and anchor_info['frame_h'] > 0 and anchor_info['frame_w'] > 0):
                    scale_y = frame_h / anchor_info['frame_h']
                    scale_x = frame_w / anchor_info['frame_w']
                    p1_scaled = (anchor_info['p1_hip_px'][0] * scale_x,
                                 anchor_info['p1_hip_px'][1] * scale_y)
                    p2_scaled = (anchor_info['p2_hip_px'][0] * scale_x,
                                 anchor_info['p2_hip_px'][1] * scale_y)
                    anchor_tracker.reset(p1_scaled, p2_scaled, frame_gray, frame_h, frame_w)
                    anchor_initialized = True

        anchor_dist = None
        if anchor_initialized:
            anchor_dist = anchor_tracker.update(frame_gray, yolo_result, frame_h, frame_w)
            if anchor_dist is None:
                anchor_initialized = False
        anchor_dists.append(anchor_dist)

        # ── F2 앵커 hip_y → YOLO 낮은 신뢰도 프레임 보완 ──────────────────────
        # F2 수동 hip 지정 씬에서 YOLO가 P1을 잃으면, LK 앵커로 추적한 P1 y좌표를
        # yolo_hip_ys에 주입하여 스트로크 신호 연속성을 유지한다.
        si_cur = _get_scene_index(i, scene_boundaries)
        f2_set = (si_cur < len(cfg.scene_configs)
                  and cfg.scene_configs[si_cur].p1_hip_px is not None)
        if f2_set and anchor_initialized:
            cur_conf = yolo_confs[-1] if yolo_confs else 0.0
            if cur_conf < 0.45:
                p1_y_anchor = anchor_tracker.get_p1_y_norm()
                if p1_y_anchor is not None:
                    yolo_hip_ys[-1] = p1_y_anchor
                    yolo_confs[-1]  = 0.55   # anchor 신뢰도 (중간값)
                    if yolo_ref_lens[-1] is None:
                        yolo_ref_lens[-1] = 0.40  # 전형적 body height / frame_h

        prev_frame = frame_resized

        if i % 5 == 0:
            pct = 15 + int((i / total_frames) * 55)
            if progress_callback:
                progress_callback(pct, 100, f"Extracting motion... ({i+1}/{total_frames})")
            if frame_callback:
                _anc1, _anc2 = anchor_tracker.get_anchor_pixels() if anchor_initialized else (None, None)
                b64 = _draw_debug_overlay(
                    frame_resized, current_roi, velocity, magnitude,
                    zoom_detected, i + 1, fps,
                    bbox=yolo_result['bbox'],
                    keypoints=yolo_result['keypoints'],
                    secondary_hip_y=yolo_result.get('secondary_hip_y'),
                    secondary_bbox=yolo_result.get('secondary_bbox'),
                    rel_dist=yolo_result.get('rel_dist'),
                    secondary_keypoints=yolo_result.get('secondary_keypoints'),
                    scene_type=scene_type,
                    anchor_p1=_anc1,
                    anchor_p2=_anc2,
                    anchor_dist=anchor_dist,
                )
                frame_callback(b64, {
                    'type': 'frame',
                    'frame': i + 1,
                    'total': total_frames,
                    'velocity': round(velocity, 3),
                    'magnitude': round(magnitude, 3),
                    'zoom': zoom_detected,
                })

    cap.release()

    if yolo_tracker is not None:
        yolo_tracker.close()

    velocity_signal = np.array(velocity_signal, dtype=np.float64)
    magnitude_signal = np.array(magnitude_signal, dtype=np.float64)

    if len(velocity_signal) < 10:
        print("Error: Not enough frames processed")
        return False

    has_yolo = yolo_tracker is not None and len(yolo_hip_ys) == len(velocity_signal)

    dual_coverage = (sum(yolo_is_duals) / max(len(yolo_is_duals), 1)
                     if yolo_is_duals else 0.0)
    print(f"Dual mode coverage: {dual_coverage:.1%}")

    contact_coverage = (sum(1 for y in contact_ys if y is not None) / max(len(contact_ys), 1)
                        if contact_ys else 0.0)
    print(f"LK contact tracking coverage: {contact_coverage:.1%}")

    st_arr  = np.array(scene_types) if scene_types else np.array([])
    bj_mask = (st_arr == 'BJ')
    bj_coverage = float(bj_mask.sum()) / max(len(st_arr), 1)
    print(f"Scene coverage — BJ: {bj_coverage:.1%}")

    anchor_coverage = (sum(1 for d in anchor_dists if d is not None) / max(len(anchor_dists), 1)
                       if anchor_dists else 0.0)
    print(f"Dual anchor tracking coverage: {anchor_coverage:.1%}")

    if (contact_coverage >= 0.40 and len(contact_ys) == len(velocity_signal)
            and dual_coverage >= 0.30):
        velocity_signal = _blend_contact_tracking(velocity_signal, contact_ys, yolo_is_duals)
        print("Using LK contact point tracking signal")
    elif (anchor_coverage >= 0.30 and len(anchor_dists) == len(velocity_signal)):
        velocity_signal = _blend_anchor_dist(velocity_signal, anchor_dists)
        print(f"Using dual-anchor LK distance signal ({anchor_coverage:.1%} coverage)")
    elif (bj_coverage >= 0.25 and len(head_hip_dists) == len(velocity_signal)
          and len(st_arr) == len(velocity_signal)):
        velocity_signal = _blend_bj_pose(velocity_signal, head_hip_dists, bj_mask)
        print(f"Using BJ head-hip distance signal ({bj_coverage:.1%} coverage)")
    elif dual_coverage >= 0.50 and len(yolo_rel_dists) == len(velocity_signal):
        velocity_signal = _blend_dual_pose(velocity_signal, yolo_rel_dists, yolo_is_duals)
        print("Using dual-person relative distance signal")
    elif has_yolo:
        velocity_signal = _blend_yolo_pose(
            velocity_signal, yolo_hip_ys, yolo_ref_lens, yolo_confs
        )
        yolo_confident = sum(1 for c in yolo_confs if c and c > 0.5)
        print(f"Pose blend (YOLO-only): {yolo_confident} high-conf frames")

    if quality_records and len(quality_records) == len(velocity_signal):
        quality_mask = _build_quality_mask(quality_records)
        bad_ratio = float(quality_mask.sum()) / max(len(quality_mask), 1)
        print(f"Quality mask: {bad_ratio:.1%} bad frames suppressed")
        velocity_signal = _postprocess_velocity(velocity_signal, quality_mask)

    suppress_frames = 30 if dual_coverage >= 0.20 else 0
    boundary_handler = SceneBoundaryHandler()
    velocity_signal = boundary_handler.smooth_at_boundaries(
        velocity_signal, scene_boundaries, fps,
        post_cut_suppress_frames=suppress_frames
    )
    magnitude_signal = boundary_handler.smooth_at_boundaries(
        magnitude_signal, scene_boundaries, fps
    )

    if progress_callback:
        progress_callback(70, 100, "Analyzing scenes...")

    if len(velocity_signal) > 5:
        velocity_smoothed = savgol_filter(velocity_signal, 5, 1)
    else:
        velocity_smoothed = velocity_signal

    scene_segmenter = SceneSegmenter(fps)
    scene_change_frames = [s for s, _ in scene_boundaries]

    yolo_overall_det = (
        sum(1 for c in yolo_confs if c and c > 0.3) / max(len(yolo_confs), 1)
        if has_yolo and yolo_confs else 0.0
    )
    use_yolo_quiet = has_yolo and yolo_overall_det >= 0.20
    print(f"YOLO overall detection rate: {yolo_overall_det:.1%} → QUIET suppression: {'ON' if use_yolo_quiet else 'OFF'}")

    segments = scene_segmenter.classify_segments(
        magnitude_signal, scene_change_frames, len(velocity_signal),
        velocity_signal=velocity_smoothed,
        yolo_confs=yolo_confs if use_yolo_quiet else None,
    )

    active_count = sum(1 for _, _, t in segments if t == 'ACTIVE')
    quiet_count = sum(1 for _, _, t in segments if t == 'QUIET')
    trans_count = sum(1 for _, _, t in segments if t == 'TRANSITION')
    print(f"Segments: {len(segments)} total "
          f"({active_count} active, {trans_count} transition, {quiet_count} quiet)")

    if progress_callback:
        progress_callback(75, 100, "Estimating position signal...")

    position_estimator = PositionEstimator(fps)
    position_raw = position_estimator.velocity_to_position(
        velocity_smoothed, segments, scene_boundaries=scene_boundaries
    )
    position_normalized = position_estimator.normalize_per_segment(position_raw, segments)
    position_normalized = position_estimator.expand_contrast(position_normalized, segments=segments)

    if progress_callback:
        progress_callback(85, 100, "Generating action points...")

    action_generator = ActionPointGenerator(fps)
    actions = action_generator.generate(position_normalized, segments, velocity_signal=velocity_smoothed)

    if frame_callback:
        graph_b64 = _draw_result_graph(
            position_normalized, actions, segments, fps, len(velocity_signal)
        )
        frame_callback(graph_b64, {
            'type': 'result',
            'actions': len(actions),
            'active_segs': active_count,
            'quiet_segs': quiet_count,
        })

    if progress_callback:
        progress_callback(90, 100, "Post-processing...")

    actions = _normalize_stroke_amplitude(actions, segments, fps, min_range=45)

    post_processor = ScriptPostProcessor()
    actions = post_processor.validate_and_fix(actions, max_speed=500)

    video_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(video_dir, base_name + ".funscript")

    roi_info = []
    for i, (roi, (s, e)) in enumerate(zip(per_scene_rois, scene_boundaries)):
        roi_info.append({
            "scene": i + 1,
            "frames": [int(s), int(e)],
            "y_range": [round(float(roi[0]), 3), round(float(roi[1]), 3)],
            "x_range": [round(float(roi[2]), 3), round(float(roi[3]), 3)]
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


def run_analysis(video_path, progress_callback=None, frame_callback=None):
    """CLI/기존 호환 자동 모드 래퍼 — pass1_analyze + pass2_extract 순차 실행."""
    r = pass1_analyze(video_path, progress_callback)
    if r is None:
        return False
    cfg = UserConfig.auto_from_pass1(r)
    return pass2_extract(video_path, r, cfg, progress_callback, frame_callback)


def _run_analysis_legacy(video_path, progress_callback=None, frame_callback=None):
    """레거시 단일 함수 버전 (참조용 보존, 실제 호출 안됨)."""
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

    # ── YoloPoseTracker 초기화 (ROI 탐지에도 사용하므로 먼저 실행) ──
    yolo_tracker = None
    if YOLO_AVAILABLE:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'models', 'yolo26x-pose.pt')
        try:
            yolo_tracker = YoloPoseTracker(model_path)
            print(f"YoloPoseTracker enabled (device: {DEVICE})")
        except Exception as e:
            yolo_tracker = None
            print(f"YoloPoseTracker init failed (skipping): {e}")

    # SceneTypeDetector — yolo_tracker 유무와 무관하게 초기화
    scene_detector = SceneTypeDetector(fps=fps)

    # ── Phase 2: ROI Detection (strategy depends on video type) ──
    if progress_callback:
        progress_callback(5, 100, "Detecting ROI...")

    flow_estimator = OpticalFlowEstimator()
    roi_detector = ROIDetector(flow_estimator)

    # Always use per-scene ROI regardless of video_type
    # (global ROI across many scenes causes full-frame [0-1,0-1] detection failure)
    per_scene_rois    = []
    per_scene_anchors = []
    anchor_selector   = SceneAnchorSelector() if yolo_tracker is not None else None

    for i, (scene_start, scene_end) in enumerate(scene_boundaries):
        scene_length = scene_end - scene_start
        scene_sample_count = min(40, max(10, scene_length // 5))
        roi = roi_detector.detect_roi(video_path, sample_count=scene_sample_count,
                                       frame_range=(scene_start, scene_end),
                                       yolo_tracker=yolo_tracker)
        per_scene_rois.append(roi)
        print(f"  Scene {i+1} [{scene_start}-{scene_end}]: "
              f"ROI y=[{roi[0]:.2f}-{roi[1]:.2f}], x=[{roi[2]:.2f}-{roi[3]:.2f}]")

        # 씬별 dual 앵커 선정
        if anchor_selector is not None:
            anchor = anchor_selector.select(
                video_path, scene_start, scene_end, yolo_tracker, n_samples=10
            )
            per_scene_anchors.append(anchor)
            print(f"  Scene {i+1} anchor: dual={anchor['is_dual']}, "
                  f"conf={anchor['confidence']:.2f}")
        else:
            per_scene_anchors.append(None)

    roi_strategy = "per_scene"

    if progress_callback:
        progress_callback(15, 100, "ROI detection complete.")

    # ══════════════════════════════════════════════
    # PASS 2: Full Motion Extraction + Processing
    # ══════════════════════════════════════════════

    # ── Phase 3: Sequential motion extraction with dynamic ROI tracking ──
    motion_extractor = MotionExtractor(flow_estimator)
    roi_tracker = DynamicROITracker(fps)

    initial_roi = _get_roi_for_frame(0, scene_boundaries, per_scene_rois)
    roi_tracker.reset(initial_roi)

    scene_change_set = {s for s, _ in scene_boundaries if s > 0}

    # 씬 경계 ±30프레임 → 저품질 마킹 집합 (quality_records용)
    boundary_frame_set = set()
    for s_frame, _ in scene_boundaries:
        if s_frame > 30:   # 처음 30프레임(DUAL_WARMUP)은 억제 제외
            for fi in range(max(0, s_frame - 5), min(total_frames, s_frame + 30)):
                boundary_frame_set.add(fi)

    # MediaPipe 완전 제거 — YOLO만 사용

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
    yolo_hip_ys    = []
    yolo_ref_lens  = []
    yolo_confs     = []
    yolo_rel_dists = []   # dual: 두 골반 상대 거리
    yolo_is_duals  = []   # dual: 해당 프레임 dual 모드 여부
    quality_records = []  # 프레임별 품질 메타데이터
    contact_ys     = []   # LK 접촉점 y (정규화 0~1, None=추적실패)
    scene_types    = []   # 프레임별 SceneTypeDetector 결과
    head_hip_dists = []   # 프레임별 P1 head ↔ P2 hip 정규화 거리 (or None)
    anchor_dists   = []   # 프레임별 DualAnchorTracker 거리 (float|None)
    contact_tracker    = ContactPointTracker()
    anchor_tracker     = DualAnchorTracker()
    anchor_initialized = False

    for i in range(total_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(
            frame,
            (RESIZE_WIDTH, int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1])))
        )
        frame_h, frame_w = frame_resized.shape[:2]

        if i in scene_change_set:
            new_scene_roi = _get_roi_for_frame(i, scene_boundaries, per_scene_rois)
            roi_tracker.reset(new_scene_roi)
            if yolo_tracker is not None:
                yolo_tracker.reset_tracking()
            scene_detector.reset()
            contact_tracker.reset()
            anchor_initialized = False

        flow = flow_estimator.estimate_flow(prev_frame, frame_resized)
        current_roi = roi_tracker.update(flow, frame_h, frame_w)

        velocity, magnitude, _, zoom_detected = motion_extractor.extract_velocity_signal(
            prev_frame, frame_resized, current_roi, precomputed_flow=flow
        )

        velocity_signal.append(velocity)
        magnitude_signal.append(magnitude)

        # ── YOLO Pose (Phase 4 - primary) ──
        yolo_result = {'hip_center_y': None, 'reference_length': None,
                       'confidence': 0.0, 'bbox': None, 'keypoints': None,
                       'secondary_hip_y': None, 'secondary_bbox': None,
                       'secondary_keypoints': None,
                       'rel_dist': None, 'is_dual': False}
        if yolo_tracker is not None:
            yolo_result = yolo_tracker.process_frame(frame_resized,
                                                      roi_fractions=current_roi)

        # ── 장면 유형 감지 ──
        scene_type = scene_detector.update(
            p1_kp=yolo_result.get('keypoints'),
            p2_kp=yolo_result.get('secondary_keypoints'),
            p2_hip_y=yolo_result.get('secondary_hip_y'),
            frame_h=frame_h,
        )

        yolo_hip_ys.append(yolo_result['hip_center_y'])
        yolo_ref_lens.append(yolo_result['reference_length'])
        yolo_confs.append(yolo_result['confidence'])
        yolo_rel_dists.append(yolo_result.get('rel_dist'))
        yolo_is_duals.append(yolo_result.get('is_dual', False))
        scene_types.append(scene_type)
        head_hip_dists.append(_compute_bj_dist(yolo_result, frame_h))
        quality_records.append({
            'conf'         : yolo_result.get('confidence', 0.0),
            'is_dual'      : yolo_result.get('is_dual', False),
            'near_boundary': i in boundary_frame_set,
        })

        # ── ContactPointTracker (LK 접촉점 추적) ──
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        if yolo_result.get('is_dual'):
            cpx, cpy = _compute_contact_pixel(yolo_result, frame_h, frame_w)
        else:
            cpx, cpy = None, None
        contact_y = contact_tracker.update(frame_gray, cpx, cpy, frame_h, frame_w)
        contact_ys.append(contact_y)

        # ── DualAnchorTracker (LK 두 점 독립 추적) ──
        if not anchor_initialized:
            si = _get_scene_index(i, scene_boundaries)
            anchor_info = per_scene_anchors[si] if si < len(per_scene_anchors) else None
            if (anchor_info and anchor_info['is_dual']
                    and anchor_info['frame_h'] > 0 and anchor_info['frame_w'] > 0):
                scale_y = frame_h / anchor_info['frame_h']
                scale_x = frame_w / anchor_info['frame_w']
                p1_scaled = (anchor_info['p1_hip_px'][0] * scale_x,
                             anchor_info['p1_hip_px'][1] * scale_y)
                p2_scaled = (anchor_info['p2_hip_px'][0] * scale_x,
                             anchor_info['p2_hip_px'][1] * scale_y)
                anchor_tracker.reset(p1_scaled, p2_scaled, frame_gray, frame_h, frame_w)
                anchor_initialized = True

        anchor_dist = None
        if anchor_initialized:
            anchor_dist = anchor_tracker.update(frame_gray, yolo_result, frame_h, frame_w)
            if anchor_dist is None:
                anchor_initialized = False
        anchor_dists.append(anchor_dist)

        prev_frame = frame_resized

        if i % 5 == 0:
            pct = 15 + int((i / total_frames) * 55)   # 15-70%
            if progress_callback:
                progress_callback(pct, 100, f"Extracting motion... ({i+1}/{total_frames})")
            if frame_callback:
                _anc1, _anc2 = anchor_tracker.get_anchor_pixels() if anchor_initialized else (None, None)
                b64 = _draw_debug_overlay(
                    frame_resized, current_roi, velocity, magnitude,
                    zoom_detected, i + 1, fps,
                    bbox=yolo_result['bbox'],
                    keypoints=yolo_result['keypoints'],
                    secondary_hip_y=yolo_result.get('secondary_hip_y'),
                    secondary_bbox=yolo_result.get('secondary_bbox'),
                    rel_dist=yolo_result.get('rel_dist'),
                    secondary_keypoints=yolo_result.get('secondary_keypoints'),
                    scene_type=scene_type,
                    anchor_p1=_anc1,
                    anchor_p2=_anc2,
                    anchor_dist=anchor_dist,
                )
                frame_callback(b64, {
                    'type': 'frame',
                    'frame': i + 1,
                    'total': total_frames,
                    'velocity': round(velocity, 3),
                    'magnitude': round(magnitude, 3),
                    'zoom': zoom_detected,
                })

    cap.release()

    if yolo_tracker is not None:
        yolo_tracker.close()

    velocity_signal = np.array(velocity_signal, dtype=np.float64)
    magnitude_signal = np.array(magnitude_signal, dtype=np.float64)

    if len(velocity_signal) < 10:
        print("Error: Not enough frames processed")
        return False

    # ── Phase 4: YOLO Pose-flow blending ──
    has_yolo = yolo_tracker is not None and len(yolo_hip_ys) == len(velocity_signal)

    # dual 모드 우선: 두 골반 상대 거리 신호
    dual_coverage = (sum(yolo_is_duals) / max(len(yolo_is_duals), 1)
                     if yolo_is_duals else 0.0)
    print(f"Dual mode coverage: {dual_coverage:.1%}")

    # LK 접촉점 우선 (dual 영상에서 40%+ 추적 성공 시)
    contact_coverage = (sum(1 for y in contact_ys if y is not None) / max(len(contact_ys), 1)
                        if contact_ys else 0.0)
    print(f"LK contact tracking coverage: {contact_coverage:.1%}")

    # scene_type 배열 → BJ 마스크 생성
    st_arr  = np.array(scene_types) if scene_types else np.array([])
    bj_mask = (st_arr == 'BJ')
    bj_coverage = float(bj_mask.sum()) / max(len(st_arr), 1)
    print(f"Scene coverage — BJ: {bj_coverage:.1%}")

    anchor_coverage = (sum(1 for d in anchor_dists if d is not None) / max(len(anchor_dists), 1)
                       if anchor_dists else 0.0)
    print(f"Dual anchor tracking coverage: {anchor_coverage:.1%}")

    if (contact_coverage >= 0.40 and len(contact_ys) == len(velocity_signal)
            and dual_coverage >= 0.30):
        velocity_signal = _blend_contact_tracking(velocity_signal, contact_ys, yolo_is_duals)
        print("Using LK contact point tracking signal")
    elif (anchor_coverage >= 0.30 and len(anchor_dists) == len(velocity_signal)):
        velocity_signal = _blend_anchor_dist(velocity_signal, anchor_dists)
        print(f"Using dual-anchor LK distance signal ({anchor_coverage:.1%} coverage)")
    elif (bj_coverage >= 0.25 and len(head_hip_dists) == len(velocity_signal)
          and len(st_arr) == len(velocity_signal)):
        velocity_signal = _blend_bj_pose(velocity_signal, head_hip_dists, bj_mask)
        print(f"Using BJ head-hip distance signal ({bj_coverage:.1%} coverage)")
    elif dual_coverage >= 0.50 and len(yolo_rel_dists) == len(velocity_signal):
        velocity_signal = _blend_dual_pose(velocity_signal, yolo_rel_dists, yolo_is_duals)
        print("Using dual-person relative distance signal")
    elif has_yolo:
        velocity_signal = _blend_yolo_pose(
            velocity_signal, yolo_hip_ys, yolo_ref_lens, yolo_confs
        )
        yolo_confident = sum(1 for c in yolo_confs if c and c > 0.5)
        print(f"Pose blend (YOLO-only): {yolo_confident} high-conf frames")

    # ── 품질 메타데이터 기반 velocity 후처리 ──
    if quality_records and len(quality_records) == len(velocity_signal):
        quality_mask = _build_quality_mask(quality_records)
        bad_ratio = float(quality_mask.sum()) / max(len(quality_mask), 1)
        print(f"Quality mask: {bad_ratio:.1%} bad frames suppressed")
        velocity_signal = _postprocess_velocity(velocity_signal, quality_mask)

    # ── Scene Boundary Smoothing: remove cross-scene flow spikes ──
    suppress_frames = 30 if dual_coverage >= 0.20 else 0
    boundary_handler = SceneBoundaryHandler()
    velocity_signal = boundary_handler.smooth_at_boundaries(
        velocity_signal, scene_boundaries, fps,
        post_cut_suppress_frames=suppress_frames
    )
    magnitude_signal = boundary_handler.smooth_at_boundaries(
        magnitude_signal, scene_boundaries, fps
    )

    # ── Phase 4: Scene Segmentation with per-scene thresholds ──
    if progress_callback:
        progress_callback(70, 100, "Analyzing scenes...")

    # Minimal smoothing on velocity before segmentation and integration
    if len(velocity_signal) > 5:
        velocity_smoothed = savgol_filter(velocity_signal, 5, 1)
    else:
        velocity_smoothed = velocity_signal

    scene_segmenter = SceneSegmenter(fps)

    # Use scene boundaries as change points for segmentation
    scene_change_frames = [s for s, _ in scene_boundaries]

    # YOLO conf 기반 QUIET 억제: 전체 탐지율 > 20% 일 때만 활성화
    # (애니메이션/YOLO 미탐지 영상에서 전체를 QUIET으로 만드는 오작동 방지)
    yolo_overall_det = (
        sum(1 for c in yolo_confs if c and c > 0.3) / max(len(yolo_confs), 1)
        if has_yolo and yolo_confs else 0.0
    )
    use_yolo_quiet = has_yolo and yolo_overall_det >= 0.20
    print(f"YOLO overall detection rate: {yolo_overall_det:.1%} → QUIET suppression: {'ON' if use_yolo_quiet else 'OFF'}")

    segments = scene_segmenter.classify_segments(
        magnitude_signal, scene_change_frames, len(velocity_signal),
        velocity_signal=velocity_smoothed,
        yolo_confs=yolo_confs if use_yolo_quiet else None,
    )

    active_count = sum(1 for _, _, t in segments if t == 'ACTIVE')
    quiet_count = sum(1 for _, _, t in segments if t == 'QUIET')
    trans_count = sum(1 for _, _, t in segments if t == 'TRANSITION')
    print(f"Segments: {len(segments)} total "
          f"({active_count} active, {trans_count} transition, {quiet_count} quiet)")

    # ── Phase 5: Position Estimation ──
    if progress_callback:
        progress_callback(75, 100, "Estimating position signal...")

    position_estimator = PositionEstimator(fps)

    # G1 (직접 위치 경로): 비활성화
    # 이유: hip_y 절대 좌표는 funscript 위치와 범용적 1:1 대응 불가.
    # 스크립터마다 다른 기준점 → Librarian composite 0.7660→0.4816 catastrophic regression 확인.
    # 기존 경로: velocity → cumsum → HPF
    position_raw = position_estimator.velocity_to_position(
        velocity_smoothed, segments, scene_boundaries=scene_boundaries
    )
    position_normalized = position_estimator.normalize_per_segment(position_raw, segments)
    position_normalized = position_estimator.expand_contrast(position_normalized, segments=segments)

    # ── Phase 6: Action Point Generation ──
    if progress_callback:
        progress_callback(85, 100, "Generating action points...")

    action_generator = ActionPointGenerator(fps)
    actions = action_generator.generate(position_normalized, segments, velocity_signal=velocity_smoothed)

    # ── Result graph (before post-processing — position signal + raw action points) ──
    if frame_callback:
        graph_b64 = _draw_result_graph(
            position_normalized, actions, segments, fps, len(velocity_signal)
        )
        frame_callback(graph_b64, {
            'type': 'result',
            'actions': len(actions),
            'active_segs': active_count,
            'quiet_segs': quiet_count,
        })

    # ── Phase 7: Post-processing ──
    if progress_callback:
        progress_callback(90, 100, "Post-processing...")

    # Stroke 진폭 자동 스트레칭 (씬별 range < 45 → 보정)
    actions = _normalize_stroke_amplitude(actions, segments, fps, min_range=45)

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
            "frames": [int(s), int(e)],
            "y_range": [round(float(roi[0]), 3), round(float(roi[1]), 3)],
            "x_range": [round(float(roi[2]), 3), round(float(roi[3]), 3)]
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
