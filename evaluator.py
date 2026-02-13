"""
Funscript quality evaluator.
Compares program-generated funscripts against human-created originals.
"""

import os
import json
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sampleVideo")
ORIGINAL_DIR = os.path.join(SAMPLE_DIR, "funscript_original")
PROGRAM_DIR = os.path.join(SAMPLE_DIR, "program")


def load_funscript(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    actions = data.get("actions", [])
    actions.sort(key=lambda a: a["at"])
    return actions


def actions_to_timeseries(actions, sample_rate_ms=50):
    """Resample actions to uniform time series for DTW comparison."""
    if not actions:
        return np.array([]), 0, 0

    t_start = actions[0]["at"]
    t_end = actions[-1]["at"]

    if t_end <= t_start:
        return np.array([actions[0]["pos"]]), t_start, t_end

    times = np.arange(t_start, t_end + 1, sample_rate_ms)
    positions = np.interp(
        times,
        [a["at"] for a in actions],
        [a["pos"] for a in actions]
    )
    return positions, t_start, t_end


def compute_dtw_similarity(orig_actions, prog_actions):
    """Compute DTW distance normalized to 0-1 similarity score."""
    orig_ts, orig_start, orig_end = actions_to_timeseries(orig_actions)
    prog_ts, prog_start, prog_end = actions_to_timeseries(prog_actions)

    if len(orig_ts) < 2 or len(prog_ts) < 2:
        return 0.0

    # Align to overlapping time range
    overlap_start = max(orig_start, prog_start)
    overlap_end = min(orig_end, prog_end)

    if overlap_end <= overlap_start:
        return 0.0

    # Re-sample both to the overlapping range
    sample_rate_ms = 50
    times = np.arange(overlap_start, overlap_end + 1, sample_rate_ms)

    orig_interp = np.interp(
        times,
        [a["at"] for a in orig_actions],
        [a["pos"] for a in orig_actions]
    )
    prog_interp = np.interp(
        times,
        [a["at"] for a in prog_actions],
        [a["pos"] for a in prog_actions]
    )

    # Normalize both to 0-1 for fair comparison
    orig_norm = orig_interp / 100.0
    prog_norm = prog_interp / 100.0

    distance, _ = fastdtw(
        orig_norm.reshape(-1, 1),
        prog_norm.reshape(-1, 1),
        dist=euclidean,
        radius=10
    )

    # Normalize distance by length
    max_possible = len(orig_norm)
    normalized_dist = distance / max_possible if max_possible > 0 else 1.0

    # Convert to similarity (0-1, higher is better)
    similarity = max(0.0, 1.0 - normalized_dist)
    return similarity


def compute_action_density_ratio(orig_actions, prog_actions):
    """Ratio of program action count to original (1.0 = perfect match)."""
    if not orig_actions:
        return 0.0
    return len(prog_actions) / len(orig_actions)


def compute_position_distribution_similarity(orig_actions, prog_actions):
    """Compare position distributions using histogram intersection."""
    if not orig_actions or not prog_actions:
        return 0.0

    bins = np.arange(0, 102, 5)  # 0-100 in bins of 5

    orig_pos = [a["pos"] for a in orig_actions]
    prog_pos = [a["pos"] for a in prog_actions]

    orig_hist, _ = np.histogram(orig_pos, bins=bins, density=True)
    prog_hist, _ = np.histogram(prog_pos, bins=bins, density=True)

    # Histogram intersection (0-1, higher is better)
    intersection = np.sum(np.minimum(orig_hist, prog_hist))
    max_possible = np.sum(np.maximum(orig_hist, prog_hist))

    if max_possible < 1e-10:
        return 0.0

    return intersection / max_possible


def compute_coverage_ratio(orig_actions, prog_actions):
    """How well does the program cover the same time span as original."""
    if not orig_actions or not prog_actions:
        return 0.0

    orig_start = orig_actions[0]["at"]
    orig_end = orig_actions[-1]["at"]
    orig_span = orig_end - orig_start

    if orig_span <= 0:
        return 0.0

    prog_start = prog_actions[0]["at"]
    prog_end = prog_actions[-1]["at"]

    # Overlap
    overlap_start = max(orig_start, prog_start)
    overlap_end = min(orig_end, prog_end)
    overlap = max(0, overlap_end - overlap_start)

    return overlap / orig_span


def compute_mean_absolute_error(orig_actions, prog_actions):
    """Mean absolute position error over overlapping time range."""
    orig_ts, orig_start, orig_end = actions_to_timeseries(orig_actions)
    prog_ts, prog_start, prog_end = actions_to_timeseries(prog_actions)

    if len(orig_ts) < 2 or len(prog_ts) < 2:
        return 100.0

    overlap_start = max(orig_start, prog_start)
    overlap_end = min(orig_end, prog_end)

    if overlap_end <= overlap_start:
        return 100.0

    sample_rate_ms = 50
    times = np.arange(overlap_start, overlap_end + 1, sample_rate_ms)

    orig_interp = np.interp(
        times,
        [a["at"] for a in orig_actions],
        [a["pos"] for a in orig_actions]
    )
    prog_interp = np.interp(
        times,
        [a["at"] for a in prog_actions],
        [a["pos"] for a in prog_actions]
    )

    return float(np.mean(np.abs(orig_interp - prog_interp)))


def evaluate_all():
    """Run evaluation on all matching funscript pairs."""
    import sys
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    if not os.path.isdir(ORIGINAL_DIR) or not os.path.isdir(PROGRAM_DIR):
        print("Error: Sample directories not found.")
        return

    original_files = {f for f in os.listdir(ORIGINAL_DIR) if f.endswith(".funscript")}
    program_files = {f for f in os.listdir(PROGRAM_DIR) if f.endswith(".funscript")}
    common = sorted(original_files & program_files)

    if not common:
        print("No matching funscript pairs found.")
        return

    print("=" * 90)
    print("FUNSCRIPT QUALITY EVALUATION")
    print("=" * 90)

    all_scores = []

    for filename in common:
        orig_path = os.path.join(ORIGINAL_DIR, filename)
        prog_path = os.path.join(PROGRAM_DIR, filename)

        orig_actions = load_funscript(orig_path)
        prog_actions = load_funscript(prog_path)

        dtw_sim = compute_dtw_similarity(orig_actions, prog_actions)
        density_ratio = compute_action_density_ratio(orig_actions, prog_actions)
        dist_sim = compute_position_distribution_similarity(orig_actions, prog_actions)
        coverage = compute_coverage_ratio(orig_actions, prog_actions)
        mae = compute_mean_absolute_error(orig_actions, prog_actions)

        # Composite score (weighted)
        composite = (
            dtw_sim * 0.35 +
            min(density_ratio, 1.0 / max(density_ratio, 0.01)) * 0.15 +
            dist_sim * 0.20 +
            coverage * 0.15 +
            max(0, 1.0 - mae / 50.0) * 0.15
        )

        scores = {
            "dtw_similarity": dtw_sim,
            "density_ratio": density_ratio,
            "distribution_similarity": dist_sim,
            "coverage": coverage,
            "mae": mae,
            "composite": composite,
        }
        all_scores.append((filename, scores))

        name = os.path.splitext(filename)[0]
        if len(name) > 40:
            name = name[:37] + "..."

        print(f"\n  {name}")
        print(f"  {'─' * 60}")
        print(f"  Actions       : original={len(orig_actions):>5}  program={len(prog_actions):>5}  ratio={density_ratio:.2f}x")
        print(f"  DTW Similarity: {dtw_sim:.4f}")
        print(f"  Distribution  : {dist_sim:.4f}")
        print(f"  Coverage      : {coverage:.4f}")
        print(f"  MAE (pos)     : {mae:.1f}")
        print(f"  Composite     : {composite:.4f}")

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    avg_composite = np.mean([s["composite"] for _, s in all_scores])
    avg_dtw = np.mean([s["dtw_similarity"] for _, s in all_scores])
    avg_dist = np.mean([s["distribution_similarity"] for _, s in all_scores])
    avg_coverage = np.mean([s["coverage"] for _, s in all_scores])
    avg_mae = np.mean([s["mae"] for _, s in all_scores])

    print(f"  Avg DTW Similarity     : {avg_dtw:.4f}")
    print(f"  Avg Distribution Match : {avg_dist:.4f}")
    print(f"  Avg Coverage           : {avg_coverage:.4f}")
    print(f"  Avg MAE                : {avg_mae:.1f}")
    print(f"  Avg Composite Score    : {avg_composite:.4f}")
    print(f"{'=' * 90}")

    return all_scores


if __name__ == "__main__":
    evaluate_all()
