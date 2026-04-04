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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import datetime

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

def load_funscript_with_meta(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    actions = data.get("actions", [])
    actions.sort(key=lambda a: a["at"])
    return actions, data.get("metadata", {})

def load_funscript(path):
    return load_funscript_with_meta(path)[0]


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


def generate_plot_base64(orig_actions, prog_actions, title):
    orig_ts, orig_start, orig_end = actions_to_timeseries(orig_actions)
    prog_ts, prog_start, prog_end = actions_to_timeseries(prog_actions)
    
    plt.figure(figsize=(12, 4))
    plt.title(f"Position timeline: {title}")
    
    if len(orig_ts) > 0:
        plt.plot(np.arange(orig_start, orig_end + 1, 50) / 1000.0, orig_ts, label='Original (Manual)', alpha=0.7, color='blue')
    if len(prog_ts) > 0:
        plt.plot(np.arange(prog_start, prog_end + 1, 50) / 1000.0, prog_ts, label='Program (AI)', alpha=0.7, color='orange')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Position (0-100)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_json_diagnostic(orig_actions, prog_actions, meta, scores, filepath):
    # Find error spikes (>20 pos diff)
    error_spikes = []
    orig_ts, orig_start, orig_end = actions_to_timeseries(orig_actions)
    prog_ts, prog_start, prog_end = actions_to_timeseries(prog_actions)
    
    overlap_start = max(orig_start, prog_start)
    overlap_end = min(orig_end, prog_end)
    
    if overlap_end > overlap_start:
        times = np.arange(overlap_start, overlap_end + 1, 50)
        orig_interp = np.interp(times, [a["at"] for a in orig_actions], [a["pos"] for a in orig_actions])
        prog_interp = np.interp(times, [a["at"] for a in prog_actions], [a["pos"] for a in prog_actions])
        diffs = np.abs(orig_interp - prog_interp)
        for i, d in enumerate(diffs):
            if d > 20: 
                error_spikes.append({"time_ms": int(times[i]), "diff": float(d), "orig": float(orig_interp[i]), "prog": float(prog_interp[i])})
                
    diag = {
        "timestamp": datetime.datetime.now().isoformat(),
        "scores": scores,
        "raw_metadata": meta,
        "error_spikes": error_spikes,
        "tracking_loss": meta.get("tracking_quality", {}).get("tracking_loss_ratio", 0.0),
        "dual_presence": meta.get("tracking_quality", {}).get("dual_presence_ratio", 0.0),
        "anchor_reliability": meta.get("tracking_quality", {}).get("anchor_reliability", 0.0),
        "analyzer": "AI Diagnostic Module v2"
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(diag, f, indent=2)

def generate_html_report(filename, scores, meta, plot_b64, filepath):
    tq = meta.get('tracking_quality', {})
    p1_b64 = tq.get('p1_snapshot_b64')
    p2_b64 = tq.get('p2_snapshot_b64')
    
    avatars_html = ""
    if p1_b64 or p2_b64:
        avatars_html += '<div style="display: flex; gap: 20px; margin-top: 20px; margin-bottom: 20px;">'
        if p1_b64:
            avatars_html += f'<div style="text-align: center; background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><img src="data:image/jpeg;base64,{p1_b64}" style="height: 120px; border-radius: 4px; border: 2px solid #e74c3c; object-fit: contain;"><div style="font-weight: bold; margin-top: 5px; color: #e74c3c;">Primary (P1)</div></div>'
        if p2_b64:
            avatars_html += f'<div style="text-align: center; background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><img src="data:image/jpeg;base64,{p2_b64}" style="height: 120px; border-radius: 4px; border: 2px solid #3498db; object-fit: contain;"><div style="font-weight: bold; margin-top: 5px; color: #3498db;">Secondary (P2)</div></div>'
        avatars_html += '</div>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Report - {filename}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f7f9fc; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
            .card {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; color: #333; }}
            .score-high {{ color: #27ae60; font-weight: bold; }}
            .score-med {{ color: #f39c12; font-weight: bold; }}
            .score-low {{ color: #c0392b; font-weight: bold; }}
            img.plot {{ width: 100%; height: auto; margin-top: 20px; border: 1px solid #eee; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Funscript Quality Report: {filename}</h1>
            
            {avatars_html}
            
            <div class="grid">
                <div class="card">
                    <h2>Tracking Quality (YOLO)</h2>
                    <table>
                        <tr><th>P1 Avg Confidence</th><td>{meta.get('tracking_quality', {}).get('yolo_p1_confidence_avg', 0.0):.3f}</td></tr>
                        <tr><th>Tracking Loss Ratio</th><td>{meta.get('tracking_quality', {}).get('tracking_loss_ratio', 0.0):.1%}</td></tr>
                        <tr><th>Dual Presence (P1&P2)</th><td>{meta.get('tracking_quality', {}).get('dual_presence_ratio', 0.0):.1%}</td></tr>
                        <tr><th>Anchor Reliability (LK)</th><td>{meta.get('tracking_quality', {}).get('anchor_reliability', 0.0):.1%}</td></tr>
                        <tr><th>Manual Anchor Stability (NCC)</th><td class="{'score-high' if meta.get('tracking_quality', {}).get('manual_anchor_ncc', 0.0)>0.85 else 'score-med'}">{meta.get('tracking_quality', {}).get('manual_anchor_ncc', 0.0):.1%}</td></tr>
                        <tr><th>Anomalies Detected</th><td>{len(meta.get('tracking_quality', {}).get('anomaly_timestamps', []))} times</td></tr>
                    </table>
                </div>
                <div class="card">
                    <h2>Signal Comparison (vs Manual)</h2>
                    <table>
                        <tr><th>DTW Similarity</th><td class="{'score-high' if scores['dtw_similarity']>0.8 else 'score-med'}">{scores['dtw_similarity']:.3f}</td></tr>
                        <tr><th>Coverage</th><td>{scores['coverage']:.1%}</td></tr>
                        <tr><th>Action Count Ratio</th><td>{scores['density_ratio']:.2f}x</td></tr>
                        <tr><th>Mean Abs Error (MAE)</th><td>{scores['mae']:.2f} (pos)</td></tr>
                    </table>
                </div>
            </div>
            
            <h2>Timeline Visualization</h2>
            <img class="plot" src="data:image/png;base64,{plot_b64}" alt="Timeline Plot"/>
            
            <div style="margin-top: 40px; font-size: 0.9em; color: #7f8c8d; text-align: center;">
                Generated automatically by Eroscript Maker Evaluator
            </div>
        </div>
    </body>
    </html>
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)


def evaluate_all():
    """Run evaluation on all matching funscript pairs."""
    import sys
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    if not os.path.isdir(ORIGINAL_DIR) or not os.path.isdir(PROGRAM_DIR):
        print("Error: Sample directories not found.")
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    original_files = {f for f in os.listdir(ORIGINAL_DIR) if f.endswith(".funscript")}
    program_files = {f for f in os.listdir(PROGRAM_DIR) if f.endswith(".funscript")}
    common = sorted(original_files & program_files)

    if not common:
        print("No matching funscript pairs found.")
        return

    print("=" * 90)
    print("FUNSCRIPT QUALITY EVALUATION & REPORT GENERATION")
    print("=" * 90)

    all_scores = []

    for filename in common:
        orig_path = os.path.join(ORIGINAL_DIR, filename)
        prog_path = os.path.join(PROGRAM_DIR, filename)

        orig_actions = load_funscript(orig_path)
        prog_actions, prog_meta = load_funscript_with_meta(prog_path)

        dtw_sim = compute_dtw_similarity(orig_actions, prog_actions)
        density_ratio = compute_action_density_ratio(orig_actions, prog_actions)
        dist_sim = compute_position_distribution_similarity(orig_actions, prog_actions)
        coverage = compute_coverage_ratio(orig_actions, prog_actions)
        mae = compute_mean_absolute_error(orig_actions, prog_actions)
        
        # Penalize composite score based on tracking loss
        trk_loss = prog_meta.get('tracking_quality', {}).get('tracking_loss_ratio', 0.0)

        composite = (
            dtw_sim * 0.35 +
            min(density_ratio, 1.0 / max(density_ratio, 0.01)) * 0.15 +
            dist_sim * 0.20 +
            coverage * 0.15 +
            max(0, 1.0 - mae / 50.0) * 0.15
        ) - (trk_loss * 0.2)

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
        
        # Generate Reports
        plot_b64 = generate_plot_base64(orig_actions, prog_actions, name)
        
        html_path = os.path.join(REPORTS_DIR, f"{name}_report.html")
        generate_html_report(filename, scores, prog_meta, plot_b64, html_path)
        
        json_path = os.path.join(REPORTS_DIR, f"{name}_diagnostic.json")
        generate_json_diagnostic(orig_actions, prog_actions, prog_meta, scores, json_path)

        print(f"\n  {name[:37]+'...' if len(name)>40 else name}")
        print(f"  {'─' * 60}")
        print(f"  Actions       : orig={len(orig_actions):>5}  prog={len(prog_actions):>5}  ratio={density_ratio:.2f}x")
        print(f"  DTW Similarity: {dtw_sim:.4f}")
        print(f"  Trk Loss Ratio: {trk_loss:.1%}")
        print(f"  Composite     : {composite:.4f}")
        print(f"  -> Generated: {os.path.basename(html_path)} & {os.path.basename(json_path)}")

    print(f"\n{'=' * 90}")
    print(f"Finished! All reports are saved in: {REPORTS_DIR}")
    print(f"{'=' * 90}")

    return all_scores


def evaluate_single(prog_path):
    """Run evaluation on a specific program-generated funscript."""
    if not os.path.exists(prog_path):
        return
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = os.path.basename(prog_path)
    orig_path = os.path.join(ORIGINAL_DIR, filename)
    
    # Try looking in the same directory as the video if original_dir doesn't have it
    # or just use original_dir if that's the standard.
    if not os.path.exists(orig_path):
        # Fallback to looking for "original" folder near the program file
        alt_orig_dir = os.path.join(os.path.dirname(prog_path), "funscript_original")
        alt_orig_path = os.path.join(alt_orig_dir, filename)
        if os.path.exists(alt_orig_path):
            orig_path = alt_orig_path

    prog_actions, prog_meta = load_funscript_with_meta(prog_path)
    
    # If original exists, do full comparison. If not, generate a basic diagnostic report.
    if os.path.exists(orig_path):
        orig_actions = load_funscript(orig_path)
        dtw_sim = compute_dtw_similarity(orig_actions, prog_actions)
        density_ratio = compute_action_density_ratio(orig_actions, prog_actions)
        dist_sim = compute_position_distribution_similarity(orig_actions, prog_actions)
        coverage = compute_coverage_ratio(orig_actions, prog_actions)
        mae = compute_mean_absolute_error(orig_actions, prog_actions)
        trk_loss = prog_meta.get('tracking_quality', {}).get('tracking_loss_ratio', 0.0)
        
        composite = (
            dtw_sim * 0.35 +
            min(density_ratio, 1.0 / max(density_ratio, 0.01)) * 0.15 +
            dist_sim * 0.20 +
            coverage * 0.15 +
            max(0, 1.0 - mae / 50.0) * 0.15
        ) - (trk_loss * 0.2)
        
        scores = {
            "dtw_similarity": dtw_sim,
            "density_ratio": density_ratio,
            "distribution_similarity": dist_sim,
            "coverage": coverage,
            "mae": mae,
            "composite": composite,
        }
        plot_b64 = generate_plot_base64(orig_actions, prog_actions, filename)
    else:
        # Diagnostic only
        orig_actions = []
        scores = {
            "dtw_similarity": 0.0,
            "density_ratio": 0.0,
            "distribution_similarity": 0.0,
            "coverage": 0.0,
            "mae": 0.0,
            "composite": 0.0,
        }
        plot_b64 = generate_plot_base64([], prog_actions, f"{filename} (no original)")

    name = os.path.splitext(filename)[0]
    html_path = os.path.join(REPORTS_DIR, f"{name}_report.html")
    generate_html_report(filename, scores, prog_meta, plot_b64, html_path)
    
    json_path = os.path.join(REPORTS_DIR, f"{name}_diagnostic.json")
    generate_json_diagnostic(orig_actions, prog_actions, prog_meta, scores, json_path)


if __name__ == "__main__":
    evaluate_all()
