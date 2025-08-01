import argparse
import os
import sys
import subprocess
from typing import List

try:
    import numpy as np
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy"])
    import numpy as np

import torch
from tqdm import tqdm


def ensure_dependencies():
    try:
        import einops
    except Exception:
        print("Installing dependencies...")
        packages = [
            "einops",
            "peft",
            "open_clip_torch",
            "protobuf",
            "sentencepiece",
            "iv2-utils",
            "matplotlib",
            "huggingface_hub",
            "tabulate",
            "tqdm",
        ]
        for package in tqdm(packages, desc="Installing packages"):
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                package,
            ])
    print("Dependencies installed/verified")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark StreamMamba vs InternVideo2 using dynamically weighted embeddings",
    )
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
    )
    parser.add_argument(
        "--output-json",
        default="window_dynamic_mae_results.json",
        help="Path to output JSON with MAE results",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file. If not provided, will download from HF.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="spfs_r64/ckpt_step_24500.pt",
        help="Checkpoint filename within the HF repo",
    )
    parser.add_argument(
        "--hf-repo",
        default="qingy2024/InternVideo2-B14",
        help="HuggingFace repo to download checkpoint from",
    )
    parser.add_argument(
        "--window-sizes",
        default="8,9,10,16",
        help="Comma separated list of window sizes to evaluate",
    )
    parser.add_argument(
        "--equal-weights",
        action="store_true",
        help="Use equal weights for all embeddings instead of confidence-based weighting",
    )
    return parser.parse_args()


def find_closest(pred, truths):
    if not truths:
        return pred
    return min(truths, key=lambda x: abs(x - pred))


def calculate_mse(preds_with_offset, data):
    errors = []
    for idx, p in enumerate(preds_with_offset):
        truth_peaks = data[idx][2]
        if not truth_peaks:
            continue
        closest_t = find_closest(p, truth_peaks)
        errors.append((p - closest_t) ** 2)
    return np.mean(errors) if errors else float("inf")


def find_best_offset(preds, data, search_range=(-30, 30)):
    best_offset = 0
    best_mse = float("inf")
    for off in range(search_range[0], search_range[1] + 1):
        shifted = [p + off for p in preds]
        mse = calculate_mse(shifted, data)
        if mse < best_mse:
            best_mse = mse
            best_offset = off
    return best_offset


def offset_predictions(preds, data):
    best = find_best_offset(preds, data)
    return [p + best for p in preds]


def compute_accuracy(preds: List[int], dataset: List) -> dict:
    thresholds = [2, 4, 8, 16, 32]
    preds_adj = offset_predictions(preds, dataset)
    totals = {t: 0 for t in thresholds}

    for pred, entry in zip(preds_adj, dataset):
        gt_frames = entry[2]
        if not gt_frames:
            continue
        diff = min(abs(pred - f) for f in gt_frames)
        for t in thresholds:
            if diff <= t:
                totals[t] += 1

    n = len(preds_adj)
    percentages = {f"within_{t}": totals[t] * 100.0 / n for t in thresholds}
    percentages["average"] = sum(percentages.values()) / len(thresholds)
    return percentages


def streammamba_predict(frames, phrase, model, device, size_t, window_size, equal_weights=False):
    from demo.utils import frames2tensor

    hidden = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
    embeddings = []
    confidences = []
    logits = []
    text_feat = model.get_txt_feat(phrase)

    for frame in frames:
        tensor = frames2tensor([frame], fnum=1, target_size=(size_t, size_t), device=device)
        tensor = tensor.squeeze(0)
        emb, hidden, spfs_info = model.get_streaming_vid_feat(
            tensor,
            hidden,
            confidence_threshold=1.0,
            max_consecutive_skips=0,
        )
        embeddings.append(emb)
        confidences.append(spfs_info.confidence)

        if len(embeddings) <= 7:
            logits.append(0.0)
            continue

        k = max(1, window_size - 7)
        if k == 1:
            use_emb = embeddings[-1]
        else:
            recent_embs = embeddings[-k:]
            if equal_weights:
                # Use uniform weights
                weights = torch.ones(k, device=device) / k
                emb_stack = torch.stack(recent_embs)
                use_emb = (emb_stack * weights.view(-1, 1)).sum(dim=0)
            else:
                # Use confidence-based weighting
                recent_conf = confidences[-k:]
                weight_scores = torch.tensor([1 - c for c in recent_conf], device=device)
                weights = torch.softmax(weight_scores, dim=0)
                emb_stack = torch.stack(recent_embs)
                use_emb = (emb_stack * weights.view(-1, 1)).sum(dim=0)

        probs, _ = model.predict_label(use_emb, text_feat, top=1)
        probs = probs.mean(dim=0)
        logits.append(probs.item())

    return int(np.argmax(logits) + 1), logits


def internvideo2_predict(frames, phrase, model, device, size_t, window_size):
    from demo.utils import frames2tensor

    text_feat = model.get_txt_feat(phrase)
    logits = []
    for idx in range(len(frames)):
        if idx + 1 < window_size:
            logits.append(0.0)
            continue
        window_frames = frames[idx + 1 - window_size : idx + 1]
        tensor = frames2tensor(
            window_frames, fnum=8, target_size=(size_t, size_t), device=device
        )
        vid_feat = model.get_vid_feat(tensor)
        probs, _ = model.predict_label(vid_feat, text_feat, top=1)
        logits.append(probs.item())
    return int(np.argmax(logits) + 1), logits


def main():
    ensure_dependencies()
    args = parse_args()

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    from huggingface_hub import hf_hub_download
    from iv2_utils.iv2 import json_read, json_write
    from tabulate import tabulate
    import cv2

    use_equal_weights = args.equal_weights

    use_spfs_weights = not use_equal_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "photography-model" not in os.listdir("."):
        subprocess.check_call([
            "git",
            "clone",
            "https://github.com/ruo2019/photography-model.git",
        ])

    act75_data = json_read("photography-model/data/ACT75.json")

    config_path = os.path.join(args.config_dir, "config.py")
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)

    model = InternVideo2_CLIP_small(config)
    model.to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.checkpoint_file)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model") or ckpt.get("module")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    size_t = config.get("size_t", 224)
    os.makedirs("results_window", exist_ok=True)

    window_sizes = [int(x) for x in args.window_sizes.split(",") if x.strip()]

    results = {}

    for N in tqdm(window_sizes, desc="Processing window sizes"):
        preds_stream = []
        logits_stream = []
        preds_intern = []
        logits_intern = []
        for video_path, phrase, _ in tqdm(
            act75_data, desc=f"Processing videos for window size {N}", leave=False
        ):
            frames = [
                f
                for f in _frame_from_video(
                    cv2.VideoCapture(os.path.join("photography-model", video_path))
                )
            ]
            pred_s, log_s = streammamba_predict(
                frames, phrase, model, device, size_t, N, use_equal_weights
            )
            preds_stream.append(pred_s)
            logits_stream.append([(float(l), i + 1) for i, l in enumerate(log_s)])

            if use_spfs_weights:
                pred_i, log_i = internvideo2_predict(
                    frames, phrase, model, device, size_t, N
                )
                preds_intern.append(pred_i)
                logits_intern.append([(float(l), i + 1) for i, l in enumerate(log_i)])

        metrics_stream = compute_accuracy(preds_stream, act75_data)
        results[N] = {
            "streammamba": metrics_stream,
        }
        if use_spfs_weights:
            metrics_intern = compute_accuracy(preds_intern, act75_data)
            results[N]["internvideo2"] = metrics_intern

        suffix = "_equal" if use_equal_weights else ""
        json_write(
            preds_stream, os.path.join("results_window", f"streammamba_preds_window_{N}{suffix}.json")
        )
        json_write(
            logits_stream, os.path.join("results_window", f"streammamba_logits_window_{N}{suffix}.json")
        )
        if use_spfs_weights:
            json_write(
                preds_intern, os.path.join("results_window", f"internvideo2_preds_window_{N}.json")
            )
            json_write(
                logits_intern, os.path.join("results_window", f"internvideo2_logits_window_{N}.json")
            )

    if use_equal_weights:
        table = [
            (N, results[N]["streammamba"]["average"])
            for N in window_sizes
        ]
        headers = ["Window Size", "StreamMamba (Equal Weights)"]
    else:
        table = [
            (N, results[N]["streammamba"]["average"], results[N]["internvideo2"]["average"])
            for N in window_sizes
        ]
        headers = ["Window Size", "StreamMamba", "InternVideo2"]
    print(tabulate(table, headers=headers, floatfmt=".2f"))

    json_write(results, args.output_json)
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
