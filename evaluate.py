import os
import clip
import torch
import numpy as np
from config import FEATURES_DIR, CLASSES, TOPK
from aggregation import uniform_pooling, tv_weighted_pooling, topk_pooling

PROMPT = "a video of a person doing {}"


def get_text_embeddings(model, device):
    texts  = [PROMPT.format(cls.replace("_", " ")) for cls in CLASSES]
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy()


def load_all_features():
    features = {}
    for cls in CLASSES:
        cls_dir = os.path.join(FEATURES_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: {cls} feature dir not found, skipping.")
            features[cls] = []
            continue
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
        features[cls] = [np.load(os.path.join(cls_dir, f)) for f in files]
        print(f"  {cls}: {len(features[cls])} videos")
    return features


def evaluate_strategy(features_by_class, text_embs, strategy_fn):
    """Returns overall accuracy and per-class accuracy dict."""
    correct, total = 0, 0
    per_class = {}

    for true_idx, cls in enumerate(CLASSES):
        cls_correct, cls_total = 0, 0
        for emb in features_by_class[cls]:
            video_emb = strategy_fn(emb)
            video_emb = video_emb / (np.linalg.norm(video_emb) + 1e-8)
            pred = int(np.argmax(text_embs @ video_emb))
            if pred == true_idx:
                correct += 1
                cls_correct += 1
            total += 1
            cls_total += 1
        per_class[cls] = cls_correct / cls_total if cls_total > 0 else 0.0

    overall = correct / total if total > 0 else 0.0
    return overall, per_class


def main():
    device = "cpu"
    print("Loading CLIP for text encoding...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    text_embs = get_text_embeddings(model, device)

    print("\nLoading video features...")
    features = load_all_features()

    strategies = {
        "Uniform Pooling (Baseline)": uniform_pooling,
        "TV-Weighted Pooling (Ours)": tv_weighted_pooling,
        "Top-K Pooling (Ablation)":   lambda e: topk_pooling(e, k=TOPK),
    }

    print("\n--- Overall Results ---")
    results         = {}
    per_class_results = {}

    for name, fn in strategies.items():
        overall, per_class = evaluate_strategy(features, text_embs, fn)
        results[name]          = overall
        per_class_results[name] = per_class
        print(f"  {name}: {overall * 100:.2f}%")

    print("\n--- Per-Class Breakdown ---")
    header = f"{'Class':<20}" + "".join(f"{n[:12]:<16}" for n in strategies)
    print(header)
    print("-" * len(header))
    for cls in CLASSES:
        row = f"{cls:<20}"
        for name in strategies:
            acc = per_class_results[name][cls] * 100
            row += f"{acc:<16.1f}"
        print(row)

    return results, per_class_results


if __name__ == "__main__":
    main()