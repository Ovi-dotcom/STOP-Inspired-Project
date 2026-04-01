import os
import io
import base64
import numpy as np
import cv2
import clip
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from aggregation import compute_frame_scores, tv_weighted_pooling
from config import CLASSES, NUM_FRAMES

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading CLIP model...")
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

PROMPT = "a video of a person doing {}"
texts  = [PROMPT.format(cls.replace("_", " ")) for cls in CLASSES]
tokens = clip.tokenize(texts).to(device)
with torch.no_grad():
    text_embs = model.encode_text(tokens)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
text_embs = text_embs.cpu().numpy()
print("CLIP ready.")


def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames


def get_clip_embeddings(frames):
    embeddings = []
    for frame in frames:
        img    = Image.fromarray(cv2.resize(frame, (224, 224)))
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.squeeze(0).cpu().numpy())
    return np.array(embeddings)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="#0a0a1a")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def plot_confidence(scores):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    colors = ["#00fff7" if i == np.argmax(scores) else "#1a4a6b"
              for i in range(len(CLASSES))]
    bars = ax.barh(CLASSES, scores * 100, color=colors,
                   edgecolor="#00fff7", linewidth=0.5)

    ax.set_xlabel("Confidence (%)", color="#00fff7", fontsize=10)
    ax.set_title("Class Confidence Scores", color="#00fff7", fontsize=12)
    ax.tick_params(colors="#00fff7")
    ax.spines["bottom"].set_color("#00fff7")
    ax.spines["left"].set_color("#00fff7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score * 100:.1f}%", va="center",
                color="#00fff7", fontsize=8)

    ax.invert_yaxis()
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_frame_weights(frames, embeddings):
    scores  = compute_frame_scores(embeddings)
    weights = scores / (scores.sum() + 1e-8)
    max_w   = weights.max() if weights.max() > 0 else 1.0
    cmap    = cm.RdYlGn
    N       = len(frames)

    fig, axes = plt.subplots(1, N, figsize=(N * 2, 2.5),
                             facecolor="#0a0a1a")
    if N == 1:
        axes = [axes]

    for i, (frame, ax) in enumerate(zip(frames, axes)):
        ax.imshow(cv2.resize(frame, (112, 112)))
        color = cmap(weights[i] / max_w)
        ax.set_title(f"w={weights[i]:.3f}", fontsize=7,
                     color=color, pad=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0a0a1a")

    fig.suptitle("Frame Temporal Variation Weights  |  Green = High Motion  |  Red = Low Motion",
                 color="#00fff7", fontsize=9, y=1.02)
    fig.patch.set_facecolor("#0a0a1a")
    fig.tight_layout()
    return fig_to_base64(fig)


@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    frames = extract_frames(video_path)
    if len(frames) < 2:
        return jsonify({"error": "Could not extract frames from video"}), 400

    embeddings = get_clip_embeddings(frames)

    video_emb = tv_weighted_pooling(embeddings)
    video_emb = video_emb / (np.linalg.norm(video_emb) + 1e-8)

    raw_scores  = text_embs @ video_emb
    temperature = 100.0
    raw_scores  = raw_scores * temperature
    exp_scores  = np.exp(raw_scores - raw_scores.max())
    conf_scores = exp_scores / exp_scores.sum()

    pred_idx   = int(np.argmax(conf_scores))
    pred_class = CLASSES[pred_idx]
    confidence = float(conf_scores[pred_idx]) * 100

    conf_chart   = plot_confidence(conf_scores)
    weight_chart = plot_frame_weights(frames, embeddings)

    os.remove(video_path)

    return jsonify({
        "predicted_class": pred_class.replace("_", " ").title(),
        "confidence":      f"{confidence:.1f}",
        "conf_chart":      conf_chart,
        "weight_chart":    weight_chart,
        "all_scores": {
            CLASSES[i].replace("_", " ").title(): f"{conf_scores[i] * 100:.1f}"
            for i in range(len(CLASSES))
        }
    })


if __name__ == "__main__":
    app.run(debug=True)