import os
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import FRAMES_DIR, FEATURES_DIR, CLASSES


def main():
    device = "cpu"
    print("Loading CLIP ViT-B/32 (downloads ~340MB on first run)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    for cls in CLASSES:
        cls_frame_dir = os.path.join(FRAMES_DIR, cls)
        cls_feat_dir  = os.path.join(FEATURES_DIR, cls)
        os.makedirs(cls_feat_dir, exist_ok=True)

        npy_files = sorted([
            f for f in os.listdir(cls_frame_dir)
            if f.endswith(".npy")
        ])

        print(f"\nExtracting CLIP features — {cls} ({len(npy_files)} videos)")

        for npy_file in tqdm(npy_files, desc=cls):
            out_path = os.path.join(cls_feat_dir, npy_file)
            if os.path.exists(out_path):
                continue

            frames = np.load(os.path.join(cls_frame_dir, npy_file))
            # frames: (N, 224, 224, 3)

            embeddings = []
            for frame in frames:
                img = Image.fromarray(frame.astype(np.uint8))
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.squeeze(0).cpu().numpy())

            np.save(out_path, np.array(embeddings))  # (N, 512)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()