# test_dataset_visualize.py
import os
import numpy as np
import torch
from dataset.builder import build_dataset
from utils.arg_parser import get_parser
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

# === 可視化関数 ===
def get_img_from_fig(fig, dpi=180):
    """Figure → np.array"""
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img



def motion2video_3d3(motion, save_path="vis/tmp.mp4", fps=2, view=[15, 105]):
    """
    motion: (C, T, V, M) or (V, 3, T)
    save_path: 保存先
    """
    motion = np.transpose(motion, (1, 2, 0))  # (V,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]

    joint_pairs = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10],
        [11, 12], [12, 13], [14, 15], [15, 16]
    ]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    for f in tqdm(range(vlen), desc="Rendering"):
        j3d = motion[:, :, f]
        fig = plt.figure(0, figsize=(8, 8))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=view[0], azim=view[1])
        plt.tight_layout()

        for limb in joint_pairs:
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if limb in joint_pairs_left:
                color = color_left
            elif limb in joint_pairs_right:
                color = color_right
            else:
                color = color_mid
            ax.plot(xs, ys, zs, color=color, lw=3, marker='o', markersize=6)

        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()

    videowriter.close()
    print(f"✅ Saved video: {save_path}")

# === Datasetから1サンプルを可視化 ===
def visualize_sample(config_path, index=0, split='train'):
    parser = get_parser()
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    parser.set_defaults(**cfg)
    arg = parser.parse_args([])

    dataset = build_dataset(arg, split=split)
    batch = next(iter(dataset))  # (data, label, index)

    data, label, _ = batch        
    data = data.numpy() if torch.is_tensor(data) else data

    # (B,C,T,V,M) → (B,M,T,V,C)
    data = np.transpose(data, (0, 4, 2, 3, 1))
    motion = data[0, 0]

    save_dir = "vis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split}_sample{index}_label{label}.mp4")

    motion2video_3d3(motion)

if __name__ == "__main__":
    visualize_sample("config/default.yaml", index=0, split='test')
