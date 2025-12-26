import os
import time
import yaml
import torch
import numpy as np
import pandas as pd

# gcn_learnの import_class と build_model を利用
from utils.util import import_class
from model.builder import build_model

# -----------------------------
# 設定読み込み
# -----------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------
# 推定時間・メモリ計測
# -----------------------------
def measure_inference(model, data, device="cuda"):
    model.eval()
    model.to(device)
    data = data.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # ウォームアップ
    with torch.no_grad():
        _ = model(data)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        _ = model(data)
    torch.cuda.synchronize()
    end_time = time.time()
    
    
    # モデルのパラメータ数を計算
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return (end_time - start_time) * 1000, torch.cuda.max_memory_allocated(device)/(1024**2), total_params, trainable_params

# -----------------------------
# メイン処理
# -----------------------------
if __name__ == "__main__":
    device = "cuda"
    print(f"Using device: {device}")

    config_dir = "config/default"
    models = ["stgcn", "aagcn", "ctrgcn"]

    results = []

    for m in models:
        cfg_path = os.path.join(config_dir, m)
        cfg = load_config(f"{cfg_path}.yaml")
        model_cfg = cfg["model"]

        print(model_cfg)

        # build_modelを利用
        model, ModelClass = build_model(model_path=cfg["model"], model_args=cfg["model_args"])
        
        # 学習済みモデルをロード
        # model.load_state_dict(torch.load(f"model/weight/{m}.pt"))

        # 入力データ（必要に応じて T, M を調整）
        sample_data = torch.randn(64, 3, 300, 17, 2)

        time_ms, mem_mb, total_p, trainable_p = measure_inference(model, sample_data, device)
        print(f"Time: {time_ms:.2f} ms | Memory: {mem_mb:.1f} MB")
        print(f"totale_param : {total_p} , trainable_param : {trainable_p}")
        print("")

        results.append([model_cfg, time_ms, mem_mb])

    # 保存
    os.makedirs("results/measure", exist_ok=True)
    np.save("results/measure/inference_performance.npy", np.array(results, dtype=object))
    df = pd.DataFrame(results, columns=["ModelClass", "Time (ms)", "Memory (MB)"])
    df.to_csv("results/measure/inference_performance.csv", index=False)
    print("\nSaved results to results/measure/inference_performance.csv")
