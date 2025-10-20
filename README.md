



| キー名                                 | 説明                                                                                 | 例                           |
| :------------------------------------- | :----------------------------------------------                                      | :-------------------------- |
| `model`                                | 使用するモデルを指定します。パスは `model/` 内のPythonモジュールに対応します。       | `model.st_gcn.Model`        |
| `model_args.in_channels`               | 入力チャネル数。3D座標なら3、2D座標なら2を指定します。                               | `3`                         |
| `model_args.num_person`                | 同時に扱う人物数。                                                                   | `2`                         |
| `model_args.num_class`                 | 分類クラス数。                                                                       | `3`                         |
| `model_args.edge_importance_weighting` | エッジの重要度を学習するかどうか。                                                   | `True`                      |
| `model_args.graph`                     | 使用する骨格構造（グラフ）を指定します。                                             | `graph.h36m.Graph`          |
| `model_args.graph_args.labeling_mode`  | グラフのラベル付けモード。通常は `"spatial"`。                                       | `"spatial"`                 |
| `loss`                                 | 使用する損失関数。PyTorchモジュールを指定します。                                    | `torch.nn.CrossEntropyLoss` |
| `loss_args`                            | 損失関数に渡す追加パラメータ。                                                       | `{}`                        |
| `evaluation_method`                    | 評価方法（データ分割戦略など）。                                                     | `walk_path_leave_pair_out`  |
| `train_data_path`                      | 学習用データファイル（NumPy形式）のパス。                                            | `./data/MB_3DP/1/data.npy`  |
| `train_label_path`                     | 学習用ラベルファイルのパス。                                                         | `./data/MB_3DP/1/label.npy` |
| `test_data_path`                       | テスト用データファイルのパス。                                                       | `./data/MB_3DP/1/data.npy`  |
| `test_label_path`                      | テスト用ラベルファイルのパス。                                                       | `./data/MB_3DP/1/label.npy` |
| `leave_pair`                           | テスト対象の被験者やペアの指定。                                                     | `[1, 2, 3, 4]`              |
| `train_walkpath`                       | 学習時に使用するシーケンスまたは被験者設定。                                         | 任意のリストやパス指定                 |




## プロジェクト構成概要 
## gcn_learn プロジェクト構成

本プロジェクトは、GCNベースのアクション認識フレームワークです。以下はディレクトリ構成と各ファイルの役割です。


gcn_learn/
├── main.py
│ └─ 実行エントリポイント
├── Processor.py
│ └─ 実験全体の制御（train/test統合）
│
├── config/
│ └── default.yaml
│ └─ モデル・学習条件などの設定ファイル
│
├── dataset/
│ ├── base_dataset.py
│ │ └─ データセットの基底クラス
│ ├── builder.py
│ │ └─ データローダ生成
│ ├── tools.py
│ │ └─ 前処理・補助関数
│ ├── leave_pair_out.py
│ │ └─ データ分割手法
│ └── walk_path_leave_pair_out.py
│ └─ 特定の分割手法実装
│
├── data/
│ └── loader.py
│ └─ DataLoader 定義
│
├── model/
│ ├── builder.py
│ │ └─ モデル構築ヘルパ
│ ├── st_gcn.py
│ │ └─ ST-GCN モデル定義
│ └── utils/
│ └── tgcn.py
│ └─ Temporal GCN モジュール
│
├── graph/
│ ├── coco.py, h36m.py, pyskl.py
│ │ └─ 関節構造（グラフ定義）
│ └── tools.py
│ └─ グラフユーティリティ
│
├── trainer/
│ └── train_loop.py
│ └─ 学習ループ管理
│
├── utils/
│ ├── arg_parser.py
│ │ └─ コマンドライン引数管理
│ ├── seed.py
│ │ └─ 乱数固定
│ └── visualize.py
│ └─ 可視化
│
├── results/
│ └── work_dir/config.yaml
│ └─ 実験ログ・結果保存先
│
└── vis/
└── tmp.mp4
└─ 可視化出力例


---

もし希望なら、**セットアップ方法や実行例、設定ファイルの使い方**まで含めたREADME完全版も作れます。作りますか？
