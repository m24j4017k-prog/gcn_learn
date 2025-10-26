



| キー名                                 | 説明                                                                                 | デフォルト                  |
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


gcn_learn/<br>
├── main.py<br>
│ └─ 実行エントリポイント<br>
├── Processor.py<br>
│ └─ 実験全体の制御（train/test統合）<br>
│<br>
├── config/<br>
│ └── default.yaml<br>
│ └─ モデル・学習条件などの設定ファイル<br>
│<br>
├── dataset/<br>
│ ├── base_dataset.py<br>
│ │ └─ データセットの基底クラス<br>
│ ├── builder.py<br>
│ │ └─ データローダ生成<br>
│ ├── tools.py<br>
│ │ └─ 前処理・補助関数<br>
│ ├── leave_pair_out.py<br>
│ │ └─ データ分割手法<br>
│ └── walk_path_leave_pair_out.py<br>
│ └─ 特定の分割手法実装<br>
│<br>
├── data/<br>
│ └── loader.py<br>
│ └─ DataLoader 定義<br>
│<br>
├── model/<br>
│ ├── builder.py<br>
│ │ └─ モデル構築ヘルパ<br>
│ ├── st_gcn.py<br>
│ │ └─ ST-GCN モデル定義<br>
│ └── utils/<br>
│ └── tgcn.py<br>
│ └─ Temporal GCN モジュール<br>
│<br>
├── graph/<br>
│ ├── coco.py, h36m.py, pyskl.py<br>
│ │ └─ 関節構造（グラフ定義）<br>
│ └── tools.py<br>
│ └─ グラフユーティリティ<br>
│<br>
├── trainer/<br>
│ └── train_loop.py<br>
│ └─ 学習ループ管理<br>
│<br>
├── utils/<br>
│ ├── arg_parser.py<br>
│ │ └─ コマンドライン引数管理<br>
│ ├── seed.py<br>
│ │ └─ 乱数固定<br>
│ └── visualize.py<br>
│ └─ 可視化<br>
│<br>
├── results/<br>
│ └── work_dir/config.yaml<br>
│ └─ 実験ログ・結果保存先<br>
│<br>
└── vis/<br>
└── tmp.mp4<br>
└─ 可視化出力例<br>


---

