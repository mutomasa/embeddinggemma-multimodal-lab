# EmbeddingGemma Multimodal Lab

`google/embeddinggemma-2` を使って、テキスト・画像・動画の意味検索と、
Matryoshka 次元（128 / 256 / 512 / 768）ごとの品質・性能差を確認するローカルデモです。

開発方針は [docs/development_policy.md](docs/development_policy.md)、進捗は [todo.md](todo.md) を参照してください。

## セットアップ

```bash
uv sync
uv run python scripts/smoke_text.py   # Phase 0: 5件の文書で日英検索のスモークテスト
uv run streamlit run app.py           # デモアプリ（http://localhost:8501）
```

GPU が他のプロセスで埋まっている場合は `EGLAB_DEVICE=cpu uv run streamlit run app.py` で CPU 実行できます。

## 画面

| ページ | 内容 |
|---|---|
| Search | 日英クエリの Top-K 検索、128 / 256 / 512 / 768 次元切り替え、ラベル絞り込み、エンコード・検索・インデックス構築時間、キャッシュ hit/miss |
| Multimodal Search | テキスト → 画像、画像 → 画像、テキスト → 動画（フレーム最大スコア / フレーム平均ベクトル / ネイティブ動画入力）、ベストフレームのタイムスタンプから再生 |
| Embedding Analysis | UMAP 2D（ラベル / モダリティ別の色分け、クリック・投げ縄で詳細表示）、ラベル順の類似度行列、次元別の k 近傍一致率、クエリ別の順位変動（top-k overlap, Kendall τ, バンプチャート） |
| Evaluation | 正解ラベル付き評価セットで 4 次元を同条件比較（Recall@K / MRR / nDCG@K / P@K、p50 / p95 レイテンシ、保存量）、誤検索ケース、CSV / JSON 出力 |

データセットはサイドバーで「サンプル文書（日英 41件、10カテゴリ）」か CSV / TXT のアップロードを選びます。
CSV は `text`（または `content` / `body`）列が必須で、任意で `doc_id` / `title` / `category` を使います。

## 評価の再実行

```bash
uv run python scripts/run_evaluation.py --task text
uv run python scripts/run_evaluation.py --task image
uv run python scripts/run_evaluation.py --task video --video-method max_frame   # mean_vector / native
```

結果は `reports/<timestamp>-<eval_set>/`（summary.csv, per_query.csv, failures.csv, report.json）に出力されます。
定義と測定方法は [docs/evaluation_protocol.md](docs/evaluation_protocol.md)、構成は [docs/architecture.md](docs/architecture.md)。

### 参考結果（RTX A1000, bfloat16, revision 914f7f8）

| 評価セット | 指標 | 128 | 256 | 512 | 768 |
|---|---|---|---|---|---|
| text（20 クエリ / 41 文書） | nDCG@10 | 0.905 | 0.907 | 0.917 | 0.916 |
| image（14 クエリ / 17 枚） | nDCG@10 | 0.948 | 0.965 | 0.965 | 0.948 |
| video max_frame（8 クエリ / 8 本） | nDCG@10 | 0.614 | 0.677 | 0.630 | 0.653 |
| — | 保存量 / ベクトル | 512 B | 1 KB | 2 KB | 3 KB |

クエリエンコードは p50 約 19 ms（次元に依存しない）、検索は全次元で 0.03 ms 程度です。
サンプルは小規模なため次元間の差は誤差の範囲で、合成動画は識別が難しく P@1 がチャンスレベルです（Issue 参照）。

## マルチモーダルのサンプルデータ

`datasets/sample/images/`・`datasets/sample/videos/` は初回起動時（または `scripts/smoke_multimodal.py`）に
`src/embeddinggemma_lab/sample_media.py` が決定的に生成する**合成データ**です（著作権フリー、Git 管理外）。

- 画像 17 枚：図形・風景・食べ物・ロボットハンドなどの簡単なイラスト
- 動画 8 本（4 秒, 10fps）：2 指グリッパがブロックを掴む 2D アニメ。結果は 成功 / 滑り / 落下 / 回復 × 赤・青

サイドバーから画像・動画を追加すると `datasets/uploads/` に保存されます（Git 管理外）。
動画は一定間隔（既定 0.5 秒）でフレーム抽出し、`video_id` と `timestamp` を保持します。

## 埋め込みキャッシュ

- `.cache/embeddings/<namespace>/<content-hash>.npy` に保存（Git 管理外）
- namespace = モデル ID・revision・次元・役割（query / document）・前処理条件のハッシュ
- 768 次元で 1 回だけエンコードし、128 / 256 / 512 / 768 の全次元分を保存するため、次元切り替えで再エンコードは発生しません

## テスト

```bash
uv run pytest -m "not model"   # モデル不要のユニットテスト（CI と同じ）
uv run pytest -m model         # 実モデルを使う統合テスト（初回はモデルをダウンロード）
EGLAB_DEVICE=cpu uv run pytest -m model   # GPU が埋まっている場合は CPU を強制
uv run ruff check . && uv run ruff format --check .
```

## 動作確認済み環境（Phase 0）

| 項目 | 値 |
|---|---|
| モデル | `google/embeddinggemma-2` @ `914f7f8` |
| Python | 3.12 |
| torch / transformers / sentence-transformers | 2.14.1 / 5.20.0 / 6.1.0 |
| GPU | NVIDIA RTX A1000 8GB（bfloat16） |

注意点:
- モデルカードの指示により **float16 は使用しません**（NaN になる）。CUDA で bf16 対応なら bfloat16、それ以外は float32。
- テキストのみでロードする場合でも、プロセッサの初期化に `pillow` と `torchvision` が必要です。
- ネイティブ動画入力のデコードには `torchcodec`（システムの FFmpeg 共有ライブラリ）が必要です。フレーム抽出は FFmpeg 同梱の OpenCV で行います。
- 次元の切り詰めは 768 次元の正規化済みベクトルを先頭から切り出して再正規化します（`truncate_dim` + `normalize_embeddings` と等価）。
