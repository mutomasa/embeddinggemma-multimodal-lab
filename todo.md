# TODO — EmbeddingGemma Multimodal Lab

`docs/development_policy.md` の Phase 0〜4 を実装する。各 Phase は feature ブランチ → PR で取り込む。

## Phase 0 — Feasibility（`feature/phase0-feasibility`）
- [x] `uv` プロジェクトと `src/` 構成を作成
- [x] モデルカード推奨コードでテキストのスモークテスト
- [x] CPU / CUDA 判定と dtype 選択（float16 禁止、bf16 / fp32）
- [x] クエリ・文書のエンコード API を統一（`SearchQuery` / `Document` プロンプト）
- [x] 5件の文書に対する日本語・英語検索がローカルで実行できる
- [x] pytest / ruff と GitHub Actions CI

## Phase 1 — MVP（`feature/phase1-mvp`）
- [x] サンプル文書セット（日英）を作成
- [x] CSV / TXT 文書の取り込み
- [x] コサイン類似度 Top-K（NumPy）
- [x] 128 / 256 / 512 / 768 次元切り替え（切り詰め → 再正規化）
- [x] 埋め込みキャッシュ（キー: モデルID・revision・次元・プロンプト・内容ハッシュ）
- [x] Streamlit 検索画面（Top-K・スコア・エンコード/検索時間）
- [x] `st.cache_resource` でモデル保持、再起動後のキャッシュ再利用

## Phase 2 — Multimodal（`feature/phase2-multimodal`）
- [x] ネイティブ画像・動画入力 API のスモークテスト
- [x] 公開可能な合成サンプル画像・動画の生成スクリプト
- [x] 画像取り込みとテキスト → 画像、画像 → 画像検索
- [x] 動画の一定間隔フレーム抽出（動画ID・timestamp メタデータ）
- [x] フレーム集約：平均 vs 最大スコア
- [x] タイムスタンプ付き動画再生

## Phase 3 — Analysis（`feature/phase3-analysis`）
- [x] UMAP 2D 散布図（モダリティ・ラベル別の色分け）
- [x] サンプル選択による詳細表示
- [x] 類似度行列ヒートマップ
- [x] 次元別の近傍変化・順位変動分析

## Phase 4 — Evaluation（`feature/phase4-evaluation`）
- [x] 正解ラベル付き評価セット（JSONL）
- [x] Recall@K / MRR / nDCG@K / P@K の実装とユニットテスト
- [x] ウォームアップ後の複数回測定（p50 / p95）、エンコード・インデックス構築・検索を分離
- [x] 実行環境（モデル revision・dtype・ハードウェア・次元・バッチサイズ）の記録
- [x] 4次元を同条件で再実行し CSV / JSON 出力
- [x] 誤検索ケースの例示
- [x] `docs/evaluation_protocol.md`, `docs/architecture.md`, README 整備
