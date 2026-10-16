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

## 既知の問題（GitHub Issues）
- [ ] #6 合成グリッパ動画の検索がチャンスレベル
- [ ] #7 サンプル評価セットが易しすぎて次元差が出ない
- [ ] #8 「滑り」クエリで「落下」文書が 1 位になる
- [ ] #9 類似度に言語による上乗せがある
- [ ] #10 依存関係の注意点（pillow / torchvision / torchcodec / FFmpeg）
- [ ] #11 CUDA OOM 時に CPU へ自動フォールバックしない
- [ ] #12 Search 画面のエンコード時間表示がキャッシュの影響を受ける
- [ ] #13 埋め込みキャッシュのスケーラビリティと容量管理

## Phase 5 — Research Demo（`feature/phase5-research-demo`）
- [x] 人手ラベル形式（動画 × 時間区間 × イベント）の読み込みと時刻 → イベント照会
- [x] イベントクエリ（graded events）とアノテーションからのフレーム / 動画単位の正解導出
- [x] 検索結果フレームと人手ラベルの照合（一致率@K、✅/❌、タイムライン表示）
- [x] 全クエリの 1 位フレーム × 人手ラベルの混同行列
- [x] 4 次元の定量評価（既存評価エンジンを再利用）と `scripts/run_research_eval.py`
- [x] 合成グリッパ動画用の区間ラベル・クエリ（`datasets/sample/research/`）
- [x] Streamlit「Research Demo」ページ、README / architecture 更新
- [ ] #14 実ロボット把持動画と人手ラベル（未同梱）
- [ ] #15 合成動画では 1 位フレームが「落下」区間に偏る
- [ ] #16 `corpus_digest` がパスを含まない
