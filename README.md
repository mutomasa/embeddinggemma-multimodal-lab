# EmbeddingGemma Multimodal Lab

`google/embeddinggemma-2` を使って、テキスト・画像・動画の意味検索と、
Matryoshka 次元（128 / 256 / 512 / 768）ごとの品質・性能差を確認するローカルデモです。

開発方針は [docs/development_policy.md](docs/development_policy.md)、進捗は [todo.md](todo.md) を参照してください。

## セットアップ

```bash
uv sync
uv run python scripts/smoke_text.py   # Phase 0: 5件の文書で日英検索のスモークテスト
```

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
- 次元の切り詰めは 768 次元の正規化済みベクトルを先頭から切り出して再正規化します（`truncate_dim` + `normalize_embeddings` と等価）。
