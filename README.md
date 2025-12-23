# RAG

各種ドキュメント（PDF, Word, Excel, PowerPoint, Text）を読み込み、それらの内容に基づいて回答するRAG（Retrieval-Augmented Generation）システムです。
ローカルLLM（Ollama）とHuggingFaceのEmbeddingモデル、ChromaDBを使用しています。

## 特徴

- **多様なファイル形式に対応**: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), Text (.txt)
- **ローカル環境で動作**: Ollamaを使用し、プライバシーを重視したローカルLLMでの運用が可能
- **Web UI**: Gradioを使用した使いやすいチャットインターフェース

## 必要条件

- Python 3.13 以上
- NVIDIA CUDA 12.6 以上
- [Ollama](https://ollama.com/) (LLMの実行用)
- [uv](https://github.com/astral-sh/uv) (推奨パッケージマネージャー) または pip

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/dai-ichiro/langchain_rag
cd langchain_rag
```

### 2. 依存関係のインストール

このプロジェクトではパッケージ管理に `uv` を使用しています。

```bash
uv sync
```

### 3. Ollama モデルの準備

設定ファイル (`settings.yaml`) で指定されているモデルを Ollama で pull しておく必要があります。デフォルト設定の場合:

```bash
ollama pull gpt-oss:latest
```
※ `gpt-oss:latest` は設定例です。実際に使用するモデル名に合わせて変更してください。

## 設定

`settings.yaml` ファイルで各種設定を変更できます。

```yaml
# 使用するLLMモデル (Ollama上のモデル名)
llm_model: "gpt-oss:latest"

# 埋め込みモデル (HuggingFace)
embedding_model: "sbintuitions/sarashina-embedding-v2-1b"

# テキスト分割設定
chunk_size: 3000
chunk_overlap: 200

# ドキュメント配置ディレクトリ
docs_dir: "docs"

# ベクトルデータベース保存ディレクトリ
chroma_dir: "chroma_db"
```

## 使い方

### 1. ドキュメントの準備

`docs` ディレクトリ（設定で変更可能）を作成し、検索対象にしたいファイルを配置してください。

```bash
mkdir docs
# ここに .pdf, .docx, .xlsx, .pptx, .txt などを配置
```

### 2. データベースの作成

以下のコマンドを実行して、ドキュメントを読み込み、ベクトルデータベースを作成します。

```bash
uv run make_database.py
```
実行が完了すると、`chroma_db` ディレクトリにデータベースが作成されます。
※ ドキュメントを追加・変更した場合は、再度このコマンドを実行してください。

### 3. アプリケーションの起動

以下のコマンドでWebインターフェースを起動します。

```bash
uv run main.py
```

起動後、ブラウザで表示されるURL（通常は `http://127.0.0.1:7860`）にアクセスしてください。

## 対応ファイル形式

- PDF (`.pdf`)
- Word (`.docx`)
- PowerPoint (`.pptx`)
- Excel (`.xlsx`)
- Text (`.txt`)
