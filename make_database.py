import yaml
import shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, Docx2txtLoader, PyPDFLoader, TextLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 設定ファイルの読み込み
settings_path = Path(__file__).parent / "settings.yaml"
with open(settings_path, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)

# 対応するファイルパターンとローダークラスの対応表
LOADER_CONFIG = [
    ("**/*.pdf", PyPDFLoader),
    ("**/*.docx", Docx2txtLoader),
    ("**/*.pptx", UnstructuredPowerPointLoader),
    ("**/*.xlsx", UnstructuredExcelLoader),
    ("**/*.txt", TextLoader),
]

# ドキュメント配置ディレクトリ
DOCS_DIR = settings["docs_dir"]

def load_documents_with_metadata(docs_dir: str) -> list[Document]:
    """
    docs_dir 以下から PDF・Word・Excel・PowerPoint 等を読み込む。
    """
    documents: list[Document] = []

    for pattern, loader_cls in LOADER_CONFIG:
        print(f"[SCAN] pattern={pattern}")

        loader = DirectoryLoader(
            docs_dir,
            glob=pattern,
            loader_cls=loader_cls,
            show_progress=True,
            silent_errors=True,
        )

        # 指定パターンにマッチする全ファイルをロード
        for doc in loader.load():
            documents.append(doc)

            path = doc.metadata.get("source")
            print(f"  🟢 追加: {path}")

    return documents

docs = load_documents_with_metadata(DOCS_DIR)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings["chunk_size"],
    chunk_overlap=settings["chunk_overlap"],
    separators=["\n\n", "\n", "。", " ", ""],
    length_function=len,
)
texts = splitter.split_documents(docs)


from langchain_chroma import Chroma
#from langchain_ollama import OllamaEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

# 利用する埋め込みモデル名
EMBEDDING_MODEL = settings["embedding_model"]
# Chroma の永続化ディレクトリ
CHROMA_DIR = settings["chroma_dir"]

# Chroma永続化ディレクトリが存在する場合は削除
chroma_path = Path(CHROMA_DIR)
if chroma_path.exists():
    print(f"[INFO] 既存のChromaディレクトリを削除します: {CHROMA_DIR}")
    shutil.rmtree(chroma_path)
    print(f"[INFO] 削除完了")

#embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

vectorstore.add_documents(documents=texts)
print(f"[INFO] {len(texts)} 件のドキュメントをChromaに追加しました")