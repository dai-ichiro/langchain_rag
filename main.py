import yaml
from pathlib import Path
import gradio as gr
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings

# 設定ファイルの読み込み
settings_path = Path(__file__).parent / "settings.yaml"
with open(settings_path, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)


# 利用する LLM モデル名
LLM_MODEL = settings["llm_model"]
# 利用する埋め込みモデル名
EMBEDDING_MODEL = settings["embedding_model"]
# Chroma の永続化ディレクトリ
CHROMA_DIR = settings["chroma_dir"]


# ===== モデル／ベクターストアの初期化 =====

model = ChatOllama(model=LLM_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


# ===== ミドルウェア: 検索コンテキスト付きプロンプト =====

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """
    ベクターストアから関連ドキュメントを検索し、
    その内容をコンテキストとしてプロンプトに挿入するミドルウェア。
    """
    # 直近メッセージの text をクエリとして利用
    last_query = request.state["messages"][-1].text

    # ベクターストアから関連ドキュメントを検索
    retrieved_docs = vector_store.similarity_search(last_query)

    # 検索結果ドキュメントの本文を結合
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # システムメッセージとして LLM に渡すプロンプト
    system_message = (
        "あなたは社内ドキュメントに詳しいアシスタントです。"
        "以下のコンテキストを元に、日本語で丁寧に回答してください。"
        "不明な点があれば、その旨を正直に伝えてください。:"
        f"\n\n{docs_content}"
    )

    return system_message


# ===== Agent の生成 =====

agent = create_agent(model, tools=[], middleware=[prompt_with_context])


def process_query_stream(query: str):
    """
    クエリを受け取り、RAGで検索・回答をストリーミング生成する関数
    """
    if not query.strip():
        yield "質問を入力してください。"
        return

    # Agent からストリーミング形式で応答を取得
    response_text = ""
    for token, _ in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="messages",
    ):
        if token.content_blocks:
            response_text += token.content_blocks[0]["text"]
            yield response_text


if __name__ == "__main__":
    # Gradio インターフェースの構築
    with gr.Blocks(title="社内ドキュメント検索RAG") as demo:
        gr.Markdown("# 社内ドキュメント検索システム")
        gr.Markdown("質問を入力すると、ベクトルデータベースから関連ドキュメントを検索して回答します。")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="質問を入力",
                    placeholder="ここに質問を入力して下さい",
                    lines=3
                )
                submit_btn = gr.Button("検索", variant="primary")

            with gr.Column():
                output_text = gr.Markdown(
                    label="回答",
                    buttons=["copy"]
                )

        # ボタンクリック時の処理（ストリーミング対応）
        submit_btn.click(
            fn=process_query_stream,
            inputs=input_text,
            outputs=output_text
        )

        # Enterキーでも送信可能に（ストリーミング対応）
        input_text.submit(
            fn=process_query_stream,
            inputs=input_text,
            outputs=output_text
        )

    # Gradio アプリケーションを起動
    demo.launch()
