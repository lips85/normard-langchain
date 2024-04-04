import re
import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


if "api_key" not in st.session_state:
    st.session_state["api_key"] = None


if "api_key_check" not in st.session_state:
    st.session_state["api_key_check"] = False
    st.warning("OPENAI_API_KEY를 넣어주세요.")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = None
    st.warning("OpenAI Model을 골라주세요.")

API_KEY_pattern = r"sk-.*"

Model_pattern = r"gpt-*"

openai_models = ["gpt-4-0125-preview", "gpt-3.5-turbo-0125"]

st.title("DocumentGPT")

st.markdown(
    """
안녕하세요! 이 페이지는 문서를 읽어주는 AI입니다. 문서를 업로드하고 질문을 하면 문서에 대한 답변을 해줍니다.
"""
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    os.makedirs("./.cache/files", exist_ok=True)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/open_ai/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", ".", "?", "!"],
        chunk_size=1000,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.session_state["api_key"],
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_check"] = True


def save_openai_model(openai_model):
    st.session_state["openai_model"] = openai_model
    st.session_state["openai_model_check"] = True


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    api_key = st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        disabled=st.session_state["api_key"] != None,
    ).strip()

    if api_key:
        save_api_key(api_key)
        st.write("API_KEY가 저장되었습니다.")

    button = st.button("저장")

    if button:
        save_api_key(api_key)
        if api_key == "":
            st.warning("OPENAI_API_KEY를 넣어주세요.")

    openai_model = st.selectbox("OpneAI Model을 골라주세요.", openai_models)

    if re.match(Model_pattern, openai_model):
        save_openai_model(openai_model)
        st.write("모델이 저장되었습니다.")

    st.write("Made by harry.")

    st.write(
        "https://github.com/lips85/normard-langchain/blob/main/pages/01_DoucumentGPT.py"
    )

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks={
        ChatCallbackHandler(),
    },
    model=st.session_state["openai_model"],
    openai_api_key=st.session_state["api_key"],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI that reads documents for me. Please answer based on the document given below. 
            If the information is not in the document, answer the question with “The required information is not in the document.” Never make up answers.
            Please answer in the questioner's language 
            
            Context : {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


if (st.session_state["api_key_check"] == True) and (
    st.session_state["api_key"] != None
):
    if file:
        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            if re.match(API_KEY_pattern, st.session_state["api_key"]):
                send_message(message, "human")
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                )
                with st.chat_message("ai"):
                    chain.invoke(message)
            else:
                st.warning("OPENAI_API_KEY가 잘못되었습니다. 다시 넣어주세요.")
                message = "OPENAI_API_KEY가 잘못되었습니다. 다시 넣어주세요."
                send_message(message, "ai")

    else:
        st.session_state["messages"] = []
