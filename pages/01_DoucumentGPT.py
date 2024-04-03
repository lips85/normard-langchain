import time
import streamlit as st

from langchain.chat_models.ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

# 시험 결과 embedding, llm 성능 모두 openai가 더 좋았음
# 또한 ollama는 openai보다 더 느림
# ollama는 embedding이 매우 잘못되어 있음... 이유를 모르겠음... (제대로 확인해 볼 문제... 내가 설정하는 과정이 문제일 수 있음)

LLM_model, models = ["openai", "GPT-3.5-turbo"]
# LLM_model, models = ["ollama", "openhermes:latest"]

file_name = "document.txt"

llm = (
    ChatOllama(temperature=0.1, model=models)
    if LLM_model == "ollama"
    else ChatOpenAI(temperature=0.1)
)

loader = UnstructuredFileLoader(f"./files/{file_name}")
cache_dir = LocalFileStore(f"./.cache/embeddings/{LLM_model}/{models}/{file_name}")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    separators=["\n\n", ".", "?", "!"],
    chunk_size=1000,
    chunk_overlap=100,
)

docs = loader.load_and_split(text_splitter=splitter)
embeddings = (
    OllamaEmbeddings(model=models) if LLM_model == "ollama" else OpenAIEmbeddings()
)

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = FAISS.from_documents(docs, cached_embeddings)

retriever = vectorstore.as_retriever()

memory = ConversationBufferMemory(
    llm=llm,
    return_messages=True,
    memory_key="history",
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI that reads documents for you. Please answer based on the document given below. 
            If the information is not in the document, answer the question with “The required information is not in the document.” Never make up answers. \n\n{context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "history": RunnableLambda(load_memory),
    }
    | prompt
    | llm
)


def invoke_chain(question):
    result = chain.invoke(question).content
    memory.save_context(
        {"input": question},
        {"output": result},
    )
    print(result)


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_bool" not in st.session_state:
    st.session_state["api_key_bool"] = False


st.title("DocumentGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,
    )


message = st.chat_input("Send a message to the ai ")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)


def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_bool"] = True


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

with st.sidebar:
    api_key = st.text_input(
        "OPENAI_API_KEY를 넣어야 작동합니다.",
        disabled=st.session_state["api_key"] is not None,
    ).strip()

    if api_key:
        save_api_key(api_key)
        st.write("API_KEY가 저장되었습니다.")

    button = st.button("저장")

    if button:
        save_api_key(api_key)
        if api_key == "":
            st.write("API_KEY를 넣어주세요.")


with st.sidebar:
    st.write("Made by harry.")

    st.write(
        "https://github.com/lips85/normard-langchain/blob/main/pages/01_DoucumentGPT.py"
    )
