import re, os
import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.memory.buffer import ConversationBufferMemory

# 클라우드페어 공식문서 사이트맵?
# https://developers.cloudflare.com/sitemap.xml


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
    layout="wide",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_check" not in st.session_state:
    st.session_state["api_key_check"] = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "선택해주세요"

API_KEY_pattern = r"sk-.*"

Model_pattern = r"gpt-*"

openai_models = ["선택해주세요", "gpt-3.5-turbo-0125", "gpt-4-0125-preview"]


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_check"] = True


def save_openai_model(openai_model):
    st.session_state["openai_model"] = openai_model
    st.session_state["openai_model_check"] = True


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


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
    
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    history = inputs["history"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "history": history,
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    history = inputs["history"]
    choose_chain = choose_prompt | llm_for_last
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "history": history,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


def load_memory(_):
    return memory.load_memory_variables({})["history"]


llm_for_last = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks={
        ChatCallbackHandler(),
    },
    model=st.session_state["openai_model"],
    openai_api_key=st.session_state["api_key"],
)
llm = ChatOpenAI(
    temperature=0.1,
    model=st.session_state["openai_model"],
    openai_api_key=st.session_state["api_key"],
)

memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
    memory_key="history",
)


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    os.makedirs("./.cache/sitemap", exist_ok=True)
    cache_dir = LocalFileStore(f"./.cache/sitemap/embeddings/{url_name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"https:\/\/developers.cloudflare.com/ai-gateway.*",
            r"https:\/\/developers.cloudflare.com/vectorize.*",
            r"https:\/\/developers.cloudflare.com/workers-ai.*",
        ],
    )
    loader.requests_per_second = 50
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.session_state["api_key"],
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


with st.sidebar:
    api_key = st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        disabled=st.session_state["api_key"] != None,
    ).strip()

    if api_key:
        save_api_key(api_key)
        st.write("😄API_KEY가 저장되었습니다.😄")

    button = st.button("저장")

    if button:
        save_api_key(api_key)
        if api_key == "":
            st.warning("OPENAI_API_KEY를 넣어주세요.")

    st.divider()

    openai_model = st.selectbox(
        "OpneAI Model을 골라주세요.",
        options=openai_models,
    )
    if openai_model != "선택해주세요":
        if re.match(Model_pattern, openai_model):
            save_openai_model(openai_model)
            st.write("😄모델이 선택되었습니다.😄")

    st.divider()
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com/sitemap.xml",
    )

    url_name = url.split("://")[1].replace("/", "_") if url else None

    st.write(
        """
             

        Made by hary.
             
        Github
        https://github.com/lips85/normard-langchain/blob/main/pages/04_SiteGPT.py

        streamlit
        https://nomad-fullstack-langchain-hary.streamlit.app/

        """
    )


if not api_key:
    st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

if not url:
    st.warning("Please write down a **:blue[Sitemap URL]** on the sidebar.")


if (st.session_state["api_key_check"] == True) and (
    st.session_state["api_key"] != None
):
    if url:
        if ".xml" not in url:
            with st.sidebar:
                st.error("Please write down a Sitemap URL.")
        else:
            retriever = load_website(url)
            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()
            message = st.chat_input("Ask a question to the website.")
            if message:
                if re.match(API_KEY_pattern, st.session_state["api_key"]) and re.match(
                    Model_pattern, st.session_state["openai_model"]
                ):
                    send_message(message, "human")
                    try:
                        chain = (
                            {
                                "docs": retriever,
                                "question": RunnablePassthrough(),
                                "history": RunnableLambda(load_memory),
                            }
                            | RunnableLambda(get_answers)
                            | RunnableLambda(choose_answer)
                        )

                        def invoke_chain(question):
                            result = chain.invoke(question).content
                            memory.save_context(
                                {"input": question},
                                {"output": result},
                            )
                            return result

                        with st.chat_message("ai"):
                            invoke_chain(message)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.warning("OPENAI_API_KEY or 모델 선택을 다시 진행해주세요.")

                else:
                    message = "OPENAI_API_KEY or 모델 선택이 잘못되었습니다. 사이드바를 다시 확인하세요."
                    send_message(message, "ai")
    else:
        st.session_state["messages"] = []
