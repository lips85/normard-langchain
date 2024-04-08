import re
import os
import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler


# 클라우드페어 공식문서 사이트맵?
# https://developers.cloudflare.com/sitemap.xml
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_check" not in st.session_state:
    st.session_state["api_key_check"] = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "선택해주세요"

API_KEY_pattern = r"sk-.*"

Model_pattern = r"gpt-*"

openai_models = ["선택해주세요", "gpt-4-0125-preview", "gpt-3.5-turbo-0125"]


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


llm = ChatOpenAI(
    temperature=0.1,
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
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
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


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


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
        placeholder="https://example.com",
    )

    st.write(
        """
             

        Made by hary.
             
        Github
        https://github.com/lips85/normard-langchain/blob/main/pages/04_SiteGPT.py

        streamlit
        https://nomad-fullstack-langchain-hary.streamlit.app/

        """
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


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
