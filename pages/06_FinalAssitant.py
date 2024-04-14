
# (EN)
# Refactor the agent you made in the previous assignment into an OpenAI Assistant.
# Give it a user interface with Streamlit that displays the conversation history.
# Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
# Using st.sidebar put a link to the Github repo with the code of your Streamlit app.

# (KR)
# 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
# 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.

from datetime import datetime
import re
import os
import json
import streamlit as st
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.document_loaders.web_base import WebBaseLoader
import openai as client


# 클라우드페어 공식문서 사이트맵?
# https://developers.cloudflare.com/sitemap.xml


st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
    # 🚀 리서치 마스터  🚀 
    
    검색은 저에게 맡겨주세요! 여러분들의 시간을 아껴드리겠습니다.
    (OpenAI Assistant APi 사용)
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

if "assistant" not in st.session_state:
    st.session_state["assistant"] = ""

API_KEY_pattern = r"sk-.*"

Model_pattern = r"gpt-*"

openai_models = ["선택해주세요", "gpt-3.5-turbo-0125", "gpt-4-0125-preview"]


class api_key_and_model:
    def __init__(self):
        pass

    def save_api_key(self,api_key):
        st.session_state["api_key"] = api_key
        st.session_state["api_key_check"] = True


    def save_openai_model(self,openai_model):
        st.session_state["openai_model"] = openai_model
        st.session_state["openai_model_check"] = True


class chat_message:
    def __init__(self):
        pass

    def save_message(message, role):
        st.session_state["messages"].append({"message": message, "role": role})


    def send_message(self,message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)

        if save:
            self.save_message(message, role)


    def paint_history(self):
        for message in st.session_state["messages"]:
            self.send_message(
                message["message"],
                message["role"],
                save=False,
            )

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outpus,
    )


with st.sidebar:
    api_key = st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        disabled=st.session_state["api_key"] is not None,
    ).strip()

    if api_key:
        api_key_and_model.save_api_key(api_key)
        st.write("😄API_KEY가 저장되었습니다.😄")

    button = st.button("저장")

    if button:
        api_key_and_model.save_api_key(api_key)
        if api_key == "":
            st.warning("OPENAI_API_KEY를 넣어주세요.")

    st.divider()

    openai_model = st.selectbox(
        "OpneAI Model을 골라주세요.",
        options=openai_models,
    )
    if openai_model != "선택해주세요":
        if re.match(Model_pattern, openai_model):
            api_key_and_model.save_openai_model(openai_model)
            st.write("😄모델이 선택되었습니다.😄")

    st.write(
        """


        Made by hary.

        Github
        https://github.com/lips85/normard-langchain/blob/main/pages/06_FinalAssitant.py

        streamlit
        https://nomad-fullstack-langchain-hary.streamlit.app/FinalAssitant

        """
    )

def get_websites_by_wikipedia_search(inputs):
    w = WikipediaAPIWrapper()
    query = inputs["query"]
    return w.run(query)


def get_websites_by_duckduckgo_search(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    query = inputs["query"]
    return ddg.run(query)


def get_document_text(inputs):
    url = inputs["url"]
    loader = WebBaseLoader([url])
    docs = loader.load()
    return docs[0].page_content

functions_map = {
    "get_websites_by_wikipedia_search": get_websites_by_wikipedia_search,
    "get_websites_by_duckduckgo_search": get_websites_by_duckduckgo_search,
    "get_document_text": get_document_text,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_websites_by_wikipedia_search",
            "description": "Use this tool to find the websites for the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for. Example query: Research about the XZ backdoor",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_websites_by_duckduckgo_search",
            "description": "Use this tool to find the websites for the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for. Example query: Research about the XZ backdoor",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_text",
            "description": "Use this tool to load the website for the given url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url you will load. Example url: https://en.wikipedia.org/wiki/Backdoor_(computing)",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

if not api_key:
    st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

if openai_model == "선택해주세요":
    st.warning("Please write down a **:blue[OpenAI Model Select]** on the sidebar.")

@st.cache_resource(show_spinner="Loading...")
def create_thread(message_content):
    return client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ]
    )

def create_run(thread_id, assistant_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )



if api_key and openai_model != "선택해주세요":
    if st.session_state["assistant"] == "": 
        C_A = st.button("Create Assistant")
        if C_A:
            st.session_state["assistant"] = client.beta.assistants.create(
                name="Super Search Assistant",
                instructions="""
                0. 당신은 user의 Research Assistant 입니다.
                1. query 에 대해서 검색하고
                2. 검색 결과 목록에 website url 목록이 있으면, 각각의 website 내용을 text로 추출해줘.  

                """,
                model=st.session_state["openai_model"],
                tools=functions,
            )

            st.write(st.session_state["assistant"])

        message_content = "Research about the XZ backdoor"
        R_A = st.button("Reset Assistant")
        if R_A:
            st.session_state["assistant"] = ""
            st.session_state["messages"] = []





# if (st.session_state["api_key_check"] is True) and (
#     st.session_state["api_key"] is not None
# ):
#     if url:
#         if ".xml" not in url:
#             with st.sidebar:
#                 st.error("Please write down a Sitemap URL.")
#         else:
#             retriever = load_website(url)
#             send_message("I'm ready! Ask away!", "ai", save=False)
#             paint_history()
#             message = st.chat_input("Ask a question to the website.")
#             if message:
#                 if re.match(API_KEY_pattern, st.session_state["api_key"]) and re.match(
#                     Model_pattern, st.session_state["openai_model"]
#                 ):
#                     send_message(message, "human")
#                     try:
#                         chain = (
#                             {
#                                 "docs": retriever,
#                                 "question": RunnablePassthrough(),
#                                 "history": RunnableLambda(load_memory),
#                             }
#                             | RunnableLambda(get_answers)
#                             | RunnableLambda(choose_answer)
#                         )

#                         def invoke_chain(question):
#                             result = chain.invoke(question).content
#                             memory.save_context(
#                                 {"input": question},
#                                 {"output": result},
#                             )
#                             return result

#                         with st.chat_message("ai"):
#                             invoke_chain(message)

#                     except Exception as e:
#                         st.error(f"An error occurred: {e}")
#                         st.warning("OPENAI_API_KEY or 모델 선택을 다시 진행해주세요.")

#                 else:
#                     message = "OPENAI_API_KEY or 모델 선택이 잘못되었습니다. 사이드바를 다시 확인하세요."
#                     send_message(message, "ai")
#     else:
#         st.session_state["messages"] = []
