import os
import functools
import json
from datetime import datetime
import streamlit as st
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from openai import OpenAI, AuthenticationError
from common.time_logger import time_logger


@time_logger
def main():
    st.set_page_config(
        page_title="Assistant | D31 과제",
        page_icon="👼",
    )

    st.title("D33 | OpenAI Assistants (졸업 과제)")
    with st.expander("과제 내용 보기", expanded=False):
        # st.snow()
        st.markdown(
            """
        ### D31 (2024-04-10) 과제
        - 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
        - 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
        - 유저가 자체 OpenAI API 키를 사용하도록 허용하고,`st.sidebar` 내부의 `st.input`에서 이를 로드합니다.
        - `st.sidebar`를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.
        - 예시 요청
            - Research about the XZ backdoor
        
        """
        )

    with st.sidebar:
        if "api_key" not in st.session_state:
            st.session_state["api_key"] = ""

        api_key_input = st.empty()

        def reset_api_key():
            st.session_state["api_key"] = ""

        if st.button(":red[Reset API_KEY]"):
            reset_api_key()

        api_key = api_key_input.text_input(
            "**:blue[OpenAI API_KEY]**",
            value=st.session_state["api_key"],
            key="api_key_input",
        )

        if api_key != st.session_state["api_key"]:
            st.session_state["api_key"] = api_key
            st.rerun()

        st.divider()
        st.markdown(
            """
            GitHub 링크: https://github.com/LifeFi/py_w08_fullstack_gpt_d15/blob/d33_final_assistant/pages/D33_OpenAI_Assistants.py
            """
        )

        log_box = st.expander(":blue[Assistant API Log]", expanded=True)

        if "logs" not in st.session_state:
            st.session_state["logs"] = []

        if log_box.button("Clear Log"):
            st.session_state["logs"] = []

        def paint_logs():
            for log in st.session_state["logs"]:
                log_box.write(log)

        def add_log(*args, seperator=" | "):
            log_message = seperator.join(str(arg) for arg in args)
            log = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}{seperator}{log_message}"
            st.session_state["logs"].append(log)
            log_box.write(log)

        paint_logs()

    def assistant_api_logger(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if func.__name__ in ["create_run", "get_run", "get_runs_poll"]:
                add_log(f":blue[{func.__name__}]", f":red[{result.status}]", result.id)
            else:
                add_log(f":blue[{func.__name__}]", result.id)
            return result

        return wrapper

    if not api_key:
        st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")
        st.stop()

    client = OpenAI(api_key=api_key)

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

    @assistant_api_logger
    @st.cache_data
    def create_assistant():
        return client.beta.assistants.create(
            name="Research Assistant",
            instructions="""
            0. 당신은 user의 Research Assistant 입니다.
            1. query 에 대해서 검색하고
            2. 검색 결과 목록에 website url 목록이 있으면, 각각의 website 내용을 text로 추출해줘.  

            """,
            model="gpt-4-1106-preview",
            tools=functions,
        )

    @assistant_api_logger
    @st.cache_data
    def create_thread(message_content):
        return client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": message_content,
                }
            ]
        )

    @assistant_api_logger
    @st.cache_data
    def create_run(thread_id, assistant_id):
        return client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

    with st.sidebar:
        if st.button("Recreate Assistant", use_container_width=True):
            create_assistant.clear()

        if st.button("Recreate Thread", use_container_width=True):
            create_thread.clear()

        if st.button("Recreate Run", use_container_width=True):
            create_run.clear()

    @assistant_api_logger
    def get_run(run_id, thread_id):
        return client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )

    @assistant_api_logger
    def get_runs_poll(run_id, thread_id):
        return client.beta.threads.runs.poll(
            run_id=run_id,
            thread_id=thread_id,
            poll_interval_ms=500,
            timeout=20,
        )

    def send_message(thread_id, content):
        return client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

    def get_messages(thread_id, assistant_only=True):
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        messages = list(messages)
        messages.reverse()
        # print(messages)
        result = ""
        for message in messages:
            if assistant_only and message.role != "assistant":
                continue
            result = result + f"\n\n{message.content[0].text.value}"
            # print(f"{message.role}: {message.content[0].text.value}")
        return result

    def get_tool_outputs(run_id, thread_id):
        run = get_run(run_id, thread_id)
        # st.write(run.required_action)
        outputs = []
        for action in run.required_action.submit_tool_outputs.tool_calls:
            # st.write(action)
            action_id = action.id
            function = action.function
            print(f"Calling function: {function.name} with arg {function.arguments}")
            outputs.append(
                {
                    "output": functions_map[function.name](
                        json.loads(function.arguments)
                    ),
                    "tool_call_id": action_id,
                }
            )
        return outputs

    def submit_tool_outputs(run_id, thread_id):
        outputs = get_tool_outputs(run_id, thread_id)
        return client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=outputs,
        )

    def make_download_button(text, file_path):
        research_dt = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        os.makedirs("./outputs", exist_ok=True)
        file_name = f"./outputs/{research_dt}_{file_path}"

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(text)

        file_bytes = open(file_name, "rb").read()

        return st.download_button(
            label="Download file",
            data=file_bytes,
            file_name=f"{research_dt}_{file_path}",
            mime="text/plain",
            key=f"{research_dt}_{file_path}",
            type="primary",
        )
        # return f"Text saved to {research_dt}_{file_path}"

    def save_chat_message(message, role):
        st.session_state["messages"].append({"message": message, "role": role})

    def send_chat_message(message, role, save=True, download=True):
        with st.chat_message(role):
            st.markdown(message)
        if save:
            save_chat_message(message, role)

    def paint_chat_history():
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        for index, message in enumerate(st.session_state["messages"]):
            send_chat_message(
                message["message"],
                message["role"],
                save=False,
                download=False,
            )
            if message["role"] == "assistant":
                download_placholder = st.empty()
                with download_placholder:
                    make_button = st.button(
                        "Make Download", key=f"make_download_{index}"
                    )

                if make_button:
                    download_placholder.empty()
                    with download_placholder:
                        make_download_button(
                            text=message["message"],
                            file_path=f"{index}_research.txt",
                        )

    try:
        assistant = create_assistant()

        if "message" not in st.session_state:
            st.session_state["message"] = ""

        send_chat_message(
            "I'm ready! Ask away!", "assistant", save=False, download=False
        )
        paint_chat_history()

        if message := st.chat_input(
            "What do you want reaearch about?", key="message_input"
        ):
            st.session_state["message"] = message

            send_chat_message(message, "user")

            thread = create_thread(message)

            # create 시에는 status 가 무조건 queued 로 시작된다고 함(run.status == "queued")
            run = create_run(thread.id, assistant.id)

            is_new_result = False

            with st.chat_message("assistant"):
                with st.status(":blue[Polling Run Status...]") as status:
                    # st.write(run.id, " ===== ", run.status)

                    while True:
                        run = get_runs_poll(run.id, thread.id)

                        # run_status = ( "queued", "in_progress", "completed", "requires_action", "expired", "cancelling", "cancelled", "failed" )
                        # waiting_status = ("queued", "in_progress", "cancelling")
                        if run.status == "requires_action":
                            status.update(
                                label=f"Running: {run.status}", state="running"
                            )
                            submit_tool_outputs(run.id, thread.id)

                        if run.status in ("expired", "cancelled", "failed"):
                            st.write(run.last_error)
                            status.update(
                                label=f":red[{run.status}]",
                                state="error",
                                expanded=True,
                            )
                            break

                        if run.status == "completed":
                            is_new_result = True
                            status.update(label=run.status, state="complete")
                            break

            if is_new_result:
                result = get_messages(thread.id)
                send_chat_message(result, "assistant")
                # last message도 paint_chat_history()로 렌더링 ( 다운로드 버튼 생성 쉽게 하기 위해 )
                st.rerun()

    except AuthenticationError as e:
        st.error(f"Authentication 오류: API_KEY 를 확인해 주세요.")

    except Exception as e:
        if "no assistant" in str(e).lower():
            create_assistant.clear()
            st.rerun()
        else:
            st.error(f"Error: {e}")


main()
