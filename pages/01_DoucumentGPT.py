import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

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
