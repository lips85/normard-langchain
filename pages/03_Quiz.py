import re
import os
import json
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="QuizGPT❓❗️",
    page_icon="❓",
)

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_check" not in st.session_state:
    st.session_state["api_key_check"] = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "선택해주세요"

if "quiz_subject" not in st.session_state:
    st.session_state["quiz_subject"] = ""

if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False


API_KEY_pattern = r"sk-.*"

Model_pattern = r"gpt-*"

openai_models = ["선택해주세요", "gpt-4-0125-preview", "gpt-3.5-turbo-0125"]

st.title("QuizGPT❓❗️")

st.markdown(
    """
    안녕하세요! 이 페이지는 문서를 읽어주는 AI입니다.😄 
    
    문서를 업로드하고 질문을 하면 문서에 대한 답변을 해줍니다.
    """
)


def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_check"] = True


def save_openai_model(openai_model):
    st.session_state["openai_model"] = openai_model
    st.session_state["openai_model_check"] = True


def set_quiz_submitted(value: bool):
    st.session_state.update({"quiz_submitted": value})


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

    openai_model = st.selectbox(
        "OpneAI Model을 골라주세요.",
        options=openai_models,
    )
    if openai_model != "선택해주세요":
        if re.match(Model_pattern, openai_model):
            save_openai_model(openai_model)
            st.write("😄모델이 선택되었습니다.😄")

    # print(api_key)

    st.divider()
    st.markdown(
        """
                     

        Made by hary.
             
        Github
        https://github.com/lips85/normard-langchain/blob/main/Quiz.py

        streamlit
        https://nomad-langchain-quiz-hary.streamlit.app/

        """
    )

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}
# ChatOpenAI model 정보
# - https://platform.openai.com/docs/models/gpt-3-5-turbo
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-3.5-turbo-0125",
    temperature=0.1,
).bind(
    function_call="auto",
    functions=[
        function,
    ],
)

prompt = PromptTemplate.from_template(
    """            
    Please create a quiz based on the following criteria:

    Topic: {subject}
    Number of Questions: {count}
    Difficulty Level: Level-{difficulty}/5
    Language: Korean

    The quiz should be well-structured with clear questions and correct answers.
    Ensure that the questions are relevant to the specified topic and adhere to the selected difficulty level.
    The quiz format should be multiple-choice,
    and each question should be accompanied by four possible answers, with only one correct option.
    """,
)


if st.session_state["api_key_check"] == False:
    st.warning("Please provide an :blue[OpenAI API Key] on the sidebar.")

else:
    try:

        @st.cache_data(show_spinner="퀴즈 나옵니다...")
        def run_quiz_chain(*, subject, count, difficulty):
            chain = prompt | llm
            return chain.invoke(
                {
                    "subject": subject,
                    "count": count,
                    "difficulty": difficulty,
                }
            )

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                """
            #### 자~ 이제 퀴즈를 만들어 볼까요?
            """
            )
        with col2:

            def reset_quiz():
                st.session_state["quiz_subject"] = ""
                run_quiz_chain.clear()

            # 제대로 동작하지 않음. => 수정 필요
            if st.button(":red[퀴즈 초기화]"):
                reset_quiz()
                set_quiz_submitted(False)

        with st.form("quiz_create_form"):

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                quiz_subject = st.text_input(
                    ":blue[주제]",
                    placeholder="무엇을 주제로 퀴즈를 만들까요?",
                    value=st.session_state["quiz_subject"],
                    # label_visibility="collapsed",
                )

            with col2:
                quiz_count = st.number_input(
                    ":blue[개수]",
                    placeholder="개수",
                    value=10,
                    min_value=2,
                    # label_visibility="collapsed",
                )

            with col3:
                quiz_difficulty = st.selectbox(
                    ":blue[레벨]",
                    ["1", "2", "3", "4", "5"],
                    # label_visibility="collapsed",
                )

            st.form_submit_button(
                "**:blue[퀴즈 만들기 시작]**",
                use_container_width=True,
                on_click=set_quiz_submitted,
                args=(False,),
            )

        if quiz_subject:
            response_box = st.empty()
            response = run_quiz_chain(
                subject=quiz_subject,
                count=quiz_count,
                difficulty=quiz_difficulty,
            )
            response = response.additional_kwargs["function_call"]["arguments"]
            response = json.loads(response)

            generated_quiz_count = len(response["questions"])

            with st.form("quiz_questions_form"):
                solved_count = 0
                correct_count = 0
                answer_feedback_box = []
                answer_feedback_content = []

                for index, question in enumerate(response["questions"]):
                    st.write(f"{index+1}. {question['question']}")
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                        label_visibility="collapsed",
                        key=f"[{quiz_subject}_{quiz_count}_{quiz_difficulty}]question_{index}",
                    )

                    answer_feedback = st.empty()
                    answer_feedback_box.append(answer_feedback)

                    if value:
                        solved_count += 1

                        if {"answer": value, "correct": True} in question["answers"]:
                            answer_feedback_content.append(
                                {
                                    "index": index,
                                    "correct": True,
                                    "feedback": "정답! :100:",
                                }
                            )
                            # st.success("정답! :100:")
                            correct_count += 1
                        else:
                            # st.error("다시 도전해 보아요! :sparkles:")
                            answer_feedback_content.append(
                                {
                                    "index": index,
                                    "correct": False,
                                    "feedback": "다시 도전해 보아요! :sparkles:",
                                }
                            )
                            # answer_feedback_content[index] = st.error(
                            #     "다시 도전해 보아요! :sparkles:"
                            # )

                is_quiz_all_submitted = solved_count == generated_quiz_count

                if is_quiz_all_submitted:
                    for answer_feedback in answer_feedback_content:
                        index = answer_feedback["index"]
                        with answer_feedback_box[index]:
                            if answer_feedback["correct"]:
                                st.success(answer_feedback["feedback"])
                            else:
                                st.error(answer_feedback["feedback"])

                st.divider()

                result = st.empty()

                st.form_submit_button(
                    (
                        "**:blue[제출하기]**"
                        if solved_count < generated_quiz_count
                        else (
                            "**:blue[:100: 축하합니다~ 새로운 주제로 도전해 보세요!]**"
                            if correct_count == generated_quiz_count
                            else "**:blue[다시 도전 💪]**"
                        )
                    ),
                    use_container_width=True,
                    disabled=correct_count == generated_quiz_count,
                    # on_click=lambda: setattr(st.session_state, "submitted", True), 동일함.
                    on_click=set_quiz_submitted,
                    args=(True,),
                )

                if st.session_state["quiz_submitted"]:

                    if not is_quiz_all_submitted:
                        result.error(
                            f"퀴즈를 모두 풀고 제출해 주세요. ( 남은 퀴즈 개수: :red[{generated_quiz_count - solved_count}] / 답변한 퀴즈 개수: :blue[{solved_count}] )"
                        )
                    else:
                        result.subheader(
                            f"결과: :blue[{correct_count}] / {generated_quiz_count}"
                        )

                    if correct_count == generated_quiz_count:
                        for _ in range(3):
                            st.balloons()

    except Exception as e:
        if (
            "api_key" in str(e)
            or "api-key" in str(e)
            or "API key" in str(e)
            or "API Key" in str(e)
        ):
            st.error("API_KEY 를 확인해 주세요.")
        st.expander("Error Details", expanded=True).write(f"Error: {e}")

        if "response" in locals():
            response_box.json(response)
