import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import textwrap
import streamlit as st
from src import helper as lch

st.set_page_config(page_title="LLM Powered Assistants", page_icon="🤖", layout="wide")
st.title("🤖 LLM Powered Assistants")
st.caption("Powered by Google Gemini 2.5 Flash & LangChain")

# ── Sidebar: Pet Name Generator ───────────────────────────────────────────────
with st.sidebar:
    st.header("🐾 Pet Name Generator")
    user_animal_type = st.selectbox(
        "What is your pet?",
        ("cat", "dog", "parrot", "cow", "hamster"),
        key="pet_type",
    )
    user_pet_color = st.text_area(
        f"What is the color of your {user_animal_type}?",
        max_chars=15,
        key="pet_color",
    )
    generate_name_button = st.button("Generate Pet Name")

if generate_name_button and user_animal_type and user_pet_color:
    with st.spinner("Generating pet name..."):
        response = lch.generate_pet_name(user_animal_type, user_pet_color)
        st.sidebar.success(response)

# ── YouTube Video Q&A Assistant ───────────────────────────────────────────────
st.header("🎬 YouTube Video Q&A Assistant")
st.markdown(
    "Paste a YouTube URL and ask any question. The app downloads the transcript, "
    "builds a vector database, and answers using only the video content."
)

with st.form(key="youtube_qa_form"):
    youtube_url = st.text_input(
        label="YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url_input",
    )
    query = st.text_area(
        label="Your question about the video",
        max_chars=200,
        placeholder="What is the main topic discussed in this video?",
        key="youtube_query_input",
    )
    submit_youtube_qa = st.form_submit_button(label="Get Answer")

if submit_youtube_qa and youtube_url and query:
    with st.spinner("Fetching transcript and generating answer..."):
        response = lch.get_youtube_qa_response(youtube_url, query)
        st.session_state.youtube_qa_response = response

if st.session_state.get("youtube_qa_response"):
    st.subheader("Answer")
    st.write(textwrap.fill(st.session_state.youtube_qa_response, width=100))

st.divider()

# ── General Purpose LangChain Agent (Wikipedia) ───────────────────────────────
st.header("🔍 General Purpose Agent (Wikipedia)")
st.markdown("Ask the ReAct agent anything — it will search Wikipedia to find the answer.")

with st.form(key="agent_form"):
    agent_query = st.text_area(
        label="Your question",
        max_chars=200,
        placeholder="What is the capital of France?",
        key="agent_query_input",
    )
    run_agent_button = st.form_submit_button(label="Run Agent")

if run_agent_button and agent_query:
    with st.spinner("Agent is thinking..."):
        agent_output = lch.langchain_agent(agent_query)
        st.text_area("Agent's Answer", value=agent_output, height=150, disabled=True)
