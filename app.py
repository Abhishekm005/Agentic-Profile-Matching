"""Clean Streamlit chat interface for the conversational LangGraph agent."""
from __future__ import annotations

import streamlit as st

from matching_agent import MatchingAgent

st.set_page_config(page_title="Agentic Profile Matching", page_icon="🤖", layout="centered")
st.title("Agentic Profile Matching")
st.caption("AI-assisted candidate search, ranking, comparison, and recruiter refinement")


@st.cache_resource(show_spinner=False)
def get_agent() -> MatchingAgent:
    return MatchingAgent()


if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"messages": [], "screening_round": 1}

agent = get_agent()

with st.sidebar:
    st.subheader("What I can do")
    st.markdown(
        "Search and rank candidates, compare matches, explain ranking decisions, "
        "refine requirements, and generate interview questions."
    )
    st.divider()
    st.subheader("Try an example")
    examples = [
        "Find candidates with Python and 3+ years experience",
        "Compare the top 3 candidates",
        "Why did candidate #1 rank higher than #2?",
        "Add AWS as a must-have",
        "Exclude Java",
        "Generate interview questions for candidate #1",
    ]
    for example in examples:
        st.caption(example)

    st.divider()
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.agent_state = {"messages": [], "screening_round": 1}
        st.rerun()

for message in st.session_state.agent_state.get("messages", []):
    role = "user" if message.__class__.__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(str(message.content))

prompt = st.chat_input("Ask about candidates or refine the requirements...")
if prompt:
    try:
        with st.spinner("Analyzing candidates..."):
            st.session_state.agent_state = agent.invoke(prompt, st.session_state.agent_state)
    except Exception:
        st.error("I couldn't complete that request. Please check the terminal for technical details.")
    else:
        response = st.session_state.agent_state.get("report", "I couldn't generate a response.")
        with st.chat_message("assistant"):
            st.markdown(response)
