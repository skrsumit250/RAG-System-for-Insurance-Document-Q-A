import streamlit as st
from src.retrieval import main

st.set_page_config(page_title="Insurance Chatbot", page_icon="", layout="centered")

st.title("Insurance Policy Chatbot")
st.caption("I can answer questions about your insurance documents.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you with your insurance documents today?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your policy"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching documents and generating answer..."):
        response = main.ask_rag_pipeline(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
























