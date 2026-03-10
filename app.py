import streamlit as st
from rag_pipeline import ask_question, generate_question, evaluate_answer

st.title("AI Interview Preparation Assistant")

# Sidebar mode selector
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Chat Mode", "Quiz Mode"]
)

st.sidebar.header("Upload Documents")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    save_path = f"data/{uploaded_file.name}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    from document_loader import add_document

    add_document(save_path)

    st.sidebar.success("Document added to knowledge base!")

# ---------------- CHAT MODE ----------------

if mode == "Chat Mode":

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask something")

    if st.button("Send") and query:
        
        

        import time

        answer, docs = ask_question(query, st.session_state.chat_history)

        # Streaming output
        message_placeholder = st.empty()
        full_response = ""

        for word in answer.split():
            full_response += word + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response)

        st.session_state.chat_history.append(
            {"user": query, "bot": full_response}
        )

    # Display conversation
    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["user"])

        with st.chat_message("assistant"):
            st.write(chat["bot"])


# ---------------- QUIZ MODE ----------------

elif mode == "Quiz Mode":

    if "question" not in st.session_state:

        question, context = generate_question()

        st.session_state.question = question
        st.session_state.context = context

    st.write("### Question")
    st.write(st.session_state.question)

    user_answer = st.text_area("Your Answer")

    if st.button("Submit Answer"):

        result = evaluate_answer(
            st.session_state.question,
            user_answer,
            st.session_state.context
        )

        st.write("### Evaluation")
        st.write(result)

    if st.button("Next Question"):

        question, context = generate_question()

        st.session_state.question = question
        st.session_state.context = context