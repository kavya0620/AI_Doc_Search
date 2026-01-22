import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader,TextLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda



load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.set_page_config(page_title="AI Document Search", page_icon=" ", layout="wide")

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .chat-user {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .chat-assistant {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .sidebar-header {
        font-size: 1.2em;
        font-weight: bold;
        color: #ff7f0e;
    }
    .success-msg {
        color: #2ca02c;
        font-weight: bold;
    }
    .error-msg {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"> AI Document Copilot</div>', unsafe_allow_html=True)
st.markdown("Upload your documents and chat with them like a modern AI assistant!")

st.divider()

# ----------------- SESSION STATE -----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()

# ----------------- SIDEBAR: FILE UPLOAD AND PROCESSING -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-header"> Document Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    # Display uploaded files list as table
    if uploaded_files:
        st.markdown('<div class="sub-header">📋 Uploaded Files</div>', unsafe_allow_html=True)
        file_data = []
        for file in uploaded_files:
            size_kb = len(file.getvalue()) / 1024
            file_data.append({"Name": file.name, "Size (KB)": f"{size_kb:.1f}"})
        st.table(file_data)

    # Process documents
    if uploaded_files:
        if st.button("🚀 Process Documents"):
            with st.spinner("Processing documents..."):
                temp_dir = tempfile.mkdtemp()

                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                txt_docs = DirectoryLoader(
                    temp_dir, glob="*.txt", loader_cls=TextLoader
                ).load()

                pdf_docs = DirectoryLoader(
                    temp_dir, glob="*.pdf", loader_cls=PyPDFLoader
                ).load()

                all_docs = txt_docs + pdf_docs
                shutil.rmtree(temp_dir)

                if all_docs:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )

                    splits = splitter.split_documents(all_docs)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2"
                    )

                    st.session_state.vector_store = Chroma.from_documents(
                        splits, embeddings
                    )

                    st.session_state.documents_processed = True
                    st.markdown(f'<div class="success-msg">✅ {len(splits)} chunks created successfully!</div>', unsafe_allow_html=True)

    # Clear button
    if st.session_state.documents_processed:
        if st.button("🗑️ Clear Documents"):
            # Clear vector store
            st.session_state.vector_store = None
            # Reset flags
            st.session_state.documents_processed = False
            # Clear chat history
            st.session_state.chat_history = []
            # Reset memory (important!)
            st.session_state.memory = ChatMessageHistory()
            # Force Streamlit to refresh everything
            st.rerun()

# ----------------- CHAT INTERFACE -----------------
st.header("💬 Chat with Your Documents")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat['question'])
    with st.chat_message("assistant"):
        st.markdown(chat['answer'])
        with st.expander("📄 Sources"):
            for i, doc in enumerate(chat["sources"], 1):
                st.markdown(f"**Source {i}:**")
                st.markdown(doc.page_content[:500] + "...")
                st.markdown("---")

# Chat input at the bottom
if st.session_state.documents_processed and st.session_state.vector_store:
    if prompt := st.chat_input("Ask about your documents..."):
        with st.spinner("Searching..."):

            # --------- LLM ---------
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2
            )

            # --------- RETRIEVER ---------
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )

            # --------- PROMPT ---------
            prompt_template = ChatPromptTemplate.from_template(
                """Answer the question using ONLY the context below.
If the answer is not present in the context, say:
"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
            )

            # --------- RAG CHAIN ---------
            qa_chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )

            # --------- OPTION 3: SAFE GEMINI CALL ---------
            try:
                answer = qa_chain.invoke(prompt)
            except Exception as e:
                st.error(
                    "🚫 Gemini API quota exceeded.\n\n"
                    "Please wait 1 minute and try again."
                )
                st.stop()

            # --------- GET SOURCE DOCUMENTS ---------
            docs = retriever.invoke(prompt)

            # --------- SAVE CHAT HISTORY ---------
            st.session_state.chat_history.append({
                "question": prompt,
                "answer": answer,
                "sources": docs
            })

            # Rerun to display the new message
            st.rerun()

