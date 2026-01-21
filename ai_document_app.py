"""
AI Document Copilot - A Streamlit app for chatting with your documents using RAG.

This application allows users to upload documents (TXT, PDF, PPTX) and ask questions
about their content using Google's Gemini AI model for natural language responses.
"""

# Standard library imports
import os
import tempfile

# Third-party imports
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Streamlit page
st.set_page_config(
    page_title="AI Document Copilot",
    page_icon="🤖",
    layout="wide"
)

# App title and description
st.title("🤖 AI Document Copilot")
st.markdown("Upload your documents and chat with your AI assistant about their content!")

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

# Sidebar: Document Upload and Management
with st.sidebar:
    st.header("📁 Document Upload")

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['txt', 'pdf', 'pptx'],
        accept_multiple_files=True,
        help="Upload text, PDF, or PowerPoint files"
    )

    # Update uploaded files list
    if uploaded_files:
        st.session_state.uploaded_files_list = [{"name": f.name, "size": f.size} for f in uploaded_files]

    # Display uploaded files
    if st.session_state.uploaded_files_list:
        st.subheader("Uploaded Files:")
        for file_info in st.session_state.uploaded_files_list:
            st.markdown(f"- **{file_info['name']}** ({file_info['size']} bytes)")

    # Process button
    if uploaded_files and not st.session_state.documents_processed:
        if st.button("🚀 Process Documents"):
            with st.spinner("Processing documents..."):
                # Import heavy libraries only when needed
                from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma

                all_docs = []

                for uploaded_file in uploaded_files:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        # Load document based on file type
                        if uploaded_file.name.endswith('.txt'):
                            loader = TextLoader(tmp_path, encoding='utf-8')
                        elif uploaded_file.name.endswith('.pdf'):
                            loader = PyPDFLoader(tmp_path)
                        elif uploaded_file.name.endswith('.pptx'):
                            loader = UnstructuredPowerPointLoader(tmp_path)
                        else:
                            st.error(f"Unsupported file type: {uploaded_file.name}")
                            continue

                        docs = loader.load()
                        all_docs.extend(docs)

                        # Clean up temporary file
                        os.unlink(tmp_path)

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue

                if all_docs:
                    # Split text
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(all_docs)

                    # Create embeddings and vector store
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.vector_store = Chroma.from_documents(splits, embeddings)

                    st.session_state.documents_processed = True
                    st.success(f"✅ Successfully processed {len(all_docs)} documents into {len(splits)} chunks!")

    # Clear documents button
    if st.session_state.documents_processed:
        if st.button("🗑️ Clear Documents"):
            st.session_state.vector_store = None
            st.session_state.documents_processed = False
            st.session_state.chat_history = []
            st.session_state.uploaded_files_list = []
            st.rerun()

# Main area: Chat Interface
if st.session_state.documents_processed and st.session_state.vector_store:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
                if "chunks" in message:
                    with st.expander("📄 Relevant Document Chunks"):
                        for i, chunk in enumerate(message["chunks"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                            st.markdown("---")
            else:
                st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Ask about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Process query
        with st.spinner("Thinking..."):
            # Import heavy libraries only when needed
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Set up retriever
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

            # Set up LLM
            llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

            # Create prompt template
            template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
            prompt = ChatPromptTemplate.from_template(template)

            # Create chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Get answer
            answer = chain.invoke(query)

            # Get relevant chunks
            docs = retriever.invoke(query)
            chunks = [doc.page_content for doc in docs]

            # Add assistant message to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "chunks": chunks})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("📄 Relevant Document Chunks"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    st.markdown("---")

else:
    if not st.session_state.documents_processed:
        st.info("👆 Please upload and process some documents in the sidebar to start chatting!")
    else:
        st.error("No documents processed. Please upload files again.")

