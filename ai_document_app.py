import streamlit as st
st.success("App started")
import os
import tempfile
import jwt
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Table, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import json
import logging
import pytesseract
if os.name == "nt":  # Windows only
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import tabula
import fitz  # PyMuPDF
import cv2

from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
import time
import numpy as np

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration constants
VECTOR_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVER_K = 3
TESSERACT_CONFIG = '--oem 3 --psm 6'
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "default-encryption-key")
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")
LLM_MODEL = "gemini-pro"

LLM_TEMPERATURE = 0.2
ANALYTICS_DB_PATH = "analytics.db"
ROLES = {
    "admin": ["read", "write", "delete", "manage_users"],
    "faculty": ["read", "write"],
    "student": ["read"]
}
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = ["txt", "pdf", "png", "jpg", "jpeg", "pptx", "docx"]

# Set up logging
logger = logging.getLogger(__name__)

# Security Manager Class
class SecurityManager:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.roles = ROLES
        self.users = {
            "admin": {"password": "admin123", "role": "admin"},
            "faculty": {"password": "faculty123", "role": "faculty"},
            "student": {"password": "student123", "role": "student"}
        }

    def authenticate_user(self, username, password):
        if username in self.users and self.users[username]["password"] == password:
            payload = {
                "username": username,
                "role": self.users[username]["role"],
                "exp": datetime.utcnow() + timedelta(hours=1)
            }
            token = jwt.encode(payload, self.secret_key, algorithm="HS256")
            return token
        return None

    def oauth_authenticate(self, token):
        payload = {
            "username": "oauth_user",
            "role": "student",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def get_user_role(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload["role"]
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def logout_user(self, token):
        pass

    def check_permission(self, role, action):
        if role in self.roles:
            return action in self.roles[role]
        return False

    def audit_log(self, action, user, details):
        print(f"AUDIT: {datetime.now()} - {user} - {action} - {details}")

# Analytics Tracker Class
class AnalyticsTracker:
    def __init__(self):
        self.query_history = []
        self.document_access = defaultdict(int)
        self.performance_metrics = []
        self.knowledge_graph = nx.DiGraph()
        self._load_data()

    def _load_data(self):
        try:
            if os.path.exists(ANALYTICS_DB_PATH):
                with open(ANALYTICS_DB_PATH, 'r') as f:
                    data = json.load(f)
                    self.query_history = data.get('query_history', [])
                    self.document_access = defaultdict(int, data.get('document_access', {}))
                    self.performance_metrics = data.get('performance_metrics', [])
        except Exception as e:
            logger.warning(f"Could not load analytics data: {e}")

    def _save_data(self):
        try:
            data = {
                'query_history': self.query_history[-1000:],
                'document_access': dict(self.document_access),
                'performance_metrics': self.performance_metrics[-1000:]
            }
            with open(ANALYTICS_DB_PATH, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Could not save analytics data: {e}")

    def track_query(self, question, answer, sources, response_time, username="anonymous"):
        query_entry = {
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'question': question,
            'answer': answer,
            'sources': [doc.page_content[:200] for doc in sources],
            'response_time': response_time,
            'source_count': len(sources)
        }
        self.query_history.append(query_entry)
        for doc in sources:
            doc_id = doc.metadata.get('source', 'unknown')
            self.document_access[doc_id] += 1
        self._update_knowledge_graph(question, sources)
        if len(self.query_history) % 10 == 0:
            self._save_data()
        logger.info(f"Tracked query: {question[:50]}... (response time: {response_time:.2f}s)")

    def get_most_asked_questions(self, limit=10, days=30):
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_queries = [
            q for q in self.query_history
            if datetime.fromisoformat(q['timestamp']) > cutoff_date
        ]
        question_counts = Counter(q['question'] for q in recent_queries)
        return question_counts.most_common(limit)

    def get_frequently_accessed_documents(self, limit=10):
        return self.document_access.most_common(limit)

    def get_performance_stats(self, days=7):
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.performance_metrics
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        if not recent_metrics:
            return {}
        response_times = [m['value'] for m in recent_metrics if m['metric'] == 'response_time']
        stats = {
            'total_queries': len([m for m in recent_metrics if m['metric'] == 'query_count']),
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0
        }
        return stats

    def _update_knowledge_graph(self, question, sources):
        question_words = set(question.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who'}
        topics = question_words - stop_words
        for topic in topics:
            self.knowledge_graph.add_node(topic, type='topic')
            for doc in sources:
                doc_id = doc.metadata.get('source', 'unknown')
                self.knowledge_graph.add_node(doc_id, type='document')
                self.knowledge_graph.add_edge(topic, doc_id, weight=1.0)

    def suggest_followup_questions(self, current_question, limit=3):
        current_words = set(current_question.lower().split())
        related_questions = []
        for q in self.query_history[-100:]:
            q_words = set(q['question'].lower().split())
            similarity = len(current_words & q_words) / len(current_words | q_words)
            if similarity > 0.3 and q['question'] != current_question:
                related_questions.append((q['question'], similarity))
        related_questions.sort(key=lambda x: x[1], reverse=True)
        return [q[0] for q in related_questions[:limit]]

# Multi-Modal Processor Class
class MultiModalProcessor:
    def __init__(self):
        # Do NOT load heavy models at startup
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None

    def _load_models(self):
        # Load models only when actually needed
        if self.blip_processor is None:
            from transformers import (
                BlipProcessor,
                BlipForConditionalGeneration,
                CLIPProcessor,
                CLIPModel,
            )
            self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL)

    def extract_text_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image, config=TESSERACT_CONFIG).strip()
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""

    def generate_image_caption(self, image_path):
        try:
            self._load_models()  # 👈 models load here, not at startup
            image = Image.open(image_path)
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs)
            return self.blip_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Image captioning failed for {image_path}: {e}")
            return "Image description unavailable"

    def extract_tables_from_pdf(self, pdf_path):
        try:
            tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            return "\n\n".join(table.to_string(index=False) for table in tables)
        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path}: {e}")
            return ""

    def analyze_document_layout(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            layout_info = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("dict").get("blocks", [])

                headers, footers, sections, columns = [], [], [], []

                for block in blocks:
                    if "lines" in block:
                        x0, y0, x1, y1 = block["bbox"]

                        if y0 < 100:
                            headers.append(block)
                        elif y1 > page.rect.height - 100:
                            footers.append(block)
                        else:
                            sections.append(block)

                        columns.append("left" if x0 < page.rect.width / 2 else "right")

                layout_info.append({
                    "page": page_num + 1,
                    "headers": len(headers),
                    "footers": len(footers),
                    "sections": len(sections),
                    "columns": len(set(columns)),
                })

            doc.close()
            return layout_info
        except Exception as e:
            logger.error(f"Layout analysis failed for {pdf_path}: {e}")
            return []

    def process_multi_modal_document(self, file_path, file_type):
        documents = []

        if file_type in ["png", "jpg", "jpeg"]:
            ocr_text = self.extract_text_from_image(file_path)
            caption = self.generate_image_caption(file_path)

            documents.append(
                Document(
                    page_content=f"OCR Text: {ocr_text}\n\nImage Caption: {caption}",
                    metadata={"source": file_path, "type": "image"},
                )
            )

        elif file_type == "pdf":
            ocr_text = self.extract_text_from_image(file_path)
            tables_text = self.extract_tables_from_pdf(file_path)
            layout_info = self.analyze_document_layout(file_path)

            documents.append(
                Document(
                    page_content=(
                        f"OCR Text: {ocr_text}\n\n"
                        f"Extracted Tables:\n{tables_text}\n\n"
                        f"Layout Analysis:\n{layout_info}"
                    ),
                    metadata={"source": file_path, "type": "pdf"},
                )
            )

        return documents

# Advanced Vector Store Class
class AdvancedVectorStore:
    def __init__(self):
        self.client = PersistentClient(path=VECTOR_DB_PATH)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.document_collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )
        self.metadata_collection = self.client.get_or_create_collection(
            name="metadata",
            embedding_function=self.embedding_function
        )
        self.bm25_index = None
        self.document_texts = []
        self.version_history = {}
        self._load_existing_data()

    def _load_existing_data(self):
        try:
            results = self.document_collection.get(include=['documents', 'metadatas'])
            if results['documents']:
                self.document_texts = results['documents']
                self.bm25_index = BM25Okapi([text.split() for text in self.document_texts])
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")

    def add_documents(self, documents):
        if not documents:
            return
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(documents))]
        self.document_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        self.document_texts.extend(texts)
        self.bm25_index = BM25Okapi([text.split() for text in self.document_texts])
        for i, doc in enumerate(documents):
            doc_id = ids[i]
            self.version_history[doc_id] = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": datetime.now().isoformat(),
                "version": 1
            }
        logger.info(f"Added {len(documents)} documents to vector store")

    def hybrid_search(self, query, k=3, semantic_weight=0.7, keyword_weight=0.3):
        semantic_results = self.document_collection.query(
            query_texts=[query],
            n_results=k
        )
        keyword_scores = self.bm25_index.get_scores(query.split())
        keyword_indices = np.argsort(keyword_scores)[-k:][::-1]
        combined_results = []
        for i, doc_id in enumerate(semantic_results['ids'][0]):
            score = semantic_results['distances'][0][i] if semantic_results['distances'] else 0
            combined_results.append({
                'id': doc_id,
                'content': semantic_results['documents'][0][i],
                'metadata': semantic_results['metadatas'][0][i],
                'score': semantic_weight * (1 / (1 + score))
            })
        for idx in keyword_indices:
            if idx < len(self.document_texts):
                content = self.document_texts[idx]
                score = keyword_scores[idx]
                combined_results.append({
                    'id': f"bm25_{idx}",
                    'content': content,
                    'metadata': {},
                    'score': keyword_weight * score
                })
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:k]

    def cluster_documents(self, n_clusters=5):
        try:
            results = self.document_collection.get(include=['embeddings'])
            if not results['embeddings'] or len(results['embeddings']) == 0:
                return []
            embeddings = np.array(results['embeddings'])
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            clustered_docs = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_docs:
                    clustered_docs[cluster_id] = []
                clustered_docs[cluster_id].append({
                    'id': results['ids'][i],
                    'content': results['documents'][i][:200] + "...",
                    'metadata': results['metadatas'][i]
                })
            return clustered_docs
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return []

# Utility functions
def get_advanced_vector_store():
    return AdvancedVectorStore()

def get_analytics_tracker():
    return AnalyticsTracker()

def get_multi_modal_processor():
    return MultiModalProcessor()

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
if 'user_token' not in st.session_state:
    st.session_state.user_token = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = "student"
if 'security_manager' not in st.session_state:
    st.session_state.security_manager = SecurityManager()
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = get_advanced_vector_store()
if 'analytics_tracker' not in st.session_state:
    st.session_state.analytics_tracker = get_analytics_tracker()
if 'multi_modal_processor' not in st.session_state:
    st.session_state.multi_modal_processor = get_multi_modal_processor()
if 'chunk_info' not in st.session_state:
    st.session_state.chunk_info = {}

# ----------------- STREAMLIT CONFIG -----------------
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
    .login-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- AUTHENTICATION -----------------
if not st.session_state.user_authenticated:
    st.markdown('<div class="main-header">🔐 AI Document Copilot</div>', unsafe_allow_html=True)
    st.markdown("Enterprise-grade AI document search with advanced security")

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.subheader("🔑 Login to Access System")

        login_tab1, login_tab2 = st.tabs(["Standard Login", "OAuth Login"])

        with login_tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            st.info("Demo credentials: admin/admin123, faculty/faculty123, student/student123")

            if st.button("Login", key="login_btn"):
                token = st.session_state.security_manager.authenticate_user(username, password)
                if token:
                    st.session_state.user_token = token
                    st.session_state.user_authenticated = True
                    st.session_state.user_role = st.session_state.security_manager.get_user_role(token)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with login_tab2:
            st.write("OAuth authentication would integrate with Google/Microsoft/etc.")
            if st.button("Simulate OAuth Login", key="oauth_btn"):
                token = st.session_state.security_manager.oauth_authenticate("dummy_token")
                if token:
                    st.session_state.user_token = token
                    st.session_state.user_authenticated = True
                    st.session_state.user_role = st.session_state.security_manager.get_user_role(token)
                    st.success("OAuth login successful!")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# Show user info and logout
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header"> AI Document Copilot</div>', unsafe_allow_html=True)
    st.markdown(f"Welcome, {st.session_state.user_role.title()} User! Upload your documents and chat with them like a modern AI assistant!")
with col2:
    if st.button("Logout", key="logout_btn"):
        if hasattr(st.session_state, 'user_token'):
            st.session_state.security_manager.logout_user(st.session_state.user_token)
        st.session_state.user_authenticated = False
        st.session_state.user_token = None
        st.session_state.user_role = "student"
        st.rerun()

st.divider()

# ----------------- SIDEBAR: FILE UPLOAD -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">📁 Document Upload</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, DOCX, PNG, JPG"
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

        # Process uploaded files
        if st.button("Process Documents", key="process_btn"):
            with st.spinner("Processing documents..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    file_path = f"temp_{uploaded_file.name}"

                    # Save uploaded file temporarily
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        # Load document based on type
                        if file_type == 'txt':
                            loader = TextLoader(file_path)
                            docs = loader.load()
                        elif file_type == 'pdf':
                            loader = PyPDFLoader(file_path)
                            docs = loader.load()
                        elif file_type in ['docx', 'pptx']:
                            if file_type == 'docx':
                                loader = UnstructuredWordDocumentLoader(file_path)
                            else:
                                loader = UnstructuredPowerPointLoader(file_path)
                            docs = loader.load()
                        elif file_type in ['png', 'jpg', 'jpeg']:
                            # Use multi-modal processor for images
                            docs = st.session_state.multi_modal_processor.process_multi_modal_document(file_path, file_type)
                        else:
                            st.error(f"Unsupported file type: {file_type}")
                            continue

                        # Chunk the documents
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=CHUNK_SIZE,
                            chunk_overlap=CHUNK_OVERLAP
                        )
                        chunked_docs = text_splitter.split_documents(docs)

                        # Add metadata
                        for doc in chunked_docs:
                            doc.metadata.update({
                                "source": uploaded_file.name,
                                "type": file_type,
                                "upload_date": time.strftime("%Y-%m-%d %H:%M:%S")
                            })

                        # Store in vector database
                        st.session_state.vector_store.add_documents(chunked_docs)

                        # Track chunk info
                        st.session_state.chunk_info[uploaded_file.name] = len(chunked_docs)
                        total_chunks += len(chunked_docs)

                        st.success(f"Processed {uploaded_file.name}: {len(chunked_docs)} chunks created")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(file_path):
                            os.remove(file_path)

                if total_chunks > 0:
                    st.success(f"Total documents processed: {len(uploaded_files)}, Total chunks: {total_chunks}")
                    st.info("Documents are now ready for querying!")

    st.divider()

    # Search options
    st.markdown('<div class="sidebar-header">🔍 Search Options</div>', unsafe_allow_html=True)

    search_type = st.selectbox(
        "Search Type",
        ["Semantic Search", "Keyword Search", "Hybrid Search"],
        help="Choose search method"
    )

    # Metadata filters
    with st.expander("Advanced Filters"):
        doc_name_filter = st.text_input("Document Name")
        file_type_filter = st.selectbox("File Type", ["All"] + ALLOWED_EXTENSIONS)

    st.divider()

    # Analytics placeholder
    st.markdown('<div class="sidebar-header">📊 Analytics</div>', unsafe_allow_html=True)
    st.info("Analytics dashboard would be here...")

# ----------------- MAIN CONTENT: CHAT INTERFACE -----------------
st.markdown('<div class="sub-header">💬 Chat with Your Documents</div>', unsafe_allow_html=True)

# Display chunk information
if st.session_state.chunk_info:
    with st.expander("📊 Document Processing Summary"):
        total_chunks = sum(st.session_state.chunk_info.values())
        st.write(f"**Total chunks created:** {total_chunks}")
        for doc_name, chunks in st.session_state.chunk_info.items():
            st.write(f"- {doc_name}: {chunks} chunks")

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-user">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 {message["content"]}</div>', unsafe_allow_html=True)

# Chat input
user_input = st.text_input("Ask a question about your documents...", key="chat_input")

if st.button("Send", key="send_btn") and user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    start_time = time.time()

    # Retrieve relevant documents
    try:
        if search_type == "Semantic Search":
            results = st.session_state.vector_store.document_collection.query(
                query_texts=[user_input],
                n_results=3
            )
            retrieved_docs = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in zip(results['documents'][0], results['metadatas'][0])
            ]
        elif search_type == "Keyword Search":
            # Use BM25 for keyword search
            keyword_results = st.session_state.vector_store.bm25_index.get_scores(user_input.split())
            top_indices = np.argsort(keyword_results)[-3:][::-1]
            retrieved_docs = [
                Document(page_content=st.session_state.vector_store.document_texts[idx])
                for idx in top_indices if idx < len(st.session_state.vector_store.document_texts)
            ]
        else:  # Hybrid Search
            retrieved_docs = st.session_state.vector_store.hybrid_search(user_input, k=3)

        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create prompt with context
        prompt = f"""You are a helpful AI assistant that answers questions based ONLY on the provided document content.

IMPORTANT INSTRUCTIONS:
- Answer ONLY using information from the provided document content below.
- If the question cannot be answered using the provided content, say "I don't have enough information in the documents to answer this question."
- Do not make up information or use external knowledge.
- Be specific and quote relevant parts of the documents when possible.
- For questions about diagrams or images, look for OCR text, captions, or descriptions in the content.

Document Content:
{context}

Question: {user_input}

Answer:"""

        # Generate AI response using LLM with context
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)
        ai_response = llm.invoke(prompt).content

        # Track analytics
        response_time = time.time() - start_time
        st.session_state.analytics_tracker.track_query(
            question=user_input,
            answer=ai_response,
            sources=retrieved_docs,
            response_time=response_time,
            username=st.session_state.user_role
        )

    except Exception as e:
        ai_response = f"Sorry, I encountered an error: {str(e)}. Please check your API key and try again."

    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    st.rerun()

# Clear chat
if st.button("Clear Chat", key="clear_btn"):
    st.session_state.chat_history = []
    st.rerun()

st.divider()

# Document Clustering Visualization
st.markdown('<div class="sub-header">📚 Document Clusters</div>', unsafe_allow_html=True)

# Check if there are documents to cluster
try:
    results = st.session_state.vector_store.document_collection.get(include=['documents'])
    if results['documents'] and len(results['documents']) > 0:
        # Perform clustering
        clusters = st.session_state.vector_store.cluster_documents(n_clusters=min(5, len(results['documents'])))

        if clusters:
            # Display clusters
            for cluster_id, docs in clusters.items():
                with st.expander(f"📁 Cluster {cluster_id + 1} ({len(docs)} documents)"):
                    for doc in docs:
                        st.write(f"**{doc['id']}**")
                        st.write(f"*{doc['content']}...*")
                        if 'metadata' in doc and doc['metadata']:
                            st.caption(f"Type: {doc['metadata'].get('type', 'Unknown')}, Source: {doc['metadata'].get('source', 'Unknown')}")
                        st.divider()
        else:
            st.info("Not enough documents to perform clustering. Upload and process more documents.")
    else:
        st.info("No documents available for clustering. Upload and process documents first.")
except Exception as e:
    st.error(f"Error loading clusters: {str(e)}")
    st.info("Document clustering visualization would appear here once documents are processed.")



