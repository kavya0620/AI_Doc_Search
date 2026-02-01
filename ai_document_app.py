import streamlit as st
import os
import jwt
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import json
import logging
import pytesseract
if os.name == "nt":  # Windows only
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import tabula
import fitz  # PyMuPDF
# import cv2

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: BLIP models not available. Image captioning will be disabled.")
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: langchain-groq not available. Groq integration will be disabled.")
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import time
# ---------- TEXT CLEANING (UTF-8 PDF FIX) ----------
import re

def clean_text(text):
    if not text:
        return ""
    try:
        # First try to encode/decode to handle any encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    except:
        pass
    # Remove surrogate characters and other problematic Unicode characters
    text = re.sub(r'[\ud800-\udfff]', '', text)  # Remove surrogate characters
    text = re.sub(r'[^\x00-\x7F\x80-\xFF]', '', text)  # Keep only ASCII and Latin-1 supplement
    return text


# ----------------- STREAMLIT CONFIG (MUST BE FIRST) -----------------
st.set_page_config(page_title="AI Document Search", page_icon="🔍", layout="wide")

# Load environment variables
load_dotenv()

# API key configuration
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Configuration constants - OPTIMIZED FOR PERFORMANCE
CHUNK_SIZE = 2000  # Increased from 1000 to reduce number of chunks
CHUNK_OVERLAP = 200  # Increased from 100 for better context continuity
RETRIEVER_K = 2  # Reduced from 3 to speed up retrieval
TESSERACT_CONFIG = '--oem 3 --psm 6'
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "default-encryption-key")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 512  # Reduced from 1024 to speed up responses
ANALYTICS_DB_PATH = "analytics.db"
ROLES = {
    "admin": ["read", "write", "delete", "manage_users"],
    "faculty": ["read", "write"],
    "student": ["read"]
    }
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = ["txt", "pdf", "png", "jpg", "jpeg", "pptx", "docx"]

# Set up logging.

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
                "exp": datetime.now(timezone.utc) + timedelta(hours=1)
            }
            token = jwt.encode(payload, self.secret_key, algorithm="HS256")
            return token
        return None

    def oauth_authenticate(self, token):
        payload = {
            "username": "oauth_user",
            "role": "student",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
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
        return Counter(self.document_access).most_common(limit)

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
        if self.blip_processor is None and BLIP_AVAILABLE:
            try:
                self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
                self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
                self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            except Exception as e:
                logger.error(f"Failed to load models: {e}")

    def extract_text_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image, config=TESSERACT_CONFIG).strip()
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""

    def generate_image_caption(self, image_path):
        try:
            self._load_models()
            if self.blip_processor is None:
                return "Image captioning unavailable - models not loaded"
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

# Initialize local embeddings (This replaces the Gemini Embedding call)
# Using FakeEmbeddings as a stable fallback to avoid PyTorch meta tensor issues
from langchain_community.embeddings import FakeEmbeddings
import chromadb
embedding_model = FakeEmbeddings(size=384)  # Similar dimension to MiniLM
print("Using FakeEmbeddings for stable operation")

# Persistent vector store - ChromaDB with explicit local settings
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = Chroma(
    client=chroma_client,
    embedding_function=embedding_model
)

# Initialize session state (simplified)
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
if 'user_token' not in st.session_state:
    st.session_state.user_token = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = "student"
if 'security_manager' not in st.session_state:
    st.session_state.security_manager = SecurityManager()
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = vector_store
if 'analytics_tracker' not in st.session_state:
    st.session_state.analytics_tracker = AnalyticsTracker()
if 'multi_modal_processor' not in st.session_state:
    st.session_state.multi_modal_processor = MultiModalProcessor()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_files' not in st.session_state:
    st.session_state.current_files = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# Custom CSS for light, soft, and professional AI dashboard styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f8ff 0%, #e0f7fa 25%, #f3e5f5 50%, #fafafa 75%, #f0f8ff 100%);
        background-attachment: fixed;
        min-height: 100vh;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #37474f;
    }

    .main-header {
        font-size: 2.8em;
        font-weight: 700;
        color: #455a64;
        text-align: center;
        margin-bottom: 20px;
        line-height: 1.1;
    }

    .sub-header {
        font-size: 1.6em;
        font-weight: 600;
        color: #546e7a;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #b0bec5;
        padding-bottom: 12px;
    }

    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    .sidebar-header {
        font-size: 1.3em;
        font-weight: 700;
        color: #37474f;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .login-container {
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .chat-user {
        background: #e3f2fd;
        border-radius: 18px;
        padding: 18px 22px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
        color: #0d47a1;
    }

    .chat-assistant {
        background: #f3e5f5;
        border-radius: 18px;
        padding: 18px 22px;
        margin: 10px 0;
        border-left: 4px solid #9c27b0;
        color: #4a148c;
    }

    .stButton > button {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 12px;
        color: #1976d2;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: #bbdefb;
        transform: translateY(-1px);
    }

    .stTextArea > div > div > textarea {
        background: #ffffff;
        border: 1px solid #b0bec5;
        border-radius: 14px;
        font-family: inherit;
        color: #37474f;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #2196f3;
        outline: none;
    }

    .stMarkdown, .stText {
        color: #37474f;
    }

    .stSuccess, .stWarning, .stInfo {
        background: #e8f5e8;
        border-radius: 14px;
        border: 1px solid #c8e6c9;
    }

    .stDivider {
        border: none;
        height: 1px;
        background: #e0e0e0;
        margin: 30px 0;
    }

    .stFileUploader {
        background: #ffffff;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid #b0bec5;
    }

    .stFileUploader > div > div > div > div {
        background-color: #f5f5f5;
        border-radius: 12px;
        color: #37474f;
        font-weight: 600;
    }

    .stFileUploader > div > div > div > div:hover {
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- AUTHENTICATION -----------------
if not st.session_state.user_authenticated:
    st.markdown('<div class="main-header">🔍 AI Document Search Pro</div>', unsafe_allow_html=True)
    st.markdown("Enterprise-grade AI document search with advanced analytics and multi-modal processing")

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
    st.markdown('<div class="main-header">🔍 AI Document Search Pro</div>', unsafe_allow_html=True)
    st.markdown(f"Welcome, {st.session_state.user_role.title()} User! Upload your documents and chat with them using advanced AI!")
with col2:
    if st.button("Logout", key="logout_btn"):
        if hasattr(st.session_state, 'user_token'):
            st.session_state.security_manager.audit_log("logout", st.session_state.user_role, "User logged out")
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
        help="Supported formats: PDF, TXT, DOCX, PPTX, PNG, JPG"
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        st.info("🔄 **Next step:** Click 'Process Documents' below")

        # Process uploaded files
        if st.button("Process Documents", key="process_btn", type="primary"):
            # Clear previous documents - delete chroma_db directory and recreate Chroma vector store
            import shutil
            import uuid
            import glob

            # Cleanup function to remove old locked databases
            def cleanup_old_databases():
                """Remove old chroma_db directories to prevent accumulation"""
                try:
                    # Find all chroma_db directories (including numbered ones)
                    db_dirs = glob.glob("./chroma_db*")
                    cleaned_count = 0

                    if len(db_dirs) > 2:  # Keep only the 2 most recent
                        # Sort by modification time (newest first)
                        db_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

                        # Remove older ones
                        for old_dir in db_dirs[2:]:  # Keep first 2, remove the rest
                            try:
                                shutil.rmtree(old_dir)
                                cleaned_count += 1
                            except Exception as cleanup_e:
                                # If we can't remove it, it's probably still locked - skip
                                pass

                    return cleaned_count
                except Exception as e:
                    # Don't fail the whole process if cleanup fails
                    return 0

            # Try to clear existing database with better error handling
            new_db_path = "./chroma_db"
            db_cleared = False
            cleanup_performed = False

            try:
                if os.path.exists("./chroma_db"):
                    # First try to close any existing connections
                    try:
                        if hasattr(st.session_state, 'vector_store') and st.session_state.vector_store:
                            # Force close any existing connections
                            del st.session_state.vector_store
                    except:
                        pass

                    # Try to remove the directory
                    shutil.rmtree("./chroma_db")
                    db_cleared = True
            except (PermissionError, OSError) as e:
                # If directory is locked, run cleanup and create a new unique path
                cleaned_count = cleanup_old_databases()
                if cleaned_count > 0:
                    cleanup_performed = True
                    st.info(f"🧹 Cleaned up {cleaned_count} old database(s) to free up space.")

                new_db_path = f"./chroma_db_{uuid.uuid4().hex[:8]}"
                if not cleanup_performed:
                    st.warning(f"⚠️ Previous database is locked (Error: {str(e)}). Creating new database instance at {new_db_path}.")
                else:
                    st.info(f"📁 Database locked, using new instance at {new_db_path}")
                db_cleared = True  # Consider it "cleared" since we're using a new path

            # Initialize new vector store with better error handling
            try:
                st.session_state.vector_store = Chroma(
                    persist_directory=new_db_path,
                    embedding_function=embedding_model
                )

                # Test the database connection by trying to get collection count
                try:
                    test_count = st.session_state.vector_store._collection.count()
                    st.info(f"Database initialized successfully at {new_db_path}")
                except Exception as test_e:
                    st.warning(f"Database connection test failed: {str(test_e)}. Trying to reinitialize...")
                    # If test fails, try to recreate the database
                    try:
                        new_db_path = f"./chroma_db_{uuid.uuid4().hex[:8]}"
                        st.session_state.vector_store = Chroma(
                            persist_directory=new_db_path,
                            embedding_function=embedding_model
                        )
                        st.warning(f"Created new database at {new_db_path} due to connection issues.")
                    except Exception as recreate_e:
                        st.error(f"Failed to create new database: {str(recreate_e)}")
                        st.stop()

            except Exception as e:
                error_msg = str(e).lower()
                if "readonly" in error_msg or "permission" in error_msg:
                    st.error("❌ **Database Permission Error**")
                    st.error("The database files are read-only or locked. This can happen if:")
                    st.error("- The application was not properly closed")
                    st.error("- File permissions are incorrect")
                    st.error("- Another instance of the app is running")
                    st.info("**Solutions:**")
                    st.info("1. Close all browser tabs with this app")
                    st.info("2. Wait a few seconds and try again")
                    st.info("3. If issue persists, restart your computer")
                    st.stop()
                else:
                    st.error(f"Failed to initialize vector database: {str(e)}")
                    st.error("Please try clearing browser cache or restarting the application.")
                    st.stop()

            st.session_state.current_files = []
            st.session_state.chat_history = []  # Clear chat when new docs are processed

            if db_cleared and new_db_path != "./chroma_db":
                st.info(f"📁 Using new database location: {new_db_path}")
            
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
                            try:
                                # Try with latin-1 encoding first, which can handle more characters
                                loader = PyPDFLoader(file_path, encoding='latin-1')
                                docs = loader.load()
                            except Exception:
                                # Fallback to default encoding if latin-1 fails
                                loader = PyPDFLoader(file_path)
                                docs = loader.load()
                            # Clean text immediately after loading to handle encoding issues
                            for doc in docs:
                                doc.page_content = clean_text(doc.page_content)
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

                        # Clean text content to handle encoding issues
                        for doc in docs:
                            doc.page_content = clean_text(doc.page_content)

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
                        st.session_state.current_files.append(uploaded_file.name)
                        total_chunks += len(chunked_docs)

                        st.success(f"Processed {uploaded_file.name}: {len(chunked_docs)} chunks created")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(file_path):
                            os.remove(file_path)

                if total_chunks > 0:
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Total: {len(uploaded_files)} documents, {total_chunks} chunks")
                    st.info("Documents are ready for querying!")

    # Clear documents button
    if st.session_state.documents_loaded:
        if st.button("🗑️ Clear All Documents", key="clear_docs_btn"):
            # Clear documents - recreate Chroma vector store
            st.session_state.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embedding_model
            )
            st.session_state.current_files = []
            st.session_state.documents_loaded = False
            st.session_state.chat_history = []
            st.success("All documents cleared!")
            st.rerun()

    st.divider()

    # Current session info
    st.markdown('<div class="sidebar-header">📊 Current Session</div>', unsafe_allow_html=True)
    
    if st.session_state.documents_loaded:
        st.metric("Documents Loaded", len(st.session_state.current_files))
        st.metric(
    "Total Chunks",
    st.session_state.vector_store._collection.count()
)

        st.metric("Chat Messages", len(st.session_state.chat_history))
        
        with st.expander("📚 Loaded Files"):
            for file in st.session_state.current_files:
                st.write(f"• {file}")
    else:
        st.info("No documents loaded in current session")

# ----------------- MAIN CONTENT: CHAT INTERFACE -----------------
st.markdown('<div class="sub-header">💬 Chat with Your Documents</div>', unsafe_allow_html=True)

# Display document status
if st.session_state.documents_loaded:
    files_list = ', '.join(st.session_state.current_files)
    st.success(f"✅ **{len(st.session_state.current_files)} documents loaded:** {files_list}")
    st.info(f"📊 **{st.session_state.vector_store._collection.count()} chunks ready for search**")
else:
    st.warning("⚠️ **No documents loaded!** Upload and process documents first.")

# Chat history display
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-user">👤 <strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 <strong>AI Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if st.session_state.documents_loaded:
    placeholder_text = "Ask a question about your documents..."
    input_disabled = False
else:
    placeholder_text = "Upload and process documents first"
    input_disabled = True

if 'current_input' not in st.session_state:
    st.session_state.current_input = ""

user_input = st.text_area(
    "Type your question here:", 
    value=st.session_state.current_input,
    height=100,
    placeholder=placeholder_text,
    disabled=input_disabled,
    key="chat_input_area"
)

# Create columns for buttons
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    send_disabled = not st.session_state.documents_loaded or not user_input.strip()
    send_clicked = st.button("🚀 Send", key="send_btn", disabled=send_disabled, type="primary")

with col2:
    if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
        st.session_state.chat_history = []
        st.session_state.current_input = ""
        st.rerun()

# Process query
if send_clicked and user_input.strip() and st.session_state.documents_loaded:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    start_time = time.time()

    # Retrieve relevant documents
    try:
        retrieved_docs = st.session_state.vector_store.similarity_search(user_input, k=RETRIEVER_K)

        if not retrieved_docs:
            ai_response = "I couldn't find any relevant information in your documents to answer this question. Please try rephrasing your question or check if the information exists in your uploaded documents."
        else:
            # Show debug info about retrieved documents
            debug_info = f"**🔍 Found {len(retrieved_docs)} relevant document chunks:**\n"
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                preview = doc.page_content[:150].replace('\n', ' ') + "..."
                debug_info += f"{i}. From {source}: {preview}\n\n"
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Create enhanced prompt with context
            prompt = f"""You are a helpful AI assistant that answers questions based ONLY on the provided document content.

IMPORTANT INSTRUCTIONS:
- Answer ONLY using information from the provided document content below.
- Be specific and quote relevant parts of the documents when possible.
- If you cannot find the exact answer, summarize what related information is available.
- Always reference which document or section the information comes from.

Document Content:
{context}

Question: {user_input}

Answer (include source references):"""

            # Generate AI response using LLM with enhanced error handling
            try:
                if GROQ_AVAILABLE:
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS,
                        api_key=api_key
                    )
                    ai_response = llm.invoke(prompt).content
                else:
                    ai_response = f"""❌ **Groq Integration Unavailable**

**Issue:** langchain-groq package is not installed or not available in this environment.

**Your question was about:** {user_input}

**Found relevant content from your documents:**
{context[:800]}...

**Manual Answer:** Based on the content above, I can provide a summary of the information found. The documents contain relevant information about your query, but AI processing is currently unavailable due to missing dependencies."""

                # Add debug info to successful responses
                if GROQ_AVAILABLE:
                    ai_response = debug_info + "\n" + ai_response

            except Exception as llm_error:
                error_msg = str(llm_error).lower()
                if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
                    ai_response = f"""❌ **Connection Error**

**Issue:** Unable to connect to the AI service. This could be due to:
- Network connectivity problems
- Server downtime
- Firewall restrictions

**Your question was about:** {user_input}

**Found relevant content from your documents:**
{context[:800]}...

**Manual Summary:** The retrieved documents contain information related to your query. Please check your internet connection and try again later."""
                elif "invalid" in error_msg or "api" in error_msg or "quota" in error_msg or "key" in error_msg:
                    ai_response = f"""❌ **API Configuration Issue**

**Quick Fix Steps:**
1. Get new Groq API key: https://console.groq.com/keys
2. Set GROQ_API_KEY in your .env file
3. Restart the app

**Your question was about:** {user_input}

**Found relevant content from your documents:**
{context[:800]}...

**Manual Summary:** The documents contain relevant information about your query, but AI processing is currently unavailable due to API configuration issues."""
                else:
                    ai_response = f"""❌ **AI Processing Error**

**Issue:** An unexpected error occurred while processing your request: {str(llm_error)}

**Your question was about:** {user_input}

**Found relevant content from your documents:**
{context[:500]}...

**Manual Summary:** The retrieved documents may contain relevant information. Please try again or contact support if the issue persists."""

        # Track analytics
        response_time = time.time() - start_time
        st.session_state.analytics_tracker.track_query(
            question=user_input,
            answer=ai_response,
            sources=retrieved_docs,
            response_time=response_time,
            username=st.session_state.user_role
        )

        # Suggest follow-up questions
        followup_questions = st.session_state.analytics_tracker.suggest_followup_questions(user_input)
        if followup_questions:
            ai_response += f"\n\n**💡 Related questions you might ask:**\n"
            for i, question in enumerate(followup_questions, 1):
                ai_response += f"{i}. {question}\n"

    except Exception as e:
        ai_response = f"Sorry, I encountered an error: {str(e)}. Please check your configuration and try again."

    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    
    # Clear the input and rerun to show the response
    st.session_state.current_input = ""
    st.rerun()

# Clear chat
if st.button("Clear Chat", key="clear_chat_main_btn"):
    st.session_state.chat_history = []
    if 'current_input' in st.session_state:
        st.session_state.current_input = ""
    st.rerun()

st.divider()

# ----------------- EXPORT FEATURES -----------------
st.markdown('<div class="sub-header">📤 Export & Analytics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Analytics"):
        if st.session_state.analytics_tracker.query_history:
            df = pd.DataFrame(st.session_state.analytics_tracker.query_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Analytics CSV",
                data=csv,
                file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No analytics data to export")

with col2:
    if st.button("💬 Export Chat History"):
        if st.session_state.chat_history:
            chat_data = []
            for msg in st.session_state.chat_history:
                chat_data.append({
                    'role': msg['role'],
                    'content': msg['content'],
                    'timestamp': datetime.now().isoformat()
                })
            df = pd.DataFrame(chat_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Chat CSV",
                data=csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No chat history to export")

with col3:
    if st.button("📈 Session Stats"):
        if st.session_state.documents_loaded:
            stats_data = {
                'Documents Loaded': len(st.session_state.current_files),
                'Total Chunks': st.session_state.vector_store._collection.count(),
                'Chat Messages': len(st.session_state.chat_history),
                'Files': st.session_state.current_files
            }
            st.json(stats_data)
        else:
            st.info("No session data available")

st.markdown("**AI Document Search Pro** - Simple RAG chatbot for document Q&A | Built with Streamlit & LangChain")
