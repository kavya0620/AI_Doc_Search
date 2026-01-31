import streamlit as st
import os
import jwt
from datetime import datetime, timedelta
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
import streamlit as st
import os
import jwt
from datetime import datetime, timedelta
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
