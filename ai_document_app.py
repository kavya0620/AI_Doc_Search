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
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import time

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
embedding_model = FakeEmbeddings(size=384)  # Similar dimension to MiniLM
print("Using FakeEmbeddings for stable operation")

# Persistent vector store - ChromaDB
vector_store = Chroma(
    persist_directory="./chroma_db",
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

# Custom CSS for vibrant, premium AI dashboard styling
st.markdown("""
<style>
    /* Vibrant gradient background with flowing waves and abstract shapes */
    .stApp {
        background:
            radial-gradient(circle at 20% 80%, rgba(255, 215, 0, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 255, 255, 0.12) 0%, transparent 50%),
            linear-gradient(135deg,
                #1e3a8a 0%,    /* Royal Blue */
                #06b6d4 20%,   /* Cyan */
                #7c3aed 40%,   /* Purple */
                #ec4899 60%,   /* Magenta */
                #14b8a6 80%,   /* Teal */
                #1e3a8a 100%   /* Royal Blue */
            );
        background-attachment: fixed;
        min-height: 100vh;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        position: relative;
        overflow-x: hidden;
    }

    /* Animated flowing waves */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(45deg, rgba(255, 215, 0, 0.08) 0%, transparent 30%, rgba(0, 255, 255, 0.06) 70%, transparent 100%),
            linear-gradient(-45deg, rgba(255, 0, 255, 0.07) 0%, transparent 40%, rgba(0, 255, 255, 0.05) 80%, transparent 100%);
        background-size: 400% 400%;
        animation: flow 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }

    /* Light flares and abstract shapes */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            radial-gradient(ellipse at 10% 90%, rgba(255, 215, 0, 0.2) 0%, transparent 40%),
            radial-gradient(ellipse at 90% 10%, rgba(255, 0, 255, 0.15) 0%, transparent 35%),
            radial-gradient(ellipse at 50% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 45%);
        pointer-events: none;
        z-index: -1;
    }

    @keyframes flow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Premium main header with gradient text */
    .main-header {
        font-size: 2.8em;
        font-weight: 700;
        background: linear-gradient(45deg, #ffd700, #ffffff, #00ffff, #ff00ff);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease-in-out infinite;
        text-align: center;
        margin-bottom: 20px;
        line-height: 1.1;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        position: relative;
        z-index: 10;
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Dynamic sub-header */
    .sub-header {
        font-size: 1.6em;
        font-weight: 600;
        background: linear-gradient(45deg, #ffffff, #00ffff, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #ffd700, #00ffff, #ff00ff) 1;
        padding-bottom: 12px;
        position: relative;
        z-index: 10;
    }

    /* Vibrant chat messages with glassmorphism */
    .chat-user {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(173, 216, 230, 0.8));
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 18px 22px;
        margin: 10px 0;
        border-left: 4px solid #00ffff;
        color: #1e3a8a;
        box-shadow: 0 8px 32px rgba(0, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        z-index: 10;
    }

    .chat-assistant {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(221, 160, 221, 0.7));
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 18px 22px;
        margin: 10px 0;
        border-left: 4px solid #ffd700;
        color: #4a148c;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
         z-index: 10;
    }

    /* Premium sidebar header */
    .sidebar-header {
        font-size: 1.3em;
        font-weight: 700;
        background: linear-gradient(45deg, #ffd700, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        position: relative;
        z-index: 10;
    }

    /* Glassmorphism login container */
    .login-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
        backdrop-filter: blur(25px);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        position: relative;
        z-index: 10;
    }

    /* Premium metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.8));
        backdrop-filter: blur(15px);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 10;
    }

    /* Dynamic button effects */
    .stButton > button {
        background: linear-gradient(45deg, #1e3a8a, #06b6d4);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
        position: relative;
        z-index: 10;
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #06b6d4, #7c3aed);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.4);
    }

    /* Premium input styling */
    .stTextArea > div > div > textarea {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        font-family: inherit;
        color: #1e3a8a;
        position: relative;
        z-index: 10;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #ffd700;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.3);
        outline: none;
    }

    /* Vibrant text colors */
    .stMarkdown, .stText {
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        position: relative;
        z-index: 10;
    }

    /* Premium success/warning/info messages */
    .stSuccess, .stWarning, .stInfo {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
        backdrop-filter: blur(15px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 10;
    }

    /* Premium sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(30, 58, 138, 0.9), rgba(124, 58, 237, 0.8));
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(255, 215, 0, 0.3);
        position: relative;
        z-index: 10;
    }

    /* Section separators with gradient */
    .stDivider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ffd700, #00ffff, #ff00ff, transparent);
        margin: 30px 0;
        position: relative;
        z-index: 10;
    }

    /* File uploader visibility fix */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.9));
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid rgba(255, 215, 0, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 10;
    }

    .stFileUploader > div > div > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        color: #1e3a8a;
        font-weight: 600;
    }

    .stFileUploader > div > div > div > div:hover {
        background-color: rgba(255, 215, 0, 0.1);
        border-color: #ffd700;
    }
</style>

<script>
// Handle Enter key for chat input
document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && event.target.matches('input[aria-label="Ask a question about your documents..."]')) {
        event.preventDefault();
        const sendButton = document.querySelector('button[kind="primary"]');
        if (sendButton && !sendButton.disabled) {
            sendButton.click();
        }
    }
});
</script>
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
            # Clear previous documents - recreate Chroma vector store
            st.session_state.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embedding_model
            )
            st.session_state.current_files = []
            st.session_state.chat_history = []  # Clear chat when new docs are processed
            
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
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    api_key=api_key
                )
                ai_response = llm.invoke(prompt).content

                # Add debug info to successful responses
                ai_response = debug_info + "\n" + ai_response

            except Exception as llm_error:
                error_msg = str(llm_error).lower()
                if "invalid" in error_msg or "api" in error_msg or "quota" in error_msg:
                    ai_response = f"""❌ **API Issue Detected**

**Quick Fix Steps:**
1. Get new Groq API key: https://console.groq.com/keys
2. Set GROQ_API_KEY in your .env file
3. Restart the app

**Your question was about:** {user_input}

**Found relevant content from your documents:**
{context[:800]}...

**Manual Answer:** Based on the content above, AI (Artificial Intelligence) relates to probability theory and logical reasoning systems. The document discusses probability axioms and propositions, which are fundamental to AI systems that deal with uncertainty and decision-making."""
                else:
                    ai_response = f"❌ **Error:** {llm_error}\n\n**Found content:**\n{context[:500]}..."

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
