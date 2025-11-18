import streamlit as st
import os, json, bcrypt
import cv2
import numpy as np
from time import time, sleep
from datetime import datetime, timedelta
import pandas as pd
from src.detector import PlateDetector
from src.ocr import PlateOCR
import re
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import hashlib

def format_plate_text(plate_text: str) -> str:
    """
    Format plate text by adding spaces between logical segments.
    Example:
    NOTAVALIDNUMBERPLATE -> NOT A VALID NUMBER PLATE
    """
    if not plate_text:
        return ""

    # Replace common words found in invalid plates
    return (plate_text
            .replace("NOTA", "NOT A ")
            .replace("VALID", "VALID ")
            .replace("NUMBER", "NUMBER ")
            .replace("PLATE", "PLATE ")
            ).strip()

# Accepts: KA 18 EQ 0001, KA18EQ0001, etc.
PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")

def normalize_plate(text: str) -> str:
    """Uppercase, remove spaces and non-alphanumerics."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def is_indian_plate(text: str) -> bool:
    """Strict but flexible Indian plate check on normalized text."""
    return bool(PLATE_REGEX.match(normalize_plate(text)))

# Maps helpful for OCR corrections:
AMBIGUOUS_TO_DIGIT = str.maketrans({
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6'
})
AMBIGUOUS_TO_LETTER = str.maketrans({
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'
})

def clean_ocr_text(raw: str) -> str:
    """Remove non-alphanumerics and upper-case."""
    if raw is None:
        return ""
    return ''.join(ch for ch in raw.upper() if ch.isalnum())

def try_fix_plate(raw: str) -> str | None:
    """
    Try multiple simple fixes and return a plate string that matches PLATE_REGEX,
    or None if no valid candidate is found.
    """
    s0 = clean_ocr_text(raw)
    if not s0:
        return None

    # direct match
    if PLATE_REGEX.match(s0):
        return s0

    # Try mapping ambiguous letters -> digits (useful when digits misread as letters)
    s1 = s0.translate(AMBIGUOUS_TO_DIGIT)
    if PLATE_REGEX.match(s1):
        return s1

    # Try mapping ambiguous digits -> letters (useful when letters misread as digits)
    s2 = s0.translate(AMBIGUOUS_TO_LETTER)
    if PLATE_REGEX.match(s2):
        return s2

    # Try both transformations (letters->digits after digit->letter)
    s3 = s2.translate(AMBIGUOUS_TO_DIGIT)
    if PLATE_REGEX.match(s3):
        return s3

    # Sometimes OCR introduces spaces; try splitting into plausible groups
    # e.g. 'KA 01 AB 1234' -> KA01AB1234
    if ' ' in raw:
        s_no_space = clean_ocr_text(raw.replace(' ', ''))
        if PLATE_REGEX.match(s_no_space):
            return s_no_space
        s_mapped = s_no_space.translate(AMBIGUOUS_TO_DIGIT)
        if PLATE_REGEX.match(s_mapped):
            return s_mapped

    # Give up
    return None

# ==================== USERS JSON ====================
USERS_FILE = "users.json"
DETECTIONS_FILE = "detections.json"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def load_detections():
    if os.path.exists(DETECTIONS_FILE):
        try:
            with open(DETECTIONS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_detection(detection_data):
    detections = load_detections()
    detections.append(detection_data)
    # Keep only last 1000 detections to prevent file from growing too large
    if len(detections) > 1000:
        detections = detections[-1000:]
    with open(DETECTIONS_FILE, "w") as f:
        json.dump(detections, f, indent=4)

USERS = load_users()

def verify_password(username, password):
    if username not in USERS:
        return False
    stored_hash = USERS[username]["password"].encode()
    return bcrypt.checkpw(password.encode(), stored_hash)

def register_user(username, password):
    if username in USERS:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    USERS[username] = {"password": hashed_pw}
    save_users(USERS)
    return True

# ==================== SESSION STATE ====================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "register" not in st.session_state:
    st.session_state["register"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "detection_history" not in st.session_state:
    st.session_state["detection_history"] = load_detections()
if "analytics_data" not in st.session_state:
    st.session_state["analytics_data"] = []
if "camera_index" not in st.session_state:
    st.session_state["camera_index"] = 0
if "alert_plates" not in st.session_state:
    st.session_state["alert_plates"] = ["KA01AB1234", "MH20EE0071"]  # Example alert plates
if "recent_detections" not in st.session_state:
    st.session_state["recent_detections"] = {}  # Track recently detected plates to prevent duplicates
if "cooldown_period" not in st.session_state:
    st.session_state["cooldown_period"] = 30  # Seconds before same plate can be detected again

# ==================== UTILITY FUNCTIONS ====================
def get_base64_image(image):
    """Convert image to base64 for HTML display"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_analytics_charts():
    """Create matplotlib charts for analytics"""
    if not st.session_state.analytics_data:
        return None
    
    df = pd.DataFrame(st.session_state.analytics_data)
    if df.empty:
        return None
        
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#0f172a')
    axes = axes.flatten()
    
    # Chart 1: Detection timeline
    timeline_data = df.groupby(df['timestamp'].dt.floor('H')).size()
    axes[0].plot(timeline_data.index, timeline_data.values, marker='o', color='#6366f1')
    axes[0].set_title('Detections Over Time', color='white', pad=20)
    axes[0].set_facecolor('#1e293b')
    axes[0].tick_params(colors='white')
    axes[0].grid(True, alpha=0.3)
    
    # Chart 2: Confidence distribution
    conf_bins = np.linspace(0, 1, 21)
    axes[1].hist(df['confidence'], bins=conf_bins, color='#8b5cf6', alpha=0.7)
    axes[1].set_title('Confidence Distribution', color='white', pad=20)
    axes[1].set_facecolor('#1e293b')
    axes[1].tick_params(colors='white')
    axes[1].grid(True, alpha=0.3)
    
    # Chart 3: Source comparison
    source_data = df['source'].value_counts()
    axes[2].pie(source_data.values, labels=source_data.index, autopct='%1.1f%%', 
                colors=['#6366f1', '#8b5cf6'])
    axes[2].set_title('Detection Source', color='white', pad=20)
    
    # Chart 4: Validity ratio
    valid_data = df['valid'].value_counts()
    labels = ['Valid' if x else 'Invalid' for x in valid_data.index]
    axes[3].pie(valid_data.values, labels=labels, autopct='%1.1f%%', 
                colors=['#10b981', '#ef4444'])
    axes[3].set_title('Plate Validity', color='white', pad=20)
    
    plt.tight_layout()
    return fig

def is_duplicate_detection(plate_text, confidence):
    """
    Check if this plate was recently detected to prevent duplicates.
    Returns True if plate was recently detected, False otherwise.
    """
    current_time = time()
    plate_text = plate_text.upper().strip()
    
    # Clean up old entries (older than cooldown period)
    for plate, detection_time in list(st.session_state.recent_detections.items()):
        if current_time - detection_time > st.session_state.cooldown_period:
            del st.session_state.recent_detections[plate]
    
    # Check if plate was recently detected
    if plate_text in st.session_state.recent_detections:
        # Update the detection time but don't save as new detection
        st.session_state.recent_detections[plate_text] = current_time
        return True
    
    # Add to recent detections
    st.session_state.recent_detections[plate_text] = current_time
    return False

# ==================== LOGIN / REGISTER ====================
if not st.session_state["authenticated"]:
    if not st.session_state["register"]:
        # ==================== HEADER AT TOP ====================
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 0.5px;">
                <h3 style="color:#6366f1; font-size: 2.5rem; margin-bottom: 0; font-weight: 700;">
                    BAGALKOT UNIVERSITY, JAMKHANDI
                </h3>
                <h2 style="color:#8b5cf6; margin-top: 0.1rem; font-weight: 600;">
                    Department of Computer Application<br>
                    AI-Powered Vehicle Number Plate Recognition and Theft Plate Detection
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==================== TWO COLUMN LAYOUT ====================
        col1, col2 = st.columns([2, 1], gap="large")

        with col1:
            # Try to load image with error handling
            try:
                st.image("F:/vnpr_project/images/uv.png", width=500)
            except:
                st.markdown("""
                <div style="height: 300px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                            border-radius: 15px; display: flex; align-items: center; justify-content: center;">
                    <h3 style="color: white;">AI-Powered Vehicle Number Plate Recognition and Theft Plate Detection</h3>
                </div>
                """, unsafe_allow_html=True)

            # ==================== ABOUT PROJECT ====================
            st.markdown(
                """
                <div style="text-align: justify; margin-top: 20px; padding: 20px; 
                            border: 2px solid rgba(99, 102, 241, 0.3); border-radius: 15px; 
                            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
                            font-size: 16; line-height: 1.6; color: #e2e8f0; 
                            backdrop-filter: blur(10px); box-shadow: 0 8px 20px rgba(0,0,0,0.2);">
                    <h4 style="color:#6366f1; margin-top: 0; border-bottom: 2px solid #6366f1; padding-bottom: 10px;">
                        📋 About the Project
                    </h4>
                    This project is a <b style="color:#8b5cf6">AI-Powered Vehicle Number Plate Recognition and Theft Plate Detection</b>.<br>
                    • Uses <b style="color:#10b981">YOLOv8</b> for accurate number plate detection<br>
                    • <b style="color:#0ea5e9">EasyOCR</b> extracts characters from detected platesbr>
                    • Supports both <b style="color:#f59e0b">image upload</b> and <b style="color:#f59e0b">live webcam</b><br>
                    • Built with a user-friendly <b style="color:#ec4899">Streamlit interface</b><br>
                    • Works in <b style="color:#ef4444">real-time</b> with high accuracy<br>
                    • Designed for <b style="color:#84cc16">traffic monitoring and security applications</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            # Login Box with modern styling
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 00%);
                            padding: 25px; border-radius: 15px; border: 1px solid rgba(99, 102, 241, 0.3);
                            box-shadow: 0 10px 25px rgba(0,0,0,0.2); backdrop-filter: blur(10px);">
                    <h3 stylecolor:#6366f1; text-align: center; margin-bottom: 25px;">
                        🔐Secure Login
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Login form inside the styled container
            with st.container():
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                
                login_btn = st.button(
                    "🚀 Login", 
                    use_container_width=True,
                    type="primary"
                )
                
                if login_btn:
                    if verify_password(username, password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")

            st.markdown("---")
            
            # Register button
            register_btn = st.button(
                "📝 New User? Register Here", 
                use_container_width=True,
                help="Create a new account"
            )
            
            if register_btn:
                st.session_state["register"] = True
                st.rerun()

        # Footer note
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; margin-top: 20px;">
                AI-Powered Vehicle Number Plate Recognition and Theft Plate Detection | Powered by YOLOv8 and EasyOCR
            </div>
            """,
            unsafe_allow_html=True
        )

        st.stop()

# ==================== REGISTRATION PAGE ====================
if st.session_state["register"]:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color:#6366f1;">Create New Account</h2>
            <p style="color:#64748b;">Register to access the Vehicle Number Plate Recognition System</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Username", placeholder="Choose a username")
        with col2:
            new_password = st.text_input("Password", type="password", placeholder="Create a password")
        
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        submitted = st.form_submit_button("Register Account", type="primary")
        
        if submitted:
            if not new_username or not new_password:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif register_user(new_username, new_password):
                st.success("Account created successfully! Please login.")
                st.session_state["register"] = False
                st.rerun()
            else:
                st.error("Username already exists")
    
    if st.button("← Back to Login"):
        st.session_state["register"] = False
        st.rerun()
    
    st.stop()

# ==================== MAIN APP AFTER LOGIN ====================
# Modern CSS styling with professional color theme
st.markdown("""
<style>
    :root {
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --secondary: #1f2937;
        --dark: #0f172a;
        --light: #f8fafc;
        --accent: #8b5cf6;
        --teal: #0d9488;
        --card-bg: rgba(30, 41, 59, 0.9);
        --sidebar-bg: linear-gradient(135deg, #1e293b 0%, #0f72a 100%);
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --control-panel: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #f8fafc;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #f8fafc;
    }
    
    .block-container {
        background-color: var(--card-b);
        border-radius: 1rem;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, var(--primary-hover) 0%, #7c3aed 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    .success-box {
        padding: 1.25rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(13, 148, 136, 0.2) 100%);
        border-radius: 0.75rem;
        border-left: 4px solid var(--success);
        margin: 1rem 0;
        color: var(--light);
        backdrop-filter: blur(5px);
    }
    
    .warning-box {
        padding: 1.25rem;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%);
        border-radius: 0.75rem;
        border-left: 4px solid var(--warning);
        margin: 1rem 0;
        color: var(--light);
        backdrop-filter: blur(5px);
    }
    
    .alert-box {
        padding: 1.25rem;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        border-radius: 0.75rem;
        border-left: 4px solid var(--error);
        margin: 1rem 0;
        color: var(--light);
        backdrop-filter: blur(5px);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .performance-metrics {
        font-family: 'SF Mono', monospace;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        color: var(--light);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .debug-image {
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        margin: 0.5 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Control Panel Styling */
    .sidebar .sidebar-content {
        background: var(--sidebar-bg);
        box-shadow: 5px 0 15px rgba(0,0,0,0.3);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .control-panel-header {
        background: var(--control-panel);
        padding: 1.5rem;
        border-radius: 0.75;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .control-panel-section {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    
    .control-panel-section h3 {
        color: #e2e8f0;
        margin-top: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 0.75rem;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        background: transparent;
        color: #cbd5e1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="false"] {
        background: rgba(30, 41, 59, 0.5);
    }
    
    .stSlider [data-baseweb="slider"] {
        color: var(--primary) !important;
    }
    
    .stSlider [ata-baseweb="thumb"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--light) !important;
        font-weight: 700;
    }
    
    .footer {
        padding: 1.5rem;
        text-align: center;
        color: #94a3b8;
        font-size: 0.875rem;
        background: rgba(15, 23, 42, 0.7);
        border-radius: 1rem;
        margin-top: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .file-uploader {
        border: 2px dashed #475569;
        border-radius: 1rem;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(5px);
        color: #e2e8f0;
    }
    
    .file-uploader:hover {
        border-color: var(--primary);
        background: rgba(30, 41, 59, 0.8);
        transform: translateY(-2px);
    }
    
    .spinner {
        color: var(--primary) !important;
    }
    
    .st-expander {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        backdrop-filter: blur(5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
    }
    
    .st-expander .st-emotion-cache-1ffdx1i {
        padding: 1rem;
        color: #e2e8f0;
    }
    
    /* Checkbox styling */
    .stCheckbox [data-baseweb="checkbox"] {
        background: rgba(30, 41, 59, 0.5);
        border-color: #475569;
    }
    
    .stCheckbox [data-baseweb="checkbox"]:checked {
        background: var(--primary);
        border-color: var(--primary);
    }
    
    /* Text colors */
    p, div, span, label {
        color: #e2e8f0 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--primary-hover) 0%, #7c3aed 100%);
    }
    
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }
        
        .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main app content with fixed layout
st.markdown("""
<style>
.main .block-container {
    max-width: 95% !important;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    with st.spinner("🚀 Loading AI models..."):
        try:
            detector = PlateDetector()
            ocr = PlateOCR(gpu=True)
            return detector, ocr
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None

if 'detection_active' not in st.session_state:
    st.session_state.update({
        'detection_active': False,
        'output_dir': 'detected_plates',
        'detection_history': load_detections(),
        'analytics_data': [],
        'camera_index': 0,
        'alert_plates': ["KA01AB1234", "MH20EE0071"],
        'recent_detections': {},
        'cooldown_period': 30,  # 30 seconds cooldown for same plate
        'last_detection_result': None,
        'last_processed_plate': None
    })
    os.makedirs(st.session_state.output_dir, exist_ok=True)

# Modern sidebar with enhanced control panel
with st.sidebar:
    st.markdown(f"""
    <div class="control-panel-header">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="4" width="20" height="16" rx="2" ry="2"></rect>
                <path d="M2 10h20M7 14.01M11 14h.01"></path>
            </svg>
            <h1 style="margin-left: 0.5rem; margin-bottom: 0; color: white;">Control Panel</h1>
        </div>
        <p style="margin: 0; opacity: 0.9;">Welcome, {st.session_state.username}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="control-panel-section">', unsafe_allow_html=True)
        st.markdown("""
        <h3>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            Detection Settings
        </h3>
        """, unsafe_allow_html=True)
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.6, 0.05, help="Adjust the minimum confidence level for plate detection")
        min_plate_size = st.slider("Min Plate Size (px)", 50, 300, 100, help="Set the minimum size for detected plates in pixels")
        frame_skip = st.slider("Process Every N Frames", 1, 5, 2, help="Skip frames to improve performance")
        ocr_conf_threshold = st.slider("OCR confidence threshold", 0.0, 1.0, 0.35, 0.05)
        st.session_state.camera_index = st.number_input("Camera Index", 0, 10, 0, help="Select which camera to use (0 for default)")
        st.session_state.cooldown_period = st.slider("Cooldown Period (seconds)", 10, 300, 30, help="Time before the same plate can be detected again")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="control-panel-section">', unsafe_allow_html=True)
        st.markdown("""
        <h3>
            <svg xmlns="http://www.w3.org/2000/svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="4 7 4 4 20 4 20 7"></polyline>
                <line x1="9" y1="20" x2="15" y2="20"></line>
                <line x1="12" y1="4" x2="12" y2="20"></line>
            </svg>
            OCR Settings
        </h3>
        """, unsafe_allow_html=True)
        show_debug = st.checkbox("Show OCR Preprocessing", True, help="Display OCR preprocessing steps")
        ocr_confidence = st.slider("OCR Min Confidence", 0.1, 0.9, 0.6, 0.05, help="Set the minimum confidence level for text recognition")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="control-panel-section">', unsafe_allow_html=True)
        st.markdown("""
        <h3>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            Output Settings
        </h3>
        """, unsafe_allow_html=True)
        save_images = st.checkbox("Save Detected Plates", True, help="Save detected plate images to disk")
        show_metrics = st.checkbox("Show Performance Metrics", True, help="Display performance statistics")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="control-panel-section">', unsafe_allow_html=True)
        st.markdown("""
        <h3>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            Alert Settings
        </h3>
        """, unsafe_allow_html=True)
        alert_plate = st.text_input("Add Alert Plate", placeholder="Enter plate number to alert")
        if st.button("Add Alert", use_container_width=True) and alert_plate:
            if alert_plate.upper() not in st.session_state.alert_plates:
                st.session_state.alert_plates.append(alert_plate.upper())
                st.success(f"Added {alert_plate.upper()} to alert list")
        
        if st.session_state.alert_plates:
            st.write("**Alert Plates:**")
            for plate in st.session_state.alert_plates:
                col1, col2 = st.columns([3, 1])
                col1.write(plate)
                if col2.button("🗑️", key=f"delete_{plate}"):
                    st.session_state.alert_plates.remove(plate)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== LOGOUT BUTTON ====================
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.success("You have been logged out.")
        st.rerun()

# Main content area
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32 viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M17 a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path>
    </svg>
    <h1 style="margin-left: 0.75rem; margin-bottom: 0; color: var(--light)">AI-Powered Vehicle Number Plate Recognition and Theft Plate Detection</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""

""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎥 Live Detection", "📁 Image Upload", "📊 Analytics", "📋 History", "⚙️ Settings"])

# ==================== IMAGE UPLOAD TAB ====================
with tab2:
    st.markdown("""
    <div class="file-uploader">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <h3 style="margin-top: 0.5rem; color: var(--light);">Upload Vehicle Image</h3>
        <p style="color: #94a3b8; margin-bottom: 0;">JPG, PNG, or JPEG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        detector, ocr = load_models()
        
        if detector is None or ocr is None:
            st.error("Failed to load models. Please check the model files.")
            st.stop()

        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("Failed to decode image!")
                st.stop()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plates = detector.detect(frame)
            current_results = []

            col1, col2 = st.columns(2)

            with col1:
                st.image(frame_rgb, caption="Uploaded Image", use_container_width=True)

            with col2:
                if not plates:
                    st.warning("No license plates detected in the image.")
                else:
                    st.markdown("<h3 style='color: var(--light);'>🔍 Detection Results</h3>", unsafe_allow_html=True)
                    
                    for plate_img, bbox, conf in plates:
                        if conf >= conf_threshold:
                            texts = ocr.recognize_text(plate_img)
                            if texts:
                                plate_text = texts[0].strip()
                                normalized_detected = normalize_plate(plate_text)
                                
                                # Try to fix the plate text if it doesn't match the pattern
                                fixed_plate = try_fix_plate(plate_text)
                                if fixed_plate:
                                    normalized_detected = fixed_plate
                                
                                # Create normalized alert list
                                normalized_alert_list = [normalize_plate(p) for p in st.session_state.alert_plates]

                                # Check if it's a valid Indian plate
                                is_valid_indian = is_indian_plate(normalized_detected)
                                
                                # Priority-based classification
                                if normalized_detected in normalized_alert_list:
                                    status = "SUSPECT NUMBER PLATE"
                                    is_alert = True
                                    is_valid = False
                                elif is_valid_indian:
                                    status = "VALID NUMBER PLATE"
                                    is_alert = False
                                    is_valid = True
                                else:
                                    status = "INVALID' NUMBER' PLATE"
                                    is_alert = False
                                    is_valid = False

                                # Duplicate detection check
                                is_duplicate = is_duplicate_detection(normalized_detected, conf)

                                # Store detection result
                                result_data = {
                                    'text': normalized_detected,
                                    'confidence': conf,
                                    'valid': is_valid,
                                    'alert': is_alert,
                                    'status': status,
                                    'duplicate': is_duplicate,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                current_results.append(result_data)

                                # Debug preprocessed OCR image
                                if show_debug:
                                    debug_img = ocr.debug_preprocess(plate_img)
                                    st.image(debug_img, caption=f"Preprocessed: {normalized_detected}", channels="GRAY", use_container_width=True)

                                # ==================== DISPLAY RESULTS ====================
                                # ===== SUSPECT PLATE =====
                                if result_data['alert']:
                                    st.markdown(f"""
                                    <div class="alert-box">
                                        <strong style="color: red;">🚨 SUSPECT NUMBER PLATE DETECTED</strong><br>
                                        Plate: {result_data['text']}<br>
                                        Confidence: {result_data['confidence']:.2f}<br>
                                        Time: {result_data['timestamp']}
                                    </div>
                                    """, unsafe_allow_html=True)

                                # ===== VALID PLATE =====
                                elif result_data['valid']:
                                    st.markdown(f"""
                                    <div class="success-box">
                                        <strong style="color: green;">✅ VALID NUMBER PLATE</strong><br>
                                        Plate: {result_data['text']}<br>
                                        Confidence: {result_data['confidence']:.2f}<br>
                                        Time: {result_data['timestamp']}
                                    </div>
                                    """, unsafe_allow_html=True)

                                # ===== INVALID PLATE =====
                                else:
                                    st.markdown(f"""
                                    <div class="warning-box">
                                        <strong style="color: orange;">❌ INVALID NUMBER PLATE</strong><br>
                                        Plate: {result_data['text']}<br>
                                        Confidence: {result_data['confidence']:.2f}<br>
                                        Time: {result_data['timestamp']}
                                    </div>
                                    """, unsafe_allow_html=True)

                                # ===== SAVE TO HISTORY & ANALYTICS (NO DUPLICATES) =====
                                if not result_data['duplicate']:
                                    st.session_state.detection_history.append(result_data)
                                    st.session_state.analytics_data.append({
                                        'plate': result_data['text'],
                                        'confidence': result_data['confidence'],
                                        'valid': result_data['valid'],
                                        'alert': result_data['alert'],
                                        'status': result_data['status'],
                                        'timestamp': result_data['timestamp'],
                                        'source': 'image_upload'
                                    })
                                    save_detection(result_data)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# ==================== LIVE DETECTION TAB ====================
# Initialize plate history
if "plate_history" not in st.session_state:
    st.session_state["plate_history"] = []

def run_live_detection():
    detector, ocr = load_models()
    if detector is None or ocr is None:
        st.error("Failed to load models. Please check the model files.")
        st.session_state.detection_active = False
        return

    cap = cv2.VideoCapture(int(st.session_state.camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create layout for live detection
    video_col, result_col = st.columns([2, 1])
    
    with video_col:
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()
    
    with result_col:
        result_placeholder = st.empty()
        alert_placeholder = st.empty()
        plate_placeholder = st.empty()

    frame_count = 0
    prev_time = time()
    detection_stats = {
        'total_frames': 0,
        'detection_time': 0.0,
        'ocr_time': 0.0,
        'plates_detected': 0,
        'unique_plates_detected': 0
    }

    local_conf_th = float(conf_threshold)
    local_min_plate_size = int(min_plate_size)
    local_frame_skip = max(1, int(frame_skip))
    local_ocr_conf_th = float(ocr_conf_threshold)

    try:
        while st.session_state.detection_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video feed")
                break

            frame_count += 1
            detection_stats['total_frames'] += 1
            if frame_count % local_frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame_rgb.copy()

            # Detection
            start_det = time()
            plates = detector.detect(frame)
            det_time = time() - start_det
            detection_stats['detection_time'] += det_time

            latest_result = None
            alert_detected = False
            processed_plate_img = None

            for plate_img, bbox, det_conf in plates:
                x, y, w, h = bbox
                if det_conf < local_conf_th or w <= local_min_plate_size or h <= local_min_plate_size:
                    continue

                # OCR
                start_ocr = time()
                ocr_results = ocr.recognize_text(plate_img)
                ocr_time = time() - start_ocr
                detection_stats['ocr_time'] += ocr_time

                if not ocr_results:
                    continue

                plate_text = str(ocr_results[0]).strip().upper()
                norm_plate = normalize_plate(plate_text)
                
                # Try to fix the plate text if it doesn't match the pattern
                fixed_plate = try_fix_plate(plate_text)
                if fixed_plate:
                    norm_plate = fixed_plate

                # Classification rules
                alert_list = [normalize_plate(p) for p in st.session_state.get('alert_plates', [])]
                matches_format = is_indian_plate(norm_plate)

                if norm_plate in alert_list:
                    status = "SUSPECT NUMBER PLATE"
                    color = (0, 0, 255)  # Red
                    is_alert = True
                    is_valid = False
                    alert_detected = True
                elif matches_format:
                    status = "VALID NUMBER PLATE"
                    color = (0, 255, 0)  # Green
                    is_alert = False
                    is_valid = True
                else:
                    status = "INVALID NUMBER PLATE"
                    color = (0, 165, 255)  # Orange
                    is_alert = False
                    is_valid = False

                # Duplicate check
                is_duplicate = is_duplicate_detection(norm_plate, det_conf)

                # Draw results on frame
                label = f"{norm_plate} ({status}){' DUP' if is_duplicate else ''}"
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(display_frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Store processed plate image for display
                processed_plate_img = plate_img.copy()
                processed_plate_img = cv2.copyMakeBorder(
                    processed_plate_img, 10, 10, 10, 10, 
                    cv2.BORDER_CONSTANT, value=color
                )
                cv2.putText(processed_plate_img, f"{norm_plate} ({det_conf:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Save detection record
                if not is_duplicate and det_conf >= local_conf_th:
                    detection_record = {
                        'text': norm_plate,
                        'confidence': float(det_conf),
                        'valid': bool(is_valid),
                        'alert': bool(is_alert),
                        'status': status,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.detection_history.append(detection_record)
                    st.session_state.analytics_data.append({
                        'plate': norm_plate,
                        'confidence': float(det_conf),
                        'valid': bool(is_valid),
                        'alert': bool(is_alert),
                        'status': status,
                        'timestamp': detection_record['timestamp'],
                        'source': 'live_detection'
                    })
                    save_detection(detection_record)
                    detection_stats['plates_detected'] += 1
                    detection_stats['unique_plates_detected'] += 1
                    
                    # Save the plate image if enabled
                    if save_images:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{st.session_state.output_dir}/plate_{norm_plate}_{timestamp}.jpg"
                        cv2.imwrite(filename, plate_img)

                    # ✅ Save plate history
                    st.session_state.plate_history.append({
                        "image": processed_plate_img,
                        "plate": norm_plate,
                        "status": status,
                        "confidence": float(det_conf),
                        "timestamp": detection_record['timestamp']
                    })

                latest_result = (norm_plate, det_conf, status, is_alert, is_valid, is_duplicate, processed_plate_img)

            # FPS
            now = time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            # Display video feed
            video_placeholder.image(display_frame, channels="RGB",
                                    caption="Live Detection Feed", use_container_width=True)

            # Display processed plate image and results
            if latest_result:
                plate_text, conf, status, is_alert, is_valid, dup_f, proc_img = latest_result
                
                plate_placeholder.image(proc_img, caption="Processed Plate", use_container_width=True)
                
                if is_alert:
                    result_placeholder.markdown(f"""
                    <div class="alert-box">
                        <strong style="color: red;">🚨 SUSPECT NUMBER PLATE DETECTED</strong><br>
                        Plate: {plate_text}<br>
                        Confidence: {conf:.2f}<br>
                        Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </div>
                    """, unsafe_allow_html=True)
                elif is_valid:
                    result_placeholder.markdown(f"""
                    <div class="success-box">
                        <strong style="color: green;">✅ VALID NUMBER PLATE</strong><br>
                        Plate: {plate_text}<br>
                        Confidence: {conf:.2f}<br>
                        Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                   formatted_plate = format_plate_text(plate_text)
                   result_placeholder.markdown(f"""
                    <div class="warning-box">
                    <strong style="color: orange;">⚠️ INVALID NUMBER PLATE</strong><br><br>
                    <span style="color: #fbbf24;"><b>Plate:</b> {formatted_plate}</span><br>
                    <span style="color: #fbbf24;"><b>Confidence:</b> {conf:.2f}</span><br>
                    <span style="color: #fbbf24;"><b>Time:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
             </div>
             """, unsafe_allow_html=True)
            else:
                plate_placeholder.info("No plate detected")
                result_placeholder.info("Waiting for detection...")

            # Alert message
            if alert_detected:
                alert_placeholder.markdown(
                    "<div class='alert-box'><strong style='color:red;'>ALERT: Restricted Plate Detected!</strong></div>",
                    unsafe_allow_html=True
                )
            else:
                alert_placeholder.empty()

            # Metrics
            if show_metrics:
                frames = detection_stats['total_frames'] or 1
                avg_det_ms = (detection_stats['detection_time'] / frames) * 1000.0
                avg_ocr_ms = (detection_stats['ocr_time'] / max(1, detection_stats['plates_detected'])) * 1000.0
                metrics_placeholder.markdown(
                    f"""
                    <div class="performance-metrics">
                        <strong>Performance Metrics:</strong><br>
                        FPS: {fps:.1f} | Avg Detection: {avg_det_ms:.1f}ms | Avg OCR: {avg_ocr_ms:.1f}ms<br>
                        Plates Detected: {detection_stats['plates_detected']} | Unique: {detection_stats['unique_plates_detected']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    finally:
        cap.release()
        st.session_state.detection_active = False
        st.rerun()

with tab1:
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Live Detection", disabled=st.session_state.detection_active, use_container_width=True):
        st.session_state.detection_active = True
        st.rerun()

    if col2.button("⏹ Stop Detection", disabled=not st.session_state.detection_active, use_container_width=True):
        st.session_state.detection_active = False
        st.rerun()

    if st.session_state.detection_active:
        run_live_detection()
    else:
        st.info("Click 'Start Live Detection' to begin real-time plate recognition")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image("https://via.placeholder.com/640x480/1e293b/94a3b8?text=Live+Video+Feed", 
                    caption="Live Detection Feed", use_container_width=True)
        with col2:
            st.image("https://via.placeholder.com/320x240/1e293b/94a3b8?text=Processed+Plate", 
                    caption="Processed Plate", use_container_width=True)
            st.info("Waiting for detection...")
    # ==================== SHOW DETECTED HISTORY ====================
    if st.session_state.plate_history:
        st.markdown("## 📸 Detected Plate History")

        # 🗑 Clear history button
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.plate_history = []
            st.success("Detection history cleared!")

        for record in reversed(st.session_state.plate_history[-10:]):  # Show last 10
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(record["image"], caption=record["plate"], use_container_width=True)
            with col2:
                if "SUSPECT" in record["status"]:
                    st.error(f"🚨 {record['status']} | Plate: {record['plate']} | "
                             f"Confidence: {record['confidence']:.2f} | Time: {record['timestamp']}")
                elif "VALID" in record["status"]:
                    st.success(f"✅ {record['status']} | Plate: {record['plate']} | "
                               f"Confidence: {record['confidence']:.2f} | Time: {record['timestamp']}")
                else:
                    st.warning(f"❌ {record['status']} | Plate: {record['plate']} | "
                               f"Confidence: {record['confidence']:.2f} | Time: {record['timestamp']}")
            

# ==================== ANALYTICS TAB ====================
with tab3:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoinround">
            <line x1="18" y1="20" x2="18" y2="10"></line>
            <line x1="12" y1="20" x2="12" y2="4"></line>
            <line x1="6" y1="20" x2="6" y2="14"></line>
        </svg>
        <h2 style="margin-left: 0.75rem; margin-bottom: 0; color: var(--light)">Analytics Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analytics_data:
        st.info("No analytics data available. Perform some detections first.")
    else:
        # Create analytics charts
        fig = create_analytics_charts()
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Could not generate analytics charts. Not enough data.")
        
        # Display statistics
        df = pd.DataFrame(st.session_state.analytics_data)
        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Detections", len(df))
            
            with col2:
                valid_count = df['valid'].sum()
                st.metric("Valid Plates", valid_count)
            
            with col3:
                alert_count = df['alert'].sum() if 'alert' in df.columns else 0
                st.metric("Alert Plates", alert_count)
            
            with col4:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2f}")
            
            # Show data table
            st.subheader("Detection Data")
            st.dataframe(df, use_container_width=True)

# ==================== HISTORY TAB ====================
with tab4:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 99 8 9"></polyline>
        </svg>
        <h2 style="margin-left: 0.75rem; margin-bottom: 0; color: var(--light)">Detection History</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.detection_history:
        st.info("No detection history available. Perform some detections first.")
    else:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_only_valid = st.checkbox("Show only valid plates", value=False)
        with col2:
            show_alerts_only = st.checkbox("Show alerts only", value=False)
        with col3:
            min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.1)
        
        # Filter history
        filtered_history = [
            d for d in st.session_state.detection_history 
            if (not show_only_valid or d.get('valid', False)) and
            (not show_alerts_only or d.get('alert', False)) and
            d.get('confidence', 0) >= min_confidence
        ]
        
        if not filtered_history:
            st.warning("No detections match your filters")
        else:
            # Display as a table for better readability
            history_df = pd.DataFrame(filtered_history)
            history_df['Status'] = history_df.apply(lambda x: 
                '🔴 ALERT' if x.get('alert', False) else ('✅ Valid' if x.get('valid', False) else '❌ Invalid'), axis=1)
            history_df = history_df[['text', 'confidence', 'Status', 'timestamp']]
            history_df.columns = ['Plate Number', 'Confidence', 'Status', 'Timestamp']
            
            st.dataframe(
                history_df,
                use_container_width=True,
                column_config={
                    "Plate Number": st.column_config.TextColumn(
                        "Plate Number",
                        width="medium"
                    ),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        width="small"
                    ),
                    "Timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        format="YYYY-MM-DD HH:mm:ss",
                    )
                }
            )
            
            # Export option
            if st.button("Export History to CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="detection_history.csv",
                    mime="text/csv"
                )


# ==================== SETTINGS TAB ====================
with tab5:
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2  0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 .51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.6.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
        <h2 style="margin-left: 0.75rem; margin-bottom: 0; color: var(--light)">System Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Detection History", use_container_width=True):
            st.session_state.detection_history = []
            st.session_state.analytics_data = []
            st.session_state.recent_detections = {}
            # Clear the detections file
            with open(DETECTIONS_FILE, "w") as f:
                json.dump([], f)
            st.success("Detection history cleared!")
    
    with col2:
        if st.button("Export All Data", use_container_width=True):
            # Create a comprehensive export
            all_data = pd.DataFrame(st.session_state.detection_history)
            csv = all_data.to_csv(index=False)
            st.download_button(
                label="Download Complete Data",
                data=csv,
                file_name="vnpr_complete_data_export.csv",
                mime="text/csv"
            )
    
    st.subheader("System Information")
    st.info("""
    **Vehicle Number Plate Recognition System**
    - Version: 2.0.0
    - Using YOLOv8 for plate detection
    - Using EasyOCR for text recognition
    - Supports Indian license plate format
    - Real-time and image processing modes
    - Anti-duplication system with configurable cooldown
    """)
    
    st.subheader("Model Information")
    st.write("**Detector:** YOLOv8 License Plate Detector")
    st.write("**OCR:** EasyOCR with English language support")
    st.write("**GPU Acceleration:** Enabled" if True else "Disabled")

# Modern footer
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 0.5rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 711a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </svg>
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5"></path>
            <path d="M8.5 8.5v.01"></path>
            <path d="M16 15.5v.01"></path>
            <path d="M12 12v.01"></path>
            <path d="M11 17v.01"></path>
            <path d="M7 14v.01"></path>
        </svg>
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
            <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
        </svg>
    </div>
    <div>AI-Powered Vehicle Number Plate Recognition and Theft Plate Detection | Powered by YOLOv8 and EasyOCR</div>
    <div style="font-size: 0.75rem; margin-top: 0.5rem;">© 2023 Bagalkot University, Jamkhandi</div>
</div>
""", unsafe_allow_html=True)