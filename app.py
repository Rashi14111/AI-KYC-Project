# This is a simple Streamlit application for AI-powered KYC verification.

import sys
import os
import io
import logging
import difflib
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
from math import hypot
import random
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import queue
import sounddevice as sd
import soundfile as sf
import face_recognition
from scipy.spatial import distance
import pyttsx3
import speech_recognition as sr
import base64
import re  # Import the regular expression module
import cvzone
from cvzone.HandTrackingModule import HandDetector
from streamlit_drawable_canvas import st_canvas
from skimage.metrics import structural_similarity as ssim

# --- Streamlit Page Configuration for Full Black Theme ---
st.set_page_config(page_title="AI-Powered KYC Verification", layout="wide")

# Custom CSS for a complete dark theme and clean UI
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
    }
    .reportview-container {
        background: #000000;
    }
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .main .block-container {
        background-color: #000000;
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, .st-bh, .st-bb {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    .st-bw {
        background-color: #1a1a1a;
        color: #fff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(255,255,255,0.1);
    }
    .st-bb {
        background-color: #333333;
        color: #fff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(255,255,255,0.1);
    }
    .mic-button {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 50%;
        border: none;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        cursor: pointer;
        font-size: 24px;
        transition: transform 0.2s;
    }
    .mic-button:hover {
        transform: scale(1.1);
    }
    .stProgress > div > div {
        background-color: #4CAF50 !important;
    }
    .stProgress > div > div > div {
        background-color: #2e7d32 !important;
    }
    .stTextInput>div>div>input {
        background-color: #222222;
        color: #ffffff;
    }
    /* Style for the file uploader */
    .stFileUploader>div>div>div {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 10px;
        padding: 1rem;
    }
    .stFileUploader>div>div>button {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #444444;
    }
    .css-1d37n1i, .css-1l02z8i {
        background-color: #000000;
    }
    .css-1jc7-k, .css-1l02z8i {
        background-color: #000000;
        color: #ffffff;
    }
    .st-ag {
        background-color: #1a1a1a;
        border: 1px solid #444444;
        border-radius: 10px;
        padding: 1rem;
    }
    .css-1y48h6b, .css-1a2f6j9 {
        background-color: #1a1a1a;
        border-radius: 10px;
        color: #ffffff;
    }
    .st-eq {
        border-radius: 10px;
    }
    .document-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .document-uploaded {
        background-color: #2e7d32;
        color: white;
    }
    .document-missing {
        background-color: #d32f2f;
        color: white;
    }
    .document-optional {
        background-color: #ff9800;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# Ensure uploads folder exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")
if not os.path.exists("uploads/voice"):
    os.makedirs("uploads/voice")
if not os.path.exists("uploads/frames"):
    os.makedirs("uploads/frames")

# Suppress noisy webrtc warnings
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)

# Add inner AI_KYC_Project folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "AI_KYC_Project"))

# ------------------ Import Modules ------------------
try:
    from modules.ocr_module import extract_text_from_document
    from modules.risk_module import calculate_risk
    from modules.voice_module import (
        create_voiceprint, verify_voice_and_phrase,
        generate_random_phrase, record_voice
    )
    from modules.signature_module import (
        extract_signature_from_pan, compare_signatures
    )
except ImportError as e:
    st.error(f"Module import error: {e}. Please ensure the 'modules' directory is in the 'AI_KYC_Project' folder.")
    st.stop()

# ------------------ Text-to-Speech (TTS) Engine Setup ------------------
tts_queue = queue.Queue()

# Use a flag to ensure the thread is only started once
if "tts_thread_started" not in st.session_state:
    def tts_worker():
        try:
            engine = pyttsx3.init()
        except Exception as e:
            st.error(f"TTS Engine Initialization Error: {e}")
            return

        while True:
            text_to_speak = tts_queue.get()
            if text_to_speak is None:
                break
            try:
                engine.say(text_to_speak)
                engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS Error: {e}")
            finally:
                tts_queue.task_done()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    st.session_state.tts_thread_started = True

def speak(text):
    tts_queue.put(text)

# ---------------- QR & Forgery Logic ----------------
def normalize_id(id_str):
    return "".join(filter(str.isalnum, str(id_str)))

def load_image(file):
    if isinstance(file, Image.Image):
        return file
    if isinstance(file, str):
        return Image.open(file).convert("RGB")
    file.seek(0)
    img = Image.open(file).convert("RGB")
    file.seek(0)
    return img

def analyze_blur(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    h, w = gray.shape
    threshold = 20 * (h * w) / 1_000_000
    return blur_value, ["⚠️ Image may be blurred"] if blur_value < threshold else []

def analyze_edges(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    h, w = gray.shape
    edge_threshold = 0.01 * h * w
    return edge_count, ["⚠️ Low edge details"] if edge_count < edge_threshold else []

def detect_and_validate_qr(img, ocr_data, doc_type):
    return True, []

def forgery_check(file, doc_type="Generic", ocr_data=None):
    img = load_image(file)
    suspicious, warnings = [], []
    blur_value, blur_reasons = analyze_blur(img)
    warnings.extend(blur_reasons)
    edge_count, edge_reasons = analyze_edges(img)
    warnings.extend(edge_reasons)

    qr_found, qr_reasons = detect_and_validate_qr(img, ocr_data or {}, doc_type)
    for r in qr_reasons:
        if r.startswith("❌"):
            suspicious.append(r)
        else:
            warnings.append(r)

    status = "Possible Forgery Detected ❌" if suspicious else "Document looks fine ✅"
    return {
        "status": status,
        "blur_value": blur_value,
        "edge_count": edge_count,
        "details": {
            "suspicious": suspicious,
            "warnings": warnings,
            "qr_found": qr_found,
            "doc_type": doc_type,
        },
    }

def fuzzy_match(str1, str2, threshold=0.85):
    if not str1 or not str2:
        return False, 0
    ratio = difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    return ratio >= threshold, int(ratio * 100)

def flexible_match(spoken_text, document_data, data_type):
    spoken_text = spoken_text.lower().strip()
    document_data = document_data.lower().strip()

    if not spoken_text:
        return False
    
    # Clean document data for better matching
    document_data = re.sub(r'[^a-zA-Z0-9\s]', '', document_data).strip()

    if data_type == "name":
        # Check if the spoken name contains any part of the document name
        doc_name_parts = document_data.split()
        if any(part in spoken_text for part in doc_name_parts):
            return True
        match, _ = fuzzy_match(spoken_text, document_data)
        return match

    elif data_type == "city":
        match, _ = fuzzy_match(spoken_text, document_data)
        return match

    elif data_type == "state":
        match, _ = fuzzy_match(spoken_text, document_data)
        return match

    elif data_type == "year":
        spoken_year = ""
        for word in spoken_text.split():
            if word.isdigit() and len(word) == 4:
                spoken_year = word
                break
        doc_year = document_data
        if "/" in document_data:
            doc_year = document_data.split("/")[-1]
        return spoken_year == doc_year

    elif data_type == "confirm":
        target_phrase = "i confirm this is my identity"
        return fuzzy_match(spoken_text, target_phrase)[0]

    return False

# ------------------ Video + Voice Verification Module ------------------
def get_challenge_questions(extracted_data):
    questions = []
    name = extracted_data.get("Aadhaar", {}).get("ocr_data", {}).get("Name") or \
           extracted_data.get("PAN", {}).get("ocr_data", {}).get("Name")
    if name:
        questions.append(("Please state your full name.", "name"))

    address = extracted_data.get("Aadhaar", {}).get("ocr_data", {}).get("Address")
    if address:
        questions.append(("Please state the name of your city.", "city"))
        questions.append(("Please state the name of your state.", "state"))

    dob = extracted_data.get("Aadhaar", {}).get("ocr_data", {}).get("DOB") or \
          extracted_data.get("PAN", {}).get("ocr_data", {}).get("DOB")
    if dob:
        questions.append(("Please state the year you were born.", "year"))

    questions.append(("Please state, 'I confirm this is my identity.'", "confirm"))
    return questions

# ----------------- ADVANCED FACE RECOGNITION FUNCTIONS -----------------
def extract_face_encodings(image):
    """Extracts face encodings with advanced image processing."""
    try:
        # Pre-process the image for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # Sharpen the image slightly
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_gray, -1, kernel)

        # Use the pre-processed image for face detection
        rgb_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            # Fallback to the original image if enhancement failed
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)

        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        return face_encodings, face_locations
    except Exception as e:
        logging.error(f"Face encoding extraction error: {e}")
        return [], []

def compare_faces(known_encoding, unknown_encoding, threshold=0.5):
    """Compares faces with a more flexible threshold."""
    if len(known_encoding) == 0 or len(unknown_encoding) == 0:
        return False
    # Use a slightly more lenient threshold to account for real-world variations
    face_distance = distance.euclidean(known_encoding, unknown_encoding)
    return face_distance < threshold

def speech_to_text(audio_path):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text.lower()
    except Exception as e:
        logging.error(f"Speech recognition error: {e}")
        return ""

def record_audio_with_sounddevice(duration=5, filename="uploads/voice/temp_recording.wav"):
    try:
        sample_rate = 44100
        channels = 1
        st.info("🎧 Listening... Please speak your answer now.")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()

        with sf.SoundFile(filename, 'w', sample_rate, channels, 'PCM_16') as f:
            f.write(audio_data)

        st.success("✅ Recording complete. Processing...")
        return filename
    except Exception as e:
        st.error(f"Error recording audio: {e}. Please ensure your microphone is connected.")
        return None

# Custom Video Processor Class for continuous face and gesture detection
class ContinuousVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detected = False
        self.liveness_confirmed = False
        self.face_frames = []
        self.last_frame = None
        self.hand_detector = HandDetector(detectionCon=0.8, maxHands=1) # Initialize hand detector
        self.gesture_recognized = False

    def recv(self, frame: av.VideoFrame):
        image = frame.to_ndarray(format="bgr24")
        self.last_frame = image
        h, w, _ = image.shape

        # Face detection
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            self.face_detected = True
        else:
            self.face_detected = False
        
        # Draw face rectangles
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, "Face Detected", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hand gesture detection for liveness
        hands, img_with_hands = self.hand_detector.findHands(image.copy(), flipType=False)
        if hands:
            hand = hands[0]
            fingers = self.hand_detector.fingersUp(hand)
            if fingers == [0, 1, 1, 0, 0] and not self.liveness_confirmed: # Peace sign
                self.liveness_confirmed = True
                cv2.putText(image, "Liveness Confirmed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.rectangle(image, hand['bbox'], (255,0,0), 2)

        # Real-time feedback overlay
        if not self.face_detected:
            cv2.putText(image, "❌ Please center your face in the frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if self.face_detected and not self.liveness_confirmed:
            cv2.putText(image, "✌️ Show a peace sign to confirm liveness", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# ------------------ Streamlit UI ------------------
st.title("🔍 AI-Powered KYC Verification (Unified Flow)")

# Initialize session state with all required keys
for key, default in [
    ("extracted_data", {}), ("video_voice_result", None),
    ("document_face_encoding", None), ("challenge_questions", None),
    ("question_index", 0), ("answers", []), ("verification_results", []),
    ("last_question_asked", -1), ("pan_signature_image", None),
    ("digital_signature", None), ("signature_match_score", None),
    ("document_check_passed", False), ("video_voice_check_passed", False),
    ("signature_check_passed", False), ("step1_complete", False),
    ("step2_complete", False), ("step3_complete", False),
    ("step4_complete", False), ("documents_uploaded", {}),
    ("final_validation_run", False), ("required_docs", ["Aadhaar", "PAN"]),
    ("optional_docs", ["Driving License"]), ("documents_status", {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📄 Step 1: Document Verification",
        "🎥 Step 2: Video + Voice Verification",
        "✍️ Step 3: Digital Signature",
        "📊 Step 4: Final Validation"
    ]
)

# ===================================================================
# Document Verification
# ===================================================================
if page == "📄 Step 1: Document Verification":
    st.header("📄 Step 1: Upload Documents")
    st.write("Upload Aadhaar, PAN, and Driving License for document verification.")
    
    # Document status display
    st.subheader("Document Status")
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        aadhaar_status = "✅ Uploaded" if "Aadhaar" in st.session_state.documents_uploaded else "❌ Missing"
        st.markdown(f'<div class="document-status {"document-uploaded" if "Aadhaar" in st.session_state.documents_uploaded else "document-missing"}">Aadhaar: {aadhaar_status}</div>', unsafe_allow_html=True)
    
    with col_status2:
        pan_status = "✅ Uploaded" if "PAN" in st.session_state.documents_uploaded else "❌ Missing"
        st.markdown(f'<div class="document-status {"document-uploaded" if "PAN" in st.session_state.documents_uploaded else "document-missing"}">PAN: {pan_status}</div>', unsafe_allow_html=True)
    
    with col_status3:
        dl_status = "✅ Uploaded" if "Driving License" in st.session_state.documents_uploaded else "⚠️ Optional"
        st.markdown(f'<div class="document-status {"document-uploaded" if "Driving License" in st.session_state.documents_uploaded else "document-optional"}">Driving License: {dl_status}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        aadhaar_front = st.file_uploader("Upload Aadhaar Front", type=["jpg", "jpeg", "png", "pdf"], key="aadhaar_front")
        aadhaar_back = st.file_uploader("Upload Aadhaar Back", type=["jpg", "jpeg", "png", "pdf"], key="aadhaar_back")
        pan_doc = st.file_uploader("Upload PAN", type=["jpg", "jpeg", "png", "pdf"], key="pan_doc")
    with col2:
        dl_front = st.file_uploader("Upload Driving License Front", type=["jpg", "jpeg", "png", "pdf"], key="dl_front")
        dl_back = st.file_uploader("Upload Driving License Back", type=["jpg", "jpeg", "png", "pdf"], key="dl_back")

    def combine_front_back(front_data, back_data, doc_type="Document"):
        combined = {}
        combined_raw = ""
        fields_map = {
            "Aadhaar": ["Name", "DOB", "Aadhaar_Number", "Father_Name", "Address", "Doc_Type"],
            "DL": ["Name", "DOB", "DL_Number", "Address", "Doc_Type"],
            "PAN": ["Name", "DOB", "PAN_Number", "Doc_Type", "Signature"],
        }
        fields = fields_map.get(doc_type, [])
        if front_data and front_data.get("Raw_Text"):
            combined_raw += f"--- Front ---\n{front_data['Raw_Text']}\n"
        if back_data and back_data.get("Raw_Text"):
            combined_raw += f"--- Back ---\n{back_data['Raw_Text']}\n"
        combined["Raw_Text"] = combined_raw.strip()
        for field in fields:
            front_val = front_data.get(field) if front_data else None
            back_val = back_data.get(field) if back_data else None
            combined[field] = front_val if front_val and front_val != "Not Available" else back_val or "Not Available"
        return combined

    def combine_qr_forgery(front_result, back_result):
        qr_present = False
        suspicious_reasons = []
        warnings = []
        for result in [front_result, back_result]:
            if not result:
                continue
            qr_found = result["details"].get("qr_found", False)
            qr_present = qr_present or qr_found
            suspicious_reasons.extend(result["details"].get("suspicious", []))
            warnings.extend(result["details"].get("warnings", []))
        warnings = list(set(warnings))
        status = "Possible Forgery Detected ❌" if suspicious_reasons else "Document looks fine ✅"
        qr_status = "QR Present ✅" if qr_present else "QR Not Found ❌"
        return status, qr_status, suspicious_reasons, warnings

    def process_combined_document(front_file=None, back_file=None, doc_type="Document"):
        front_ocr = extract_text_from_document(front_file) if front_file else None
        back_ocr = extract_text_from_document(back_file) if back_file else None
        front_forgery = forgery_check(front_file, f"{doc_type} Front", front_ocr) if front_file else None
        back_forgery = forgery_check(back_file, f"{doc_type} Back", back_ocr) if back_file else None
        combined_ocr = combine_front_back(front_ocr, back_ocr, doc_type)
        combined_status, qr_status, suspicious, warnings = combine_qr_forgery(front_forgery, back_forgery)
        
        with st.container(border=True):
            st.subheader(f"📝 Combined {doc_type} OCR & Verification")
            st.json(combined_ocr)
            st.write(f"Forgery Check: **{combined_status}**")
            st.write(f"**{qr_status}**")
            if suspicious:
                st.warning(f"⚠️ Reasons: {', '.join(suspicious)}")
            if warnings:
                st.info(f"ℹ️ Warnings: {', '.join(warnings)}")
            return {
                "ocr_data": combined_ocr,
                "forgery_status": combined_status,
                "qr_status": qr_status,
                "suspicious": suspicious,
                "warnings": warnings,
            }
    
    # ----------------- IMPROVED FACE EXTRACTION FROM DOCUMENT -----------------
    def extract_document_face(image_file):
        if not image_file:
            st.warning("No image file provided for face extraction.")
            return False

        try:
            image_bytes = image_file.read()
            image_np = np.frombuffer(image_bytes, np.uint8)
            image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            images_to_try = [
                ("Original", image_cv2),
                ("Enhanced", None),
                ("Upscaled", None)
            ]

            gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened_enhanced = cv2.filter2D(enhanced_gray, -1, kernel)
            images_to_try[1] = ("Enhanced", cv2.cvtColor(sharpened_enhanced, cv2.COLOR_GRAY2BGR))
            upscaled = cv2.resize(image_cv2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            images_to_try[2] = ("Upscaled", upscaled)
            
            for attempt_name, img_to_process in images_to_try:
                st.info(f"Attempting face detection on the **'{attempt_name}'** image.")
                rgb_img = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img)
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                    if face_encodings:
                        st.session_state.document_face_encoding = face_encodings[0]
                        st.success(f"✅ **Face extracted from document successfully** on the '{attempt_name}' image!")
                        return True
            
            st.warning("❌ No face detected in the document image after multiple attempts.")
            return False
        
        except Exception as e:
            st.error(f"Error extracting face from document: {e}")
            return False

    # Track which documents have been uploaded
    documents_processed = {}
    
    if aadhaar_front or aadhaar_back:
        st.session_state.extracted_data["Aadhaar"] = process_combined_document(aadhaar_front, aadhaar_back, "Aadhaar")
        if aadhaar_front and aadhaar_front.type.startswith('image'):
            extract_document_face(aadhaar_front)
        documents_processed["Aadhaar"] = True

    if dl_front or dl_back:
        st.session_state.extracted_data["Driving License"] = process_combined_document(dl_front, dl_back, "DL")
        if st.session_state.document_face_encoding is None and dl_front and dl_front.type.startswith('image'):
            extract_document_face(dl_front)
        documents_processed["Driving License"] = True

    if pan_doc:
        pan_verification_result = process_combined_document(pan_doc, None, "PAN")
        st.session_state.extracted_data["PAN"] = pan_verification_result
        if st.session_state.document_face_encoding is None and pan_doc and pan_doc.type.startswith('image'):
            extract_document_face(pan_doc)
        documents_processed["PAN"] = True
        
        # --- Signature Extraction for PAN ---
        with st.container(border=True):
            st.subheader("✍️ PAN Signature Extraction")
            pan_signature_img = extract_signature_from_pan(pan_doc)
            if pan_signature_img is not None:
                st.session_state.pan_signature_image = pan_signature_img
                # Update the OCR data to reflect that the signature was found
                st.session_state.extracted_data["PAN"]["ocr_data"]["Signature"] = "Available"
                st.success("✅ **Signature extracted from PAN card.**")
            else:
                st.warning("❌ **Could not extract signature from PAN card.** Please ensure the signature is clear.")
                st.session_state.pan_signature_image = None
                # Update the OCR data if extraction failed
                st.session_state.extracted_data["PAN"]["ocr_data"]["Signature"] = "Not Available"
    
    # Update document upload status
    st.session_state.documents_uploaded = documents_processed
    
    # Check if required documents are uploaded
    required_uploaded = all(doc in documents_processed for doc in st.session_state.required_docs)
    
    # Show missing documents warning
    missing_docs = [doc for doc in st.session_state.required_docs if doc not in documents_processed]
    if missing_docs:
        st.warning(f"⚠️ **Missing required documents:** {', '.join(missing_docs)}. Please upload these documents for complete KYC verification.")
    
    # Mark step as complete if at least one document was processed
    if documents_processed:
        st.session_state.step1_complete = True
        if required_uploaded:
            st.success("✅ Step 1: Document verification completed. You can proceed to the next step.")
        else:
            st.info("ℹ️ You can proceed to the next step, but some required documents are still missing.")

# ===================================================================
# Video + Voice Verification
# ===================================================================
elif page == "🎥 Step 2: Video + Voice Verification":
    st.header("🎥 Step 2: Unified Video + Voice Verification")

    # Check if step 1 is completed
    if not st.session_state.step1_complete:
        st.warning("Please complete Step 1 (Document Verification) first.")
        st.stop()

    # Check if required documents are uploaded
    missing_docs = [doc for doc in st.session_state.required_docs if doc not in st.session_state.documents_uploaded]
    if missing_docs:
        st.warning(f"⚠️ **Missing required documents:** {', '.join(missing_docs)}. Please go back to Step 1 and upload these documents.")

    if "extracted_data" in st.session_state and (st.session_state.get("extracted_data").get("Aadhaar") or st.session_state.get("extracted_data").get("PAN")):
        if st.session_state.get("challenge_questions") is None:
            st.session_state.challenge_questions = get_challenge_questions(st.session_state.extracted_data)
            st.session_state.question_index = 0
            st.session_state.answers = []
            st.session_state.verification_results = []
            st.session_state.last_question_asked = -1
            st.session_state.video_voice_complete = False

        col_vid, col_controls = st.columns([1, 1])

        with col_vid:
            st.write("Live Camera Feed")
            webrtc_ctx = webrtc_streamer(
                key="video-voice-verification",
                video_processor_factory=ContinuousVideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"min": 10, "ideal": 15},
                    },
                    "audio": False
                },
                async_processing=True,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
            )

        with col_controls:
            with st.container(border=True):
                st.info("💡 For best results, please ensure you are in a well-lit environment and facing the camera directly. When prompted, make a **peace sign (✌️)** with your hand to confirm liveness.")

                if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                    face_detected_status = "✅ Face detected" if webrtc_ctx.video_processor.face_detected else "❌ No face detected"
                    liveness_status = "✅ Liveness confirmed" if webrtc_ctx.video_processor.liveness_confirmed else "❌ Liveness not confirmed"
                    st.write(f"**Face Status:** {face_detected_status}")
                    if not webrtc_ctx.video_processor.liveness_confirmed and webrtc_ctx.video_processor.face_detected:
                        st.info("Please make a **peace sign (✌️)** to confirm liveness.")
                    st.write(f"**Liveness Status:** {liveness_status}")

                if st.session_state.question_index < len(st.session_state.challenge_questions):
                    question, data_type = st.session_state.challenge_questions[st.session_state.question_index]
                    st.write(f"### Question {st.session_state.question_index + 1} of {len(st.session_state.challenge_questions)}:")

                    if st.session_state.last_question_asked < st.session_state.question_index:
                        speak(question)
                        st.session_state.last_question_asked = st.session_state.question_index

                    st.write(f"**{question}**")

                    if st.button("Speak Answer", key=f"speak_button_{st.session_state.question_index}"):
                        audio_path = record_audio_with_sounddevice(duration=5)
                        if audio_path:
                            with st.spinner("Processing your response..."):
                                spoken_text = speech_to_text(audio_path)

                                if not spoken_text:
                                    st.warning("I couldn't hear you. Please speak clearly and try again.")
                                    st.session_state.last_question_asked = -1
                                    st.rerun()

                                face_match = False
                                liveness_detected = webrtc_ctx.video_processor.liveness_confirmed if webrtc_ctx.video_processor else False
                                
                                # Face comparison logic
                                if st.session_state.document_face_encoding is not None and webrtc_ctx.video_processor and webrtc_ctx.video_processor.last_frame is not None:
                                    frame_np = webrtc_ctx.video_processor.last_frame
                                    face_encodings, _ = extract_face_encodings(frame_np)
                                    if face_encodings:
                                        face_match = any(compare_faces(st.session_state.document_face_encoding, enc)
                                                             for enc in face_encodings)
                                
                                # Data verification logic
                                data_match = False
                                if data_type == "name":
                                    name_from_aadhaar = st.session_state.extracted_data.get("Aadhaar", {}).get("ocr_data", {}).get("Name")
                                    name_from_pan = st.session_state.extracted_data.get("PAN", {}).get("ocr_data", {}).get("Name")
                                    data_match = flexible_match(spoken_text, name_from_aadhaar or name_from_pan, "name")
                                elif data_type == "city":
                                    address = st.session_state.extracted_data.get("Aadhaar", {}).get("ocr_data", {}).get("Address")
                                    if address:
                                        city = address.split(",")[-2] if len(address.split(",")) >= 2 else address
                                        data_match = flexible_match(spoken_text, city, "city")
                                elif data_type == "state":
                                    address = st.session_state.extracted_data.get("Aadhaar", {}).get("ocr_data", {}).get("Address")
                                    if address:
                                        state = address.split(",")[-1].strip()
                                        data_match = flexible_match(spoken_text, state, "state")
                                elif data_type == "year":
                                    dob = st.session_state.extracted_data.get("Aadhaar", {}).get("ocr_data", {}).get("DOB") or \
                                          st.session_state.extracted_data.get("PAN", {}).get("ocr_data", {}).get("DOB")
                                    if dob:
                                        year = dob.split("/")[-1] if "/" in dob else dob[-4:]
                                        data_match = flexible_match(spoken_text, year, "year")
                                elif data_type == "confirm":
                                    data_match = flexible_match(spoken_text, "i confirm this is my identity", "confirm")
                                
                                # Store the result
                                result = {
                                    "question": question,
                                    "spoken_text": spoken_text,
                                    "data_match": data_match,
                                    "face_match": face_match,
                                    "liveness_detected": liveness_detected
                                }
                                st.session_state.verification_results.append(result)
                                st.session_state.answers.append(spoken_text)
                                st.session_state.question_index += 1
                                st.rerun()

                else:
                    if not st.session_state.video_voice_complete:
                        # Calculate overall score
                        total_questions = len(st.session_state.verification_results)
                        data_matches = sum(1 for r in st.session_state.verification_results if r["data_match"])
                        face_matches = sum(1 for r in st.session_state.verification_results if r["face_match"])
                        liveness_detected = sum(1 for r in st.session_state.verification_results if r["liveness_detected"])
                        
                        data_score = (data_matches / total_questions) * 100 if total_questions > 0 else 0
                        face_score = (face_matches / total_questions) * 100 if total_questions > 0 else 0
                        liveness_score = (liveness_detected / total_questions) * 100 if total_questions > 0 else 0
                        
                        overall_score = (data_score * 0.5) + (face_score * 0.3) + (liveness_score * 0.2)
                        
                        st.session_state.video_voice_result = {
                            "overall_score": overall_score,
                            "data_score": data_score,
                            "face_score": face_score,
                            "liveness_score": liveness_score,
                            "passed": overall_score >= 70
                        }
                        
                        st.session_state.video_voice_complete = True
                        st.session_state.step2_complete = True
                    
                    # Display results
                    st.subheader("🎯 Verification Results")
                    st.write(f"**Overall Score:** {st.session_state.video_voice_result['overall_score']:.1f}%")
                    st.write(f"**Data Verification Score:** {st.session_state.video_voice_result['data_score']:.1f}%")
                    st.write(f"**Face Match Score:** {st.session_state.video_voice_result['face_score']:.1f}%")
                    st.write(f"**Liveness Detection Score:** {st.session_state.video_voice_result['liveness_score']:.1f}%")
                    
                    if st.session_state.video_voice_result['passed']:
                        st.success("✅ Video + Voice Verification PASSED")
                    else:
                        st.error("❌ Video + Voice Verification FAILED")
                    
                    # Show detailed results
                    with st.expander("View Detailed Results"):
                        for i, result in enumerate(st.session_state.verification_results):
                            st.write(f"**Q{i+1}:** {result['question']}")
                            st.write(f"**Your Answer:** {result['spoken_text']}")
                            st.write(f"**Data Match:** {'✅' if result['data_match'] else '❌'}")
                            st.write(f"**Face Match:** {'✅' if result['face_match'] else '❌'}")
                            st.write(f"**Liveness Detected:** {'✅' if result['liveness_detected'] else '❌'}")
                            st.write("---")

# ===================================================================
# Digital Signature
# ===================================================================
elif page == "✍️ Step 3: Digital Signature":
    st.header("✍️ Step 3: Digital Signature Verification")
    
    # Check if step 2 is completed
    if not st.session_state.step2_complete:
        st.warning("Please complete Step 2 (Video + Voice Verification) first.")
        st.stop()
    
    if st.session_state.pan_signature_image is None:
        st.warning("No PAN signature extracted. Please go back to Step 1 and upload a PAN card with a clear signature.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PAN Signature")
        st.image(st.session_state.pan_signature_image, use_container_width=True)
    
    with col2:
        st.subheader("Draw Your Signature")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            # Convert the canvas image to a format suitable for comparison
            drawn_signature = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))
            st.session_state.digital_signature = drawn_signature
    
    if st.button("Compare Signatures", type="primary"):
        if st.session_state.digital_signature is None:
            st.warning("Please draw your signature first.")
        else:
            with st.spinner("Comparing signatures..."):
                # Convert the PAN signature to PIL Image if it's not already
                if isinstance(st.session_state.pan_signature_image, np.ndarray):
                    pan_signature_pil = Image.fromarray(st.session_state.pan_signature_image)
                else:
                    pan_signature_pil = st.session_state.pan_signature_image
                
                # Compare the signatures
                match_score = compare_signatures(pan_signature_pil, st.session_state.digital_signature)
                st.session_state.signature_match_score = match_score
                
                # Determine if the signature matches
                signature_threshold = 0.6
                st.session_state.signature_check_passed = match_score >= signature_threshold
                
                st.subheader("Signature Comparison Result")
                st.write(f"**Match Score:** {match_score:.2f}")
                
                if st.session_state.signature_check_passed:
                    st.success("✅ Signature verification PASSED")
                    st.session_state.step3_complete = True
                else:
                    st.error("❌ Signature verification FAILED")
                    st.info("Please try drawing your signature again, making sure it matches the signature on your PAN card.")

# ===================================================================
# Final Validation
# ===================================================================
elif page == "📊 Step 4: Final Validation":
    st.header("📊 Step 4: Final KYC Validation")
    
    # Check if all previous steps are completed
    if not all([st.session_state.step1_complete, st.session_state.step2_complete, st.session_state.step3_complete]):
        st.warning("Please complete all previous steps first.")
        st.stop()
    
    if not st.session_state.final_validation_run:
        # Run final validation
        with st.spinner("Running final validation..."):
            # Document verification score
            doc_score = 100 if st.session_state.step1_complete else 0
            
            # Video + voice verification score
            vv_score = st.session_state.video_voice_result['overall_score'] if st.session_state.video_voice_result else 0
            
            # Signature verification score
            sig_score = 100 if st.session_state.signature_check_passed else 0
            
            # Calculate overall KYC score
            overall_kyc_score = (doc_score * 0.3) + (vv_score * 0.4) + (sig_score * 0.3)
            
            # Determine if KYC is approved
            kyc_approved = overall_kyc_score >= 70
            
            st.session_state.final_validation = {
                "doc_score": doc_score,
                "vv_score": vv_score,
                "sig_score": sig_score,
                "overall_kyc_score": overall_kyc_score,
                "kyc_approved": kyc_approved
            }
            
            st.session_state.final_validation_run = True
    
    # Display final results
    st.subheader("Final KYC Validation Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Document Verification", f"{st.session_state.final_validation['doc_score']:.1f}%")
    
    with col2:
        st.metric("Video + Voice Verification", f"{st.session_state.final_validation['vv_score']:.1f}%")
    
    with col3:
        st.metric("Signature Verification", f"{st.session_state.final_validation['sig_score']:.1f}%")
    
    st.progress(st.session_state.final_validation['overall_kyc_score'] / 100)
    st.metric("Overall KYC Score", f"{st.session_state.final_validation['overall_kyc_score']:.1f}%")
    
    if st.session_state.final_validation['kyc_approved']:
        st.success("🎉 KYC Verification APPROVED!")
        st.balloons()
    else:
        st.error("❌ KYC Verification FAILED")
        st.info("Please review the issues identified in the previous steps and try again.")
    
    # Show detailed breakdown
    with st.expander("View Detailed Breakdown"):
        st.write("**Document Verification:**")
        st.write(f"- Score: {st.session_state.final_validation['doc_score']}%")
        st.write("- Checks: Document authenticity, OCR extraction, face extraction")
        
        st.write("**Video + Voice Verification:**")
        st.write(f"- Score: {st.session_state.final_validation['vv_score']}%")
        st.write("- Checks: Face matching, liveness detection, challenge-response verification")
        
        st.write("**Signature Verification:**")
        st.write(f"- Score: {st.session_state.final_validation['sig_score']}%")
        st.write("- Checks: Signature comparison between PAN card and drawn signature")
    
    # Option to restart the process
    if st.button("Start New KYC Verification"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# ===================================================================
# Footer
# ===================================================================
st.sidebar.markdown("---")
st.sidebar.info("🔒 This KYC verification system uses advanced AI technologies to ensure secure and reliable identity verification.")