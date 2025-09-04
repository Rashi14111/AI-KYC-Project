# modules/video_voice_module.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import random
import time
import speech_recognition as sr
import numpy as np
import librosa

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --------------------
# Random Challenge
# --------------------
def generate_challenge():
    challenges = [
        {"type": "phrase", "text": "Please say: 'My voice confirms my identity'"},
        {"type": "phrase", "text": "Please say: 'Secure banking starts with me'"},
        {"type": "movement", "text": "Turn your head LEFT"},
        {"type": "movement", "text": "Turn your head RIGHT"},
        {"type": "movement", "text": "Blink your eyes twice"}
    ]
    return random.choice(challenges)


# --------------------
# Voiceprint Creation
# --------------------
def create_voiceprint(audio_file):
    y, sr_rate = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def verify_phrase(audio_file, expected_phrase):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            spoken = recognizer.recognize_google(audio)
            return expected_phrase.lower() in spoken.lower(), spoken
        except:
            return False, ""


# --------------------
# Video Processor
# --------------------
class VideoVoiceProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detected = False
        self.start_time = time.time()
        self.challenge_result = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            self.face_detected = True
        else:
            self.face_detected = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------
# Streamlit Page
# --------------------
def start_video_voice_verification():
    st.header("🎥 Video + Voice KYC Verification")

    # Step 1: Assign challenge if not already
    if "challenge" not in st.session_state:
        st.session_state["challenge"] = generate_challenge()
        st.session_state["challenge_passed"] = False

    challenge = st.session_state["challenge"]
    st.subheader(f"📝 Challenge: {challenge['text']}")

    # Step 2: Start Video+Audio Stream
    ctx = webrtc_streamer(
        key="video-voice",
        video_processor_factory=VideoVoiceProcessor,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

    # Step 3: Voice Verification (if phrase challenge)
    if challenge["type"] == "phrase":
        audio_file = st.file_uploader("Upload recorded phrase (WAV/MP3)", type=["wav", "mp3"])
        if audio_file:
            match, spoken = verify_phrase(audio_file, challenge["text"].replace("Please say: ", ""))
            if match:
                st.success(f"✅ Phrase verified: {spoken}")
                st.session_state["challenge_passed"] = True
            else:
                st.error(f"❌ Phrase mismatch. Detected: {spoken}")

    # Step 4: Movement Verification (basic example)
    if challenge["type"] == "movement":
        if ctx.video_processor and ctx.video_processor.face_detected:
            st.info("✅ Face detected - follow movement instruction manually")
            # (TODO: advanced: detect head orientation with landmarks)
            st.session_state["challenge_passed"] = True

    # Step 5: Final Decision
    if st.session_state.get("challenge_passed", False):
        st.success("🎉 KYC Verification PASSED")
    else:
        st.warning("⏳ Waiting for challenge completion...")
