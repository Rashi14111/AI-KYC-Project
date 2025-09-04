# modules/voice_module.py
import random
import speech_recognition as sr
import librosa
import numpy as np
import tempfile
import soundfile as sf

def generate_random_phrase():
    phrases = [
        "My voice confirms my identity",
        "KYC verification in progress",
        "Secure banking starts with me",
        "I confirm this is my account"
    ]
    return random.choice(phrases)

def create_voiceprint(audio_file):
    y, sr_rate = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def verify_voice_and_phrase(recorded_file, reference_voiceprint, expected_phrase, spoken_phrase):
    if expected_phrase.lower() not in spoken_phrase.lower():
        return False, "❌ Phrase mismatch. Please repeat correctly."
    new_print = create_voiceprint(recorded_file)
    similarity = np.dot(reference_voiceprint, new_print) / (
        np.linalg.norm(reference_voiceprint) * np.linalg.norm(new_print)
    )
    if similarity > 0.75:
        return True, f"✅ Voice verified with similarity {similarity:.2f}"
    else:
        return False, f"❌ Voice not matching. Similarity {similarity:.2f}"

def record_voice(duration=5):
    import sounddevice as sd
    from scipy.io.wavfile import write
    import io

    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    temp_file = io.BytesIO()
    write(temp_file, fs, recording)
    temp_file.seek(0)
    return temp_file

def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        try:
            return r.recognize_google(audio)
        except:
            return ""
