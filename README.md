# 🛡️ AI-Powered KYC Verification System  

An **end-to-end AI-driven KYC (Know Your Customer) system** built with **Streamlit, OpenCV, TensorFlow, and Speech Recognition**.  
It automates document verification, face recognition, liveness detection, voice confirmation, and signature verification — ensuring secure identity validation.  

---

## 🚀 Features
- 📄 **Document OCR & Forgery Detection** – Extracts text and checks authenticity of Aadhaar, PAN, Driving License  
- 🧑‍🦰 **Face Recognition** – Matches live webcam feed with ID photo  
- ✌️ **Liveness Detection** – Prevents spoofing using peace sign gesture  
- 🎙️ **Voice Q&A** – Challenge-response verification with speech recognition  
- ✍️ **Signature Verification** – Compares PAN card signature with a drawn digital signature  
- 📊 **Final KYC Scoring** – Aggregates all checks to approve/reject identity  

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** Streamlit, OpenCV, FaceRecognition, SpeechRecognition, NumPy, Pandas, cvzone  
- **Deployment:** Streamlit Cloud / Docker   

---

## 📂 Project Structure
AI-KYC-Project/

AI-KYC-Project/
│── app.py                 # Main Streamlit application
│── modules/
│   ├── ocr_module.py       # OCR extraction
│   ├── forgery_module.py   # Forgery detection
│   ├── voice_module.py     # Voice verification
│   ├── signature_module.py # Signature extraction + comparison
│   └── risk_module.py      # Risk scoring
│── requirements.txt        # Dependencies
│── Dockerfile              # Deployment config
│── uploads/                # Temporary uploads

---

## ▶️ Run Locally
git clone https://github.com/Rashi14111/AI-KYC-Project.git
cd AI-KYC-Project
pip install -r requirements.txt
streamlit run app.py

📸 Screenshots

Document Verification -  https://github.com/Rashi14111/AI-KYC-Project/blob/main/01_upload%20and%20Extractdetails%20.jpg
Face Recognition & Liveness - https://github.com/Rashi14111/AI-KYC-Project/blob/main/02_Video%26voiceverification.jpg
Signature Verification  - https://github.com/Rashi14111/AI-KYC-Project/blob/main/02_Video%26voiceverification.jpg
Final KYC Score - https://github.com/Rashi14111/AI-KYC-Project/blob/main/04_final_validation.jpg
