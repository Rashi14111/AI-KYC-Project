# 🛡️ AI-Powered KYC Verification System  

An **end-to-end AI-driven KYC (Know Your Customer) system** built with **Streamlit, OpenCV, TensorFlow, and Speech Recognition**.  
It automates document verification, face recognition, liveness detection, voice confirmation, and signature verification — ensuring secure identity validation.  

## 🚀 Features

- 📄 **Document OCR & Forgery Detection** – Extracts text and checks authenticity of Aadhaar, PAN, Driving License  
- 🧑‍🦰 **Face Recognition** – Matches live webcam feed with ID photo  
- ✌️ **Liveness Detection** – Prevents spoofing using peace sign gesture  
- 🎙️ **Voice Q&A** – Challenge-response verification with speech recognition  
- ✍️ **Signature Verification** – Compares PAN card signature with a drawn digital signature  
- 📊 **Final KYC Scoring** – Aggregates all checks to approve/reject identity  


## 🛠️ Tech Stack

- **Languages:** Python  
- **Libraries:** Streamlit, OpenCV, FaceRecognition, SpeechRecognition, NumPy, Pandas, cvzone  
- **Deployment:** Streamlit Cloud / Docker  

## 📂 Project Structure

AI-KYC-Project/
│── app.py # Main Streamlit application
│── modules/
│ ├── ocr_module.py # OCR extraction
│ ├── forgery_module.py # Forgery detection
│ ├── voice_module.py # Voice verification
│ ├── signature_module.py # Signature extraction + comparison
│ └── risk_module.py # Risk scoring
│── requirements.txt # Dependencies
│── Dockerfile # Deployment config
│── uploads/ # Temporary uploads
│── images/ # Screenshots



## 📸 Screenshots

STEP1 :  Document Verification
 
![01_upload_and_extract_details](https://github.com/user-attachments/assets/54cf933d-cc21-492c-b885-cf7dcedea03f)

STEP 2 : Face Recognition & Liveness  

![02_Video voiceverification](https://github.com/user-attachments/assets/aa225361-d659-4cda-84b8-13d8a58e774f)

STEP 3: Signature Verification 

![03_DigitalSig](https://github.com/user-attachments/assets/f8a1ea1f-b3cd-484d-896d-137c45161c80)

STEP 4 :Final KYC Score  

![04_final_validation](https://github.com/user-attachments/assets/b9097152-5b37-495c-8005-99a5ee8c33d3)


## ▶️ Run Locally

```bash
git clone https://github.com/Rashi14111/AI-KYC-Project.git
cd AI-KYC-Project
pip install -r requirements.txt
streamlit run app.py
