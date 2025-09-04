import os
import re
import google.generativeai as genai
from config.gemini_client import MODEL

# Configure Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(MODEL)


def clean_text(raw_text: str) -> str:
    """Remove markdown formatting and normalize spaces."""
    text = raw_text.replace("**", "").replace("*", "")
    return text.strip()


def parse_fields(raw_text: str):
    """
    Parse OCR raw text to extract Name, DOB, ID_Number, Doc_Type, Father_Name, Address.
    Also adjust ID field label based on document type.
    """
    text = clean_text(raw_text)

    doc_type = "Unknown"
    id_num = "Not Found"
    id_label = "ID_Number"  # Default key
    name = "Not Found"
    dob = "Not Found"
    father_name = "N/A"
    address = "N/A"

    # ---------- Detect Document Type ----------
    aadhaar = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
    pan = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    dl = re.search(r"\b[A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[0-9]{4,11}\b", text)
    passport = re.search(r"\b[A-Z][0-9]{7}\b", text)

    if aadhaar:
        doc_type = "Aadhaar"
        id_num = aadhaar.group()
        id_label = "Aadhaar_Number"
    elif pan:
        doc_type = "PAN"
        id_num = pan.group()
        id_label = "PAN_Number"
    elif dl:
        doc_type = "Driving License"
        id_num = dl.group()
        id_label = "DL_Number"
    elif passport:
        doc_type = "Passport"
        id_num = passport.group()
        id_label = "Passport_Number"

    # ---------- Line by line extraction ----------
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("dob:") or "date of birth" in line.lower():
            dob = line.split(":", 1)[1].strip()
        elif line.lower().startswith("father's name:") or line.lower().startswith("s/o") or line.lower().startswith("d/o") or line.lower().startswith("w/o"):
            father_name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("address:"):
            address = line.split(":", 1)[1].strip()

    # Build result dictionary
    result = {
        "Name": name,
        "DOB": dob,
        id_label: id_num,   # Dynamically add correct ID field name
        "Doc_Type": doc_type,
        "Father_Name": father_name,
    }

    # Only include Address if not PAN
    if doc_type != "PAN":
        result["Address"] = address

    return result


def extract_text_from_document(file):
    """
    Uses Google Gemini API to extract textual content from documents (images or PDFs).
    Returns OCR text + structured fields with proper labels.
    """
    content = file.read()
    mime_type = "application/pdf" if file.name.lower().endswith(".pdf") else "image/jpeg"

    result = model.generate_content(
        [
            "Extract all text from this document clearly. "
            "Focus on Name, DOB, Aadhaar/PAN/DL/Passport number, Father's Name, and Address (if available). "
            "Output in simple lines like:\nName: ...\nDOB: ...\nAadhaar Number: ...\nFather's Name: ...\nAddress: ...",
            {"mime_type": mime_type, "data": content},
        ]
    )

    raw_text = result.text if hasattr(result, "text") else ""
    fields = parse_fields(raw_text)

    return {
        "Raw_Text": clean_text(raw_text),
        **fields,
    }
