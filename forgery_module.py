import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract

# pyzxing support
try:
    from pyzxing import BarCodeReader
    ZXING_AVAILABLE = True
except ImportError:
    ZXING_AVAILABLE = False

# ----------------------------
def normalize_id(id_str):
    return "".join(filter(str.isalnum, str(id_str)))

# ----------------------------
def load_image(file):
    if isinstance(file, Image.Image):
        return file
    if isinstance(file, str):
        if file.lower().endswith(".pdf"):
            with open(file, "rb") as f:
                images = convert_from_bytes(f.read())
            return images[0]
        return Image.open(file).convert("RGB")
    # Assume UploadedFile
    file.seek(0)
    filename = getattr(file, "name", "file").lower()
    if filename.endswith(".pdf"):
        images = convert_from_bytes(file.read())
        img = images[0]
    else:
        img = Image.open(file).convert("RGB")
    file.seek(0)
    return img

# ----------------------------
def analyze_blur(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    h, w = gray.shape
    threshold = 20 * (h * w) / 1_000_000
    return blur_value, ["⚠️ Image may be blurred"] if blur_value < threshold else []

# ----------------------------
def analyze_edges(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    h, w = gray.shape
    edge_threshold = 0.01 * h * w
    return edge_count, ["⚠️ Low edge details"] if edge_count < edge_threshold else []

# ----------------------------
def detect_and_validate_qr(img, ocr_data, doc_type):
    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # OpenCV QR detector
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(gray)

    # Try rotated if not detected
    if not data:
        rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        data, _, _ = detector.detectAndDecode(rotated)

    # Try ZXing if available
    if not data and ZXING_AVAILABLE:
        try:
            reader = BarCodeReader()
            results = reader.decode_array(gray)
            if results and "parsed" in results[0]:
                data = results[0]["parsed"]
        except:
            pass

    # Fallback to OCR
    if not data:
        try:
            text = pytesseract.image_to_string(Image.fromarray(gray))
            if any(x in text.lower() for x in ["aadhaar", "govt", "dl"]):
                data = text
        except:
            pass

    if not data:
        if doc_type in ["Aadhaar Front", "Aadhaar Back", "DL Front", "DL Back"]:
            return False, ["⚠️ No QR detected (expected)"]
        return False, []

    qr_text = data.strip()
    reasons = []

    if "Aadhaar" in doc_type:
        aadhaar_num = normalize_id(ocr_data.get("Aadhaar_Number", ""))
        name_first = ocr_data.get("Name", "").split()[0] if ocr_data.get("Name") else ""
        dob_year = ocr_data.get("DOB", "").split("/")[-1] if ocr_data.get("DOB") else ""
        if aadhaar_num and aadhaar_num not in normalize_id(qr_text):
            reasons.append("❌ Aadhaar number mismatch")
        if name_first and name_first not in qr_text:
            reasons.append("❌ Name mismatch")
        if dob_year and dob_year not in qr_text:
            reasons.append("❌ DOB mismatch")

    if "DL" in doc_type:
        dl_num = normalize_id(ocr_data.get("DL_Number", ""))
        if dl_num and dl_num not in normalize_id(qr_text):
            reasons.append("❌ DL number mismatch")

    return True, reasons

# ----------------------------
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
    return {"status": status, "blur_value": blur_value, "edge_count": edge_count,
            "details": {"suspicious": suspicious, "warnings": warnings, "qr_found": qr_found, "doc_type": doc_type}}
