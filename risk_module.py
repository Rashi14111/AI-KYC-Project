# modules/risk_module.py

def calculate_risk(extracted_data, liveness_result, voice_result):
    """
    Calculate overall KYC risk level based on:
    - Forgery checks (blur, QR, metadata, edges)
    - Liveness detection
    - Voice verification
    """

    risk_score = 0
    risk_factors = []

    # =============================
    # 1. Document Checks
    # =============================
    for doc_type, results in extracted_data.items():
        forgery_result = results.get("forgery", {})

        # If forgery module flagged anything
        if isinstance(forgery_result, dict):
            suspicious_reasons = forgery_result["details"].get("suspicious_reasons", [])
            if suspicious_reasons:
                risk_score += len(suspicious_reasons) * 2  # each reason adds 2 points
                risk_factors.append(f"{doc_type}: {', '.join(suspicious_reasons)}")

        # If OCR extraction failed badly
        ocr_data = results.get("ocr", {})
        if not ocr_data or (ocr_data.get("Name") == "Not Found"):
            risk_score += 3
            risk_factors.append(f"{doc_type}: OCR missing key details")

    # =============================
    # 2. Liveness
    # =============================
    if not liveness_result:
        risk_score += 5
        risk_factors.append("Liveness check failed")

    # =============================
    # 3. Voice Verification
    # =============================
    if not voice_result:
        risk_score += 3
        risk_factors.append("Voice verification failed")

    # =============================
    # 4. Risk Level Classification
    # =============================
    if risk_score <= 3:
        level = "Low Risk ✅"
    elif risk_score <= 8:
        level = "Medium Risk ⚠️"
    else:
        level = "High Risk ❌"

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "factors": risk_factors
    }
