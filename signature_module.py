import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def extract_signature_from_pan(pan_image):
    """
    Extracts the signature region from a PAN card image.
    This is a placeholder and would need a robust ML model for production.
    For this demo, it assumes a fixed region.
    """
    try:
        if isinstance(pan_image, Image.Image):
            img_pil = pan_image.convert("RGB")
        else:
            # Assumes pan_image is a file-like object from Streamlit
            img_pil = Image.open(pan_image).convert("RGB")
            
        img_np = np.array(img_pil)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Placeholder for signature region on a typical PAN card
        # Adjust these coordinates based on the expected PAN card layout
        # A real solution would use a pre-trained detection model
        # Example coordinates (x, y, width, height)
        x, y, w, h = 300, 500, 300, 100 
        
        height, width = img_gray.shape
        # Heuristic to detect signature area based on image size
        if width > 1000 and height > 700:
            # Likely a high-res image
            x_start, y_start = int(0.4 * width), int(0.6 * height)
            x_end, y_end = int(0.7 * width), int(0.75 * height)
        else:
            # Lower-res image
            x_start, y_start = int(0.4 * width), int(0.65 * height)
            x_end, y_end = int(0.8 * width), int(0.85 * height)

        signature_region = img_gray[y_start:y_end, x_start:x_end]

        # Apply a binary threshold to isolate the signature lines
        _, signature_thresh = cv2.threshold(signature_region, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Find contours to crop tightly around the signature
        contours, _ = cv2.findContours(signature_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        
        # Add a small padding
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(signature_thresh.shape[1] - x, w + 2*pad)
        h = min(signature_thresh.shape[0] - y, h + 2*pad)
        
        cropped_signature = signature_region[y:y+h, x:x+w]
        
        if cropped_signature.size == 0:
            return None

        # Create a blank white canvas to place the signature on for a clean background
        final_canvas = np.full((150, 400), 255, dtype=np.uint8)
        
        # Resize the cropped signature to fit the canvas while maintaining aspect ratio
        sig_h, sig_w = cropped_signature.shape
        ratio = min(final_canvas.shape[0] / sig_h, final_canvas.shape[1] / sig_w)
        new_w = int(sig_w * ratio)
        new_h = int(sig_h * ratio)
        resized_sig = cv2.resize(cropped_signature, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Paste the resized signature onto the center of the canvas
        y_pos = (final_canvas.shape[0] - new_h) // 2
        x_pos = (final_canvas.shape[1] - new_w) // 2
        final_canvas[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = resized_sig

        return Image.fromarray(final_canvas)

    except Exception as e:
        print(f"Error extracting signature: {e}")
        return None

def compare_signatures(pan_signature_img, digital_signature_img):
    """
    Compares two signature images using the Structural Similarity Index (SSIM).
    This is a robust method that is less sensitive to small variations.
    """
    try:
        # 1. Convert to grayscale and resize to a consistent size
        size = (400, 150)
        pan_np = np.array(pan_signature_img.convert('L').resize(size, Image.LANCZOS))
        digital_np = np.array(digital_signature_img.convert('L').resize(size, Image.LANCZOS))

        # 2. Binarize the images to get clean black-and-white signatures
        _, pan_binary = cv2.threshold(pan_np, 128, 255, cv2.THRESH_BINARY_INV)
        _, digital_binary = cv2.threshold(digital_np, 128, 255, cv2.THRESH_BINARY_INV)

        # 3. Calculate SSIM. The data_range is 255 for 8-bit images.
        score, _ = ssim(pan_binary, digital_binary, full=True)
        
        # 4. Convert score to a percentage (0-100)
        return score * 100
        
    except Exception as e:
        print(f"Error comparing signatures: {e}")
        return 0