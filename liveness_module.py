# liveness_module.py
import cv2
import random
import time
from math import hypot

# Haar cascades for face & eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def eye_aspect_ratio(eye_box):
    """Approximate Eye Aspect Ratio (EAR) from bounding box."""
    x, y, w, h = eye_box
    return h / float(w)

def check_liveness(video_path=0):
    """
    Liveness detection with random challenges:
    - Blink twice
    - Turn head left
    - Turn head right
    
    The video_path=0 argument uses the default webcam.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return False

    challenges = ["blink twice", "turn head left", "turn head right"]
    challenge = random.choice(challenges)
    print(f"Challenge: Please {challenge}")

    blink_count = 0
    prev_face_x = None
    success = False
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from video stream.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            continue

        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]

        # --- Blink detection ---
        if challenge == "blink twice":
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes[:2]:
                ear = eye_aspect_ratio((ex, ey, ew, eh))
                if ear < 0.25:  # eye closed
                    blink_count += 1
                    print("Blink detected")
                    time.sleep(0.3)  # avoid multiple counts for one blink
            if blink_count >= 2:
                success = True
                break

        # --- Head turn detection ---
        else:
            center_x = x + w // 2
            if prev_face_x is not None:
                dx = center_x - prev_face_x
                face_direction = None
                if challenge == "turn head left" and dx < -15:
                    face_direction = "left"
                elif challenge == "turn head right" and dx > 15:
                    face_direction = "right"
                
                if face_direction and face_direction in challenge:
                    success = True
                    break

            prev_face_x = center_x

        # Timeout after 15s
        if time.time() - start_time > 15:
            print("Liveness check timed out.")
            break
        
        # Display the live feed with a challenge overlay
        cv2.putText(frame, f"Challenge: {challenge}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Liveness Check", frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return success

# Example usage
# if __name__ == '__main__':
#     if check_liveness():
#         print("Liveness check passed!")
#     else:
#         print("Liveness check failed.")