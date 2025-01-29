from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from deepface import DeepFace

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load OpenCV's Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api')
def api():
    return jsonify({"message": "Face Finding API"})

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'video' not in request.files or 'photo' not in request.files:
        return jsonify({'error': 'Both video and photo files are required.'}), 400

    video_file = request.files['video']
    photo_file = request.files['photo']

    if not (allowed_file(video_file.filename) and allowed_file(photo_file.filename)):
        return jsonify({
            'error': 'Invalid file types. Allowed types: mp4, avi, mov for video, jpg, jpeg, png for photos.'
        }), 400

    video_filename = secure_filename(video_file.filename)
    photo_filename = secure_filename(photo_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
    video_file.save(video_path)
    photo_file.save(photo_path)

    try:
        results = process_video(video_path, photo_path)
        results['processed_video_url'] = f"/static/uploads/{os.path.basename(results['processed_video_url'])}"
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)

def process_video(video_path, photo_path):
    """ Process video to detect, verify, and track the face """
    try:
        DeepFace.verify(photo_path, photo_path)  # Ensure a valid face exists
    except Exception as e:
        raise ValueError("No valid face found in the reference photo.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if undetected
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    processed_video_path = video_path.replace(".mp4", "_processed.mp4")
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_frame.jpg")

    matched_frames = []
    matched_timestamps = []
    frame_number = 0
    face_detected_last = False  # Track whether face was detected in the previous frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Convert frame number to MM:SS:MS format
        total_seconds = frame_number / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)  # Convert fraction of a second to milliseconds
        timestamp = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

        # Convert frame to grayscale (Haar Cascade requires grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_detected = False  # Flag to check if a face is detected in this frame

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]  # Crop detected face
            cv2.imwrite(temp_frame_path, face_roi)  # Save cropped face for verification

            # Use DeepFace to verify face identity
            try:
                result = DeepFace.verify(img1_path=photo_path, img2_path=temp_frame_path, model_name="Facenet", enforce_detection=False)

                if result["verified"]:
                    face_detected = True
                    matched_frames.append(frame_number)
                    matched_timestamps.append(timestamp)

                    # Draw a bounding box around the detected face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, f"Matched at {timestamp}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error verifying face: {e}")

        # Stop tracking when face disappears (only reset if the last frame had a face)
        if face_detected_last and not face_detected:
            print(f"Face lost at frame {frame_number}")

        face_detected_last = face_detected  # Update last detected state

        out.write(frame)

    cap.release()
    out.release()

    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

    return {
        "matched_frames": matched_frames,
        "matched_timestamps": matched_timestamps,
        "total_frames": total_frames,
        "processed_video_url": processed_video_path
    }


if __name__ == '__main__':
    app.run(debug=True)