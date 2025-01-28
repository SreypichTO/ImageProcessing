from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import face_recognition
from deepface import DeepFace

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Render the HTML page
    return render_template('index.html')

@app.route('/api')
def api():
    # Return a JSON response
    return jsonify({"message": "Face Finding API"})

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'video' not in request.files or 'photo' not in request.files:
        return jsonify({'error': 'Both video and photo files are required.'}), 400

    video_file = request.files['video']
    photo_file = request.files['photo']

    if not (allowed_file(video_file.filename) and allowed_file(photo_file.filename)):
        return jsonify({
            'error': 'Invalid file types. Allowed types are: mp4, avi, mov for video, and jpg, jpeg, png for photos.'
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
    """
    Detect and match faces in a video with a reference photo using DeepFace.

    Args:
        video_path (str): Path to the video file.
        photo_path (str): Path to the reference photo.

    Returns:
        dict: Results containing matched frame numbers, total frames processed, and processed video URL.
    """
    # Load reference image
    try:
        # Verify that the reference photo is valid
        DeepFace.verify(photo_path, photo_path)
    except Exception as e:
        raise ValueError("No valid face found in the reference photo.")

    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    matched_frames = []

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    processed_video_path = video_path.replace(".mp4", "_processed.mp4")
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Save the current frame temporarily
        temp_frame_path = "temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)

        try:
            # Use DeepFace to compare the frame with the reference photo
            result = DeepFace.verify(img1_path=photo_path, img2_path=temp_frame_path, model_name="Facenet", enforce_detection=False)
            if result["verified"]:
                matched_frames.append(total_frames)

                # Draw rectangle and label on the matched frame
                cv2.putText(frame, "Matched Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing frame {total_frames}: {e}")

        out.write(frame)

    cap.release()
    out.release()

    return {
        "matched_frames": matched_frames,
        "total_frames": total_frames,
        "processed_video_url": processed_video_path
    }
if __name__ == '__main__':
    app.run(debug=True)
