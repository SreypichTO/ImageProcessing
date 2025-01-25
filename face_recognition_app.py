import cv2
import face_recognition
import os
import numpy as np

# Load known faces
known_face_encodings = []
known_face_names = []

# Path to the directory containing known faces
known_faces_dir = "input/images/"

# Load each known face
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use the filename (without extension) as the name

# Start video capture
video_capture = cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capture a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = []

    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()