import cv2
import dlib
import face_recognition
cap = cv2.VideoCapture(0)  # Change the index if necessary

sfr = SimpleFacerec()
sfr.load_encoding_images("input/images/")

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()

        # Detect faces
        face_locations, face_name = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_name):
            print(face_loc)

        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()