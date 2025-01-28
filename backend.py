import cv2
import numpy as np

class FaceDetectionBackend:
    def __init__(self):
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def analyze_video(self, video_path):
        """
        Analyze video file for face detection
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Results containing subject_count and total_count
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = 0
        frames_with_faces = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                frames_with_faces += 1
        
        cap.release()
        
        return {
            "subject_count": frames_with_faces,
            "total_count": total_frames
        }
    
    def analyze_image(self, image_path):
        """
        Analyze single image for face detection
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Results containing subject_count and total_count
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return {
            "subject_count": len(faces),
            "total_count": 1
        }

    def detect_faces(self, frame):
        """
        Detect faces in a single frame
        
        Args:
            frame (numpy.ndarray): Input image frame
            
        Returns:
            list: List of detected face coordinates
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def draw_faces(self, frame, faces):
        """
        Draw rectangles around detected faces
        
        Args:
            frame (numpy.ndarray): Input image frame
            faces (list): List of face coordinates
            
        Returns:
            numpy.ndarray: Frame with drawn rectangles
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return frame