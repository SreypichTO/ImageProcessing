import tkinter as tk
from tkinter import ttk, filedialog
import PIL.Image, PIL.ImageTk
import cv2
from backend import FaceDetectionBackend
import threading

class FaceDetectionUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Detection Application")
        self.window.configure(bg='#1a1b26')
        
        # Initialize backend
        self.backend = FaceDetectionBackend()
        
        # Configure styles
        style = ttk.Style()
        style.configure('Custom.TButton', 
                       background='#666666',
                       foreground='white',
                       padding=10,
                       font=('Arial', 10))
        
        # Create main container
        self.main_frame = ttk.Frame(window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Preview area (black background with blue border)
        self.preview_frame = tk.Frame(
            self.main_frame,
            width=500,
            height=500,
            bg='black',
            highlightbackground='#0099ff',
            highlightthickness=2
        )
        self.preview_frame.grid(row=0, column=0, padx=10, pady=10)
        self.preview_frame.grid_propagate(False)
        
        # Preview label for image/video
        self.preview_label = tk.Label(self.preview_frame, bg='black')
        self.preview_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Control panel
        self.control_frame = ttk.Frame(self.main_frame, padding="5")
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')
        
        # Buttons
        self.video_button = ttk.Button(
            self.control_frame,
            text="Upload a video",
            style='Custom.TButton',
            command=self.upload_video
        )
        self.video_button.grid(row=0, column=0, pady=5, sticky='ew')
        
        self.photo_button = ttk.Button(
            self.control_frame,
            text="Upload a photo",
            style='Custom.TButton',
            command=self.upload_photo
        )
        self.photo_button.grid(row=1, column=0, pady=5, sticky='ew')
        
        self.analysis_button = ttk.Button(
            self.control_frame,
            text="Analysis",
            style='Custom.TButton',
            command=self.start_analysis
        )
        self.analysis_button.grid(row=2, column=0, pady=5, sticky='ew')
        
        # Results frame
        self.results_frame = ttk.Frame(self.control_frame, padding="5")
        self.results_frame.grid(row=3, column=0, pady=20, sticky='ew')
        
        # Results labels
        self.subject_count_label = ttk.Label(
            self.results_frame,
            text="Subject frame count: XX",
            foreground='white',
            background='#1a1b26'
        )
        self.subject_count_label.grid(row=0, column=0, pady=2)
        
        self.total_count_label = ttk.Label(
            self.results_frame,
            text="Total frame count: XX",
            foreground='white',
            background='#1a1b26'
        )
        self.total_count_label.grid(row=1, column=0, pady=2)
        
        self.current_file = None
        self.file_type = None

    def upload_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.current_file = file_path
            self.file_type = 'video'
            self.show_video_preview(file_path)

    def upload_photo(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.current_file = file_path
            self.file_type = 'image'
            self.show_image_preview(file_path)

    def show_video_preview(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize_for_preview(frame)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
        cap.release()

    def show_image_preview(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_for_preview(image)
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def resize_for_preview(self, image):
        target_size = (480, 480)
        h, w = image.shape[:2]
        aspect = w/h
        
        if aspect > 1:
            new_w = target_size[0]
            new_h = int(new_w/aspect)
        else:
            new_h = target_size[1]
            new_w = int(new_h*aspect)
            
        return cv2.resize(image, (new_w, new_h))

    def start_analysis(self):
        if not self.current_file:
            return
            
        # Start analysis in a separate thread to keep UI responsive
        thread = threading.Thread(
            target=self.analyze_file,
            args=(self.current_file, self.file_type)
        )
        thread.start()

    def analyze_file(self, file_path, file_type):
        if file_type == 'video':
            results = self.backend.analyze_video(file_path)
        else:
            results = self.backend.analyze_image(file_path)
            
        # Update UI in the main thread
        self.window.after(0, self.update_results, results)

    def update_results(self, results):
        self.subject_count_label.configure(
            text=f"Subject frame count: {results['subject_count']}"
        )
        self.total_count_label.configure(
            text=f"Total frame count: {results['total_count']}"
        )

def main():
    root = tk.Tk()
    root.configure(bg='#1a1b26')
    app = FaceDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()