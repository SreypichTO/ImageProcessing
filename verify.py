from deepface import DeepFace
import tensorflow as tf

# Check TensorFlow GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU in use: ", tf.test.gpu_device_name())

# Test DeepFace
result = DeepFace.verify(img1_path="C:/Users/ASUS ROG/Downloads/Telegram Desktop/20250124_144010.jpg", img2_path="C:/Users/ASUS ROG/Downloads/Telegram Desktop/e20200810_profile.jpg", model_name="Facenet")
print("DeepFace Result:", result)
