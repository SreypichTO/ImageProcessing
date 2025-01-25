import cv2
import face_recognition_app

img1 = cv2.imread("input/img1.jpg")
rgb_img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img_encoding = face_recognition_app.face_encodings(rgb_img)[0]

img2 = cv2.imread("input/img2.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
img_encoding2 = face_recognition_app.face_encodings(rgb_img2)[0]

result = face_recognition_app.compare_faces([img_encoding], img_encoding2)

print("Result: ", result)
# cv2.imshow("image: ", img)

 
cv2.waitKey(0)
cv2.destroyAllWindows()