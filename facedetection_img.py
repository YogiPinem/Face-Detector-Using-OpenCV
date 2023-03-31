import cv2

cascPath = "haarcascade_frontalface_default.xml"
wajah = cv2.CascadeClassifier(cascPath)

img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

deteksi_wajah = wajah.detectMultiScale(
    img_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
for (x, y, w, h) in deteksi_wajah:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Facedetector', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
