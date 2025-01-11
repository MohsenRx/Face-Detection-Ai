from matplotlib import pyplot as plt
import cv2

# Load the image
image = cv2.imread("Ai.jpg")
if image is None:
    print("Image not found!")
    exit()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade model
model = cv2.CascadeClassifier("model.xml")
if model.empty():
    print("Model file not found!")
    exit()

# Detect faces in the image
faces = model.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

# Check if any faces were detected
if len(faces) == 0:
    print("No faces detected.")
    exit()

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 260, 100), 5)

# Convert the image to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
