import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("hand_gesture_model.h5")

IMG_SIZE = 64

# Class labels (0–9)
class_names = [
    "Palm",
    "L",
    "Fist",
    "Fist Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm Moved",
    "C",
    "Down"
]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    roi = frame[100:400, 100:400]

    # Preprocess
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    # Predict
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = f"{class_names[class_index]} ({confidence*100:.2f}%)"

    # Draw rectangle
    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)

    # Put text
    cv2.putText(frame, label, (100,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()