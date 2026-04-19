import cv2 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model
model = load_model("rps_model.h5")

classes = ['paper', 'rock', 'scissors']

def predict_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # IMPORTANT
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

# Start camera
cap = cv2.VideoCapture(0)

print("Press 'c' to capture & predict")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        # Save image
        cv2.imwrite("captured.jpg", frame)

        # Predict
        result = predict_image(frame)

        # Show result on image
        cv2.putText(frame, f"Prediction: {result}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Result", frame)
        print("Prediction:", result)

        cv2.waitKey(2000)  # show for 2 seconds

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()