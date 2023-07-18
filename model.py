'''import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('./CNN_Casting_Inspection.hdf5')

# Set up camera capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index (0 for default camera)

# Continuously capture and process frames
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    # Preprocess the frame
    preprocessed_frame = cv2.resize(frame, (300, 300))  # Resize the frame
    preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    preprocessed_frame = preprocessed_frame / 255.0  # Normalize pixel values
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=-1)

    # Make predictions
    prediction = model.predict(preprocessed_frame)

    # Interpret the prediction
    class_label = "Defective" if prediction[0][0] >= 0.5 else "OK"
    confidence = prediction[0][0] if class_label == "Defective" else 1 - prediction[0][0]

    # Overlay annotations on the frame
    label_text = f"{class_label} ({confidence:.2f})"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with annotations
    cv2.imshow('Casting Defect Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()'''
import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('./CNN_Casting_Inspection.hdf5')

# Load and preprocess the test image
image_path = r'C:\Users\Dayanand\PycharmProjects\pythonProject\Casting Defect ' \
             r'Detection\Dataset\casting_512x512\casting_512x512\def_front\cast_def_0_238.jpeg'
# Replace with the
# Load and preprocess the test image

image = cv2.imread(image_path)
preprocessed_image = cv2.resize(image, (300, 300))
preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
preprocessed_image = preprocessed_image / 255.0
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)

# Make predictions
prediction = model.predict(preprocessed_image)

# Interpret the prediction
class_label = "Defective" if prediction[0][0] >= 0.5 else "OK"
confidence = prediction[0][0] if class_label == "Defective" else 1 - prediction[0][0]

# Draw a rectangle around the defective area if it is detected
if class_label == "Defective":
    # Calculate the coordinates of the rectangle
    x, y, w, h = 50, 50, 200, 200  # Adjust the values based on the detected defective area

    # Draw the rectangle on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result image
cv2.imshow('Casting Defect Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
