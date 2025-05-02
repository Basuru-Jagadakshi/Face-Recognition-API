import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import pyttsx3
import tensorflow as tf
import time

# Load pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained face recognition model
model_path = "text05.hdf5"
model = load_model(model_path)

# Mapping labels to names
label_to_name = {
    0: 'Basuru',
    1: 'Chamith',
    2: 'Sulakshika',
    3: 'Yashodha'
}

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Slightly slower speed to improve clarity
engine.setProperty('volume', 1.0)  # Set the volume to 100%

def get_greeting():
    """Return a greeting based on the current time."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 21:
        return "Good evening"
    else:
        return "Good night"

def recognize_face(face_image):
    """Recognize a face using the trained model and return label, name, and confidence."""
    # Resize and normalize the face image
    face_input = cv2.resize(face_image, (64, 64))  # Resize to model's input size
    face_input = np.expand_dims(face_input, axis=0) / 255.0  # Normalize
    prediction = model.predict(face_input)
    label = np.argmax(prediction)
    confidence = prediction[0][label]
    return label, label_to_name.get(label, "Unknown"), confidence

def speak_greeting(greeting, name):
    """Speak the greeting with the recognized name, adding 'Hi' before the greeting."""
    message = f"Hi! {greeting} {name}! Have a nice day!"
    engine.say(message)
    engine.runAndWait()

def capture_faces(cap, frame):
    """Capture 5 frames within 3 seconds."""
    captured_faces = []
    start_time = time.time()
    while len(captured_faces) < 5 and time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            captured_faces.append(face_image)

    return captured_faces

def main():
    """Main function to run the face detection and recognition app."""
    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    recognized = False  # Flag to ensure greeting is said only once

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Capture 5 frames within the first 3 seconds
            captured_faces = capture_faces(cap, frame)

            # After 3 seconds, compare the captured faces with the model
            if captured_faces and not recognized:
                total_confidence = 0
                name = "Unknown"
                for face_image in captured_faces:
                    label, user_name, confidence = recognize_face(face_image)
                    total_confidence += confidence
                    if user_name != "Unknown":
                        name = user_name

                    # Print the confidence level in the terminal
                    print(f"Recognized {user_name} with confidence {confidence:.2f}")

                # Calculate average confidence
                avg_confidence = total_confidence / len(captured_faces)

                # Display the greeting message without confidence level
                if name != "Unknown" and avg_confidence > 0.75:
                    greeting = get_greeting()
                    text = f"{greeting} {name}!"  # Removed confidence level from text

                    # Save the recognized face image
                    cv2.imwrite(f"recognized_{name}.jpg", captured_faces[0])

                    # Display the greeting message and the photo in a new window
                    greeting_window = np.zeros((400, 600, 3), dtype=np.uint8)  # Black window
                    cv2.putText(greeting_window, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    recognized_image = cv2.imread(f"recognized_{name}.jpg")
                    resized_image = cv2.resize(recognized_image, (300, 300))
                    greeting_window[100:400, 150:450] = resized_image
                    cv2.imshow('Greeting & Photo', greeting_window)

                    speak_greeting(greeting, name)
                    recognized = True  # Set flag to true so greeting is not repeated

                    # Exit the program after greeting
                    time.sleep(3)  # Show the window for a while before closing
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        # Display the frame
        cv2.imshow('Face Detection & Recognition', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
