import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model(os.path.join(os.path.dirname(__file__), "models", "facial_emotion_detection.keras"))
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cam = cv2.VideoCapture(1)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Process the image for face detection and emotion classification
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Draw the face detection results
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w = int(bbox.width * w)
            h = int(bbox.height * h)

            x, y = max(0, x), max(0, y)

            face_roi = frame[y:y+h, x:x+w]

            # Perform emotion classification on the face region
            if face_roi.size > 0:
                # convert to grayscale and resize to match CNN input
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                normalized_face = resized_face / 255.0

                input_face = np.expand_dims(normalized_face, axis=-1)
                input_face = np.expand_dims(input_face, axis=0)

                predicts = model.predict(input_face, verbose=0)
                # print(f"Emotion Predictions: {predicts}")

                emotion_label = EMOTIONS_LIST[np.argmax(predicts)]

                # draw bounding box and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('CNN Input view', resized_face)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()