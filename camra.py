import cv2
import numpy as np
from model import EmotionModelV2, EmotionModelV1

# Initialize face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load emotion model
weights_path = 'emotion_model.h5'
emotion_model = EmotionModelV1(weights_path)


# weights_path = 'model_weights.h5'
# emotion_model = EmotionModelV2(weights_path)

font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return b''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray_resized = cv2.resize(roi_gray, (48, 48))
            roi_input = roi_gray_resized[np.newaxis,
                                         :, :, np.newaxis] / 255.0  # Normalize

            pred = emotion_model.predict_emotion(roi_input)

            cv2.putText(frame, pred, (x, y-10), font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
