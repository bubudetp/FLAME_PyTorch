import dlib
import cv2
import numpy as np

def detect_68_landmarks(image_path, predictor_path="./model/landmark/shape_predictor_68_face_landmarks.dat"):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets = detector(gray, 1)
    if len(dets) == 0:
        raise RuntimeError("No face detected")

    shape = predictor(gray, dets[0])
    coords = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)  # (68, 2)
    return coords, img