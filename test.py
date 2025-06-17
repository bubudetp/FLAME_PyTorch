import dlib
import cv2
import numpy as np

img = cv2.imread('mo.webp')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

faces = detector(gray)
shape = predictor(gray, faces[0])

landmarks = np.array([[p.x, p.y] for p in shape.parts()])

FLAME_LANDMARKS = [
     33,  40,  37,  43,  46,  49,  52,  55,  58,  61,  # Mouth
     64,  67,  70,  73,  76,  79,  82,  85,            # Right eyebrow
     88,  91,  94,  97, 100, 103, 106, 109,            # Left eyebrow
    112, 115, 118, 121, 124, 127, 130, 133, 136, 139,  # Nose
    142, 145, 148, 151, 154, 157, 160, 163, 166, 169,  # Jawline
    172, 175, 178, 181, 184, 187, 190, 193,            # Eyes
    196, 199, 202, 205, 208, 211, 214, 217             # Chin + extra
]
