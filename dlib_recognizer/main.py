import numpy as np
from PIL import Image
import dlib
import os
import cv2
from tqdm import tqdm

datasetPath = './dataset'
videoPath = 'input2.mp4'
outputPath = 'output.mp4'
confianca = 0.5
paths = []

for pasta in os.listdir(datasetPath):
    pasta_path = os.path.join(datasetPath, pasta)
    if os.path.isdir(pasta_path):
        for arquivo in os.listdir(pasta_path):
            caminho_completo = os.path.join(pasta_path, arquivo)
            paths.append(caminho_completo)

face_detector = dlib.get_frontal_face_detector()
point_detector = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
descritor_facial_extrator = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

index = {}
idx = 0
descritores_faciais = None

for path in paths:
    image = Image.open(path).convert("RGB")
    image_np = np.array(image, "uint8")
    detections = face_detector(image_np, 1)

    for face in detections:
        pontos = point_detector(image_np, face)
        face_descriptor = descritor_facial_extrator.compute_face_descriptor(image_np, pontos)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = face_descriptor[np.newaxis, :]

        if descritores_faciais is None:
            descritores_faciais = face_descriptor
        else:
            descritores_faciais = np.concatenate((descritores_faciais, face_descriptor), axis=0)

        index[idx] = path
        idx += 1

# Configura captura e writer de vídeo
cap = cv2.VideoCapture(videoPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

# Loop com tqdm
for _ in tqdm(range(total_frames), desc="Processando vídeo", unit="frame"):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_detections = face_detector(gray, 1)

    for face in frame_detections:
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        pontos = point_detector(gray, face)
        descritor_facial = descritor_facial_extrator.compute_face_descriptor(frame, pontos)
        descritor_facial = [f for f in descritor_facial]
        descritor_facial = np.asarray(descritor_facial, dtype=np.float64)
        descritor_facial = descritor_facial[np.newaxis, :]

        distancias = np.linalg.norm(descritor_facial - descritores_faciais, axis=1)
        indice_minimo = np.argmin(distancias)
        distancia_minima = distancias[indice_minimo]

        if distancia_minima <= confianca:
            nome_previsao = os.path.basename(os.path.dirname(index[indice_minimo]))
            cor_box = (0, 255, 0)   # verde
        else:
            nome_previsao = 'Unknown'
            cor_box = (0, 140, 255) # laranja

        cv2.rectangle(frame, (l, t), (r, b), cor_box, 2)

        for p in pontos.parts():
            cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"Pred: {nome_previsao}", (l, t - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, cor_box, 1)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
