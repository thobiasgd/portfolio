# Face Recognition with Dlib and OpenCV

This project implements a face recognition pipeline using **Dlib**, **OpenCV**, and **tqdm** for progress visualization.  
It processes a video input, detects faces frame by frame, compares them against a local dataset, and exports an annotated video with bounding boxes and facial landmarks.

---

## Overview

The script builds a facial descriptor database from images stored in the `./dataset` directory.  
Each subfolder represents one identity (e.g. `Ross`, `Rachel`, `Monica`, etc.).  
Then, using Dlib’s `face_recognition_model_v1`, the script processes every frame of a video file, identifies known faces, and marks unknown ones with orange bounding boxes.

Recognized faces are drawn in green and display the predicted label above the detection.  
All frames are written to a new file called `output.mp4`.

---

## Features

- Face detection using **Dlib’s frontal face detector**
- Facial landmark extraction with **68-point shape predictor**
- Facial embeddings generated via **ResNet-based Dlib model**
- Real-time frame processing with **tqdm** progress bar
- Automatic labeling of known vs. unknown faces
- Output video saved with all annotations and landmarks

---

## Folder Structure

```
project/
│
├── dataset/
│   ├── Ross/
│   ├── Rachel/
│   ├── Monica/
│   └── ...
│
├── models/
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── input2.mp4
└── output.mp4
```

---

## Code

```python
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
        face_descriptor = np.asarray([f for f in face_descriptor], dtype=np.float64)[np.newaxis, :]

        if descritores_faciais is None:
            descritores_faciais = face_descriptor
        else:
            descritores_faciais = np.concatenate((descritores_faciais, face_descriptor), axis=0)

        index[idx] = path
        idx += 1

cap = cv2.VideoCapture(videoPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

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
        descritor_facial = np.asarray([f for f in descritor_facial], dtype=np.float64)[np.newaxis, :]

        distancias = np.linalg.norm(descritor_facial - descritores_faciais, axis=1)
        indice_minimo = np.argmin(distancias)
        distancia_minima = distancias[indice_minimo]

        if distancia_minima <= confianca:
            nome_previsao = os.path.basename(os.path.dirname(index[indice_minimo]))
            cor_box = (0, 255, 0)
        else:
            nome_previsao = 'Unknown'
            cor_box = (0, 140, 255)

        cv2.rectangle(frame, (l, t), (r, b), cor_box, 2)

        for p in pontos.parts():
            cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"Pred: {nome_previsao}", (l, t - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, cor_box, 1)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
```

---

## Requirements

```bash
pip install numpy opencv-python pillow dlib tqdm
```

You’ll also need the pretrained models:

- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

Both are available on the official [Dlib model downloads page](http://dlib.net/files/).

---

## Output Example

The final video (`output.mp4`) contains:
- Green boxes for recognized faces  
- Orange boxes for unknown faces  
- Green facial landmark points drawn for each detection  

---

## 🧾 License

This project is open for study and experimentation.  
Feel free to use or adapt it for your own computer vision applications.
