# Reconhecimento Facial com Dlib e OpenCV

Este projeto implementa um pipeline completo de **reconhecimento facial** utilizando **Dlib**, **OpenCV** e **tqdm** para exibição de progresso.  
O script processa um vídeo quadro a quadro, detecta rostos, compara com um banco de dados local e gera um vídeo anotado com caixas delimitadoras e pontos faciais.

---

## Visão Geral

O código constrói um banco de descritores faciais a partir das imagens localizadas na pasta `./dataset`.  
Cada subpasta representa uma identidade (exemplo: `Ross`, `Rachel`, `Monica` etc.). O data set usado no projeto pode ser baixado [aqui](https://drive.google.com/drive/folders/17t4kjOZONdatwjX8VJu7bXsBwwfu-M0Y?usp=sharing).
Em seguida, usando o modelo `face_recognition_model_v1` da Dlib, o script processa cada frame de um vídeo, identifica rostos conhecidos e marca os desconhecidos com caixas laranjas.

Rostos reconhecidos são desenhados com caixas verdes e o nome previsto é exibido acima da detecção.  
Todos os quadros são salvos em um novo arquivo de vídeo chamado **`output.mp4`**.

---

##  Funcionalidades

- Detecção facial usando **Dlib’s frontal face detector**
- Extração de pontos faciais com **68 landmarks**
- Geração de embeddings faciais via **modelo ResNet pré-treinado da Dlib**
- Exibição de progresso com **tqdm**
- Identificação automática de rostos conhecidos e desconhecidos
- Salvamento automático do vídeo com anotações e landmarks

---

##  Estrutura de Pastas

```
projeto/
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

##  Código

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

##  Dependências

```bash
pip install numpy opencv-python pillow dlib tqdm
```

Também é necessário baixar os modelos pré-treinados:

- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

Os modelos podem ser baixados gratuitamente no [link](https://drive.google.com/drive/folders/1FUGihMk2FjKWFSaFSwHLOtbUkYmZB51F?usp=sharing).

---

##  Saída

O vídeo final (`output.mp4`) inclui:
- Caixas **verdes** para rostos reconhecidos  
- Caixas **laranjas** para rostos desconhecidos  
- Pontos faciais desenhados em **verde** sobre cada rosto detectado

  ![Demonstração do Projeto](https://github.com/thobiasgd/portfolio/blob/021610b2538ecd56e98abf07459ce536cb05ca86/dlib_recognizer/output.gif)

---

##  Licença

Este projeto é aberto para estudo, aprendizado e adaptação em aplicações de visão computacional.
