import os
from functions import preprocessForModel, l2_normalize, get_inference
from config import *
import json
import cv2

print("[LOG] Starting dabase construction ...")
databaseEntries = [] 
failedImages = 0

def buildDatabase(datasetPath: str = 'dabaset', detectorPath: str = "models/face_detection_yunet_2023mar.onnx",
                  detectorScoreThreshold: float = 0.6):

    yunetDetector = cv2.FaceDetectorYN_create(
            model=detectorPath,
            config='',
            input_size=(320, 320),
            score_threshold=float(detectorScoreThreshold),
            nms_threshold=0.5,
            top_k=5000
        )

    for person in os.listdir(datasetPath):
        pdir = os.path.join(datasetPath, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            path = os.path.join(pdir, fn)
            img = cv2.imread(path)
            if img is None:
                failedImages += 1
                continue
            h, w = img.shape[:2]
            try:
                yunetDetector.setInputSize((w, h))
                _, faces = yunetDetector.detect(img)
            except Exception:
                faces = None
            if faces is None or len(faces) == 0:
                failedImages += 1
                continue
            f = faces[0]
            x, y, fw, fh = map(int, f[:4])
            x1 = max(0, x-10); y1 = max(0, y-10); x2 = min(w, x+fw+10); y2 = min(h, y+fh+10)
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                failedImages += 1
                continue

            yunetDetector, session, inputShape, inputName, outputName = get_inference(modelsPath, recognitionModel, detectionModel, detectorScoreThreshold)

            try:
                blob = preprocessForModel(face_crop)
                emb = session.run([outputName], {inputName: blob})[0].flatten()
                emb = l2_normalize(emb)

                entry = {
                    "person": person,
                    "file": fn,
                    "path": path,
                    "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # x,y,w,h
                    "embedding": emb.tolist()
                }
                databaseEntries.append(entry)

            except Exception:
                failedImages += 1
                continue
            
    print("[LOG] Database construction finished")

    # salvar em disco
    out_path = "database.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"entries": databaseEntries, "failedImages": failedImages}, f, ensure_ascii=False, indent=2)


