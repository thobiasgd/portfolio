import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort
import json
import os

def preprocessForModel(img):
    H = 128
    W = 128
    resizedImg = cv2.resize(img, (W, H))
    rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalizedImg = (rgbImg - 127.5) / 128
    blob = np.transpose(normalizedImg, (2, 0, 1))
    blob = np.expand_dims(blob, axis = 0).astype(np.float32)
    return blob

def l2_normalize(versor):
    versor = versor.astype(np.float32)
    norma = np.linalg.norm(versor)
    if norma > 0:
        return versor / norma
    return versor

def loadDatabaseJson(json_path):

    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"{json_path} not found.")
    data = json.loads(p.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    return entries

def buildBank(entries, normalize=True):

    emb_list = []
    names = []
    meta = []
    for e in entries:
        emb = e.get("embedding")
        name = e.get("person", "Unknown")
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.flatten()
        emb_list.append(arr)
        names.append(name)
        meta.append({"file": e.get("file"), "path": e.get("path"), "bbox": e.get("bbox")})
    if len(emb_list) == 0:
        return np.zeros((0,0), dtype=np.float32), [], meta
    bank_embeddings = np.vstack(emb_list)  # shape (N, D)
    if normalize:
        norms = np.linalg.norm(bank_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        bank_embeddings = bank_embeddings / norms
    return bank_embeddings.astype(np.float32), names, meta

def saveBankNpz(out_path, bank_embeddings, bank_names, meta=None):
    np.savez_compressed(out_path,
                        embeddings=bank_embeddings,
                        names=np.array(bank_names, dtype=object),
                        meta=np.array(meta, dtype=object) if meta is not None else np.array([], dtype=object))

def loadBankNpz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["embeddings"], d["names"].tolist(), d.get("meta", np.array([], dtype=object)).tolist()

def get_inference(modelsPath, recognitionModel, detectionModel, detectorScoreThreshold):

    detector_path = os.path.join(modelsPath, detectionModel)
    if not os.path.isfile(detector_path):
        raise FileNotFoundError(f"Detector not found: {detector_path}")

    try:
        yunetDetector = cv2.FaceDetectorYN_create(
            model=detector_path,
            config='',
            input_size=(320, 320),
            score_threshold=float(detectorScoreThreshold),
            nms_threshold=0.5,
            top_k=5000
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create Yunet detector: {e}")

    rec_path = os.path.join(modelsPath, recognitionModel)
    if not os.path.isfile(rec_path):
        raise FileNotFoundError(f"Recognition model not found: {rec_path}")

    try:
        session = ort.InferenceSession(rec_path, providers=['CPUExecutionProvider'])
        inputName = session.get_inputs()[0].name
        outputName = session.get_outputs()[0].name
        inputShape = session.get_inputs()[0].shape
    except Exception as e:
        raise RuntimeError(f"Failed to create ONNX session: {e}")

    return yunetDetector, session, inputShape, inputName, outputName