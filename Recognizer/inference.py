import os
import cv2
import onnxruntime as ort
from pathlib import Path
from functions import loadBankNpz, loadDatabaseJson, buildBank, saveBankNpz
from functions import preprocessForModel, l2_normalize
import numpy as np
from tqdm import tqdm

class Inferror:
    def __init__(self,
             detectorPath: str = "models/face_detection_yunet_2023mar.onnx",
             recognitionPath: str = "models/recognition_resnet27.onnx",
             detectorScoreThreshold: float = 0.6,
             recognizerScoreThreshold: float = 0.5):

        self.detectorScoreThreshold = float(detectorScoreThreshold)
        self.recognizerScoreThreshold = float(recognizerScoreThreshold)

        if not os.path.isfile(detectorPath):
            raise FileNotFoundError(f"Detector not found: {detectorPath}")
        
        self.yunetDetector = cv2.FaceDetectorYN_create(
            model=detectorPath,
            config='',
            input_size=(320, 320),
            score_threshold=self.detectorScoreThreshold,
            nms_threshold=0.5,
            top_k=5000
        )

        if not os.path.isfile(recognitionPath):
            raise FileNotFoundError(f"Recognition model not found: {recognitionPath}")
        self.session = ort.InferenceSession(recognitionPath, providers=['CPUExecutionProvider'])
        self.inputName = self.session.get_inputs()[0].name
        self.outputName = self.session.get_outputs()[0].name
        self.inputShape = self.session.get_inputs()[0].shape

    
    def gettingInference(self,
                     videoInput: str,
                     videoOutput: str,
                     NPZ_CACHE: str = "bank_cache.npz",
                     JSON_DB: str = "database.json",
                     detectorScoreThreshold: float | None = None,
                     colorRecognized: tuple = (0, 255, 0),
                     colorUnrecognized: tuple = (0, 100, 255),
                     recognizerScoreThreshold: float | None = None,
                     drawConfidence: bool = True):

        if detectorScoreThreshold is None:
            detectorScoreThreshold = self.detectorScoreThreshold
        if recognizerScoreThreshold is None:
            recognizerScoreThreshold = self.recognizerScoreThreshold


        if Path(NPZ_CACHE).exists():
            databaseEmbedding, databaseNames, databaseMeta = loadBankNpz(NPZ_CACHE)
        else:
            entries = loadDatabaseJson(JSON_DB)
            databaseEmbedding, databaseNames, databaseMeta = buildBank(entries, normalize=True)
            if databaseEmbedding.size > 0:
                saveBankNpz(NPZ_CACHE, databaseEmbedding, databaseNames, databaseMeta)

        cap = cv2.VideoCapture(videoInput)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=total_frames if total_frames>0 else None, desc="Inference: ", unit="frame", dynamic_ncols=True)

        if not cap.isOpened():
            raise Exception("[ERROR] Not able to open video!")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(videoOutput, fourcc, fps, (frame_w, frame_h))

        frame_idx = 0
        per_frame_preds = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            detections_this_frame = []
            
            h, w, _ = frame.shape
            self.yunetDetector.setInputSize((w, h))
            _, faces = self.yunetDetector.detect(frame)

            if faces is not None:
                for face in faces:
                    x, y, fw, fh = map(int, face[:4])

                    x1 = max(0, x-10); 
                    y1 = max(0, y-10); 
                    x2 = min(w, x+fw+10); 
                    y2 = min(h, y+fh+10)

                    ROI = frame[y1:y2, x1:x2]
                        
                    blob = preprocessForModel(ROI)
                    embedding = self.session.run([self.outputName], {self.inputName: blob})[0].flatten()
                    embedding = l2_normalize(embedding)

                    scores = np.dot(databaseEmbedding, embedding)
                    bestIndex = int(np.argmax(scores))
                    bestScore = float(scores[bestIndex])
                    predictedName = databaseNames[bestIndex] if bestScore >= recognizerScoreThreshold else "Unknown"

                    color = colorRecognized if bestScore >= recognizerScoreThreshold else colorUnrecognized
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{predictedName} {bestScore:.2f}" if drawConfidence else predictedName
                    cv2.putText(frame, label, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            pbar.update(1)
            out.write(frame)

        pbar.close()
        cap.release()
        out.release()