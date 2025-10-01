import os
import time
import cv2
import onnxruntime as ort
from pathlib import Path
from functions import loadBankNpz, loadDatabaseJson, buildBank, saveBankNpz
from functions import preprocessForModel, l2_normalize
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple


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

        # detector YuNet
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

        # session ONNX
        self.session = ort.InferenceSession(recognitionPath, providers=['CPUExecutionProvider'])
        self.inputName = self.session.get_inputs()[0].name
        self.outputName = self.session.get_outputs()[0].name
        self.inputShape = self.session.get_inputs()[0].shape

    def gettingInference(self,
                         videoInput: str,
                         videoOutput: Optional[str] = None,
                         NPZ_CACHE: str = "bank_cache.npz",
                         JSON_DB: str = "database.json",
                         detectorScoreThreshold: Optional[float] = None,
                         colorRecognized: Tuple[int, int, int] = (0, 255, 0),
                         colorUnrecognized: Tuple[int, int, int] = (0, 100, 255),
                         recognizerScoreThreshold: Optional[float] = None,
                         drawConfidence: bool = True,
                         is_stream: bool = False,
                         save_output: bool = False,
                         display: bool = True,
                         backend: Optional[str] = None,
                         reconnection_attempts: int = 5,
                         reconnection_backoff: float = 1.0):
        """
        Função principal de inferência.
        - is_stream: True quando a fonte é uma stream (RTSP ou arquivo tratado como stream)
        - save_output: se True, grava em videoOutput; se False, NÃO grava
        - display: se True, mostra janela com cv2.imshow (press 'q' para sair)
        - backend: opcional — passe 'ffmpeg' para tentar usar CAP_FFMPEG
        """

        # thresholds por parâmetro ou atributos
        if detectorScoreThreshold is None:
            detectorScoreThreshold = self.detectorScoreThreshold
        if recognizerScoreThreshold is None:
            recognizerScoreThreshold = self.recognizerScoreThreshold

        # --- carregar banco de embeddings (cache NPZ ou JSON) ---
        if Path(NPZ_CACHE).exists():
            databaseEmbedding, databaseNames, databaseMeta = loadBankNpz(NPZ_CACHE)
        else:
            entries = loadDatabaseJson(JSON_DB)
            databaseEmbedding, databaseNames, databaseMeta = buildBank(entries, normalize=True)
            if databaseEmbedding.size > 0:
                saveBankNpz(NPZ_CACHE, databaseEmbedding, databaseNames, databaseMeta)

        # normalizar formato caso banco vazio
        if databaseEmbedding is None or (isinstance(databaseEmbedding, np.ndarray) and databaseEmbedding.size == 0):
            databaseEmbedding = np.zeros((0, 512), dtype=np.float32)  # shape genérico
            databaseNames = []

        # --- helper para abrir VideoCapture com backend opcional ---
        def _open_cap(src: str):
            if backend and hasattr(cv2, "CAP_FFMPEG") and backend.lower() == "ffmpeg":
                try:
                    return cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                except Exception:
                    return cv2.VideoCapture(src)
            else:
                return cv2.VideoCapture(src)

        cap = _open_cap(videoInput)

        if not cap.isOpened():
            raise Exception(f"[ERROR] Not able to open video/source: {videoInput}")

        # tenta obter total frames (pode ser 0 para streams)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Inference: ", unit="frame", dynamic_ncols=True)

        # não inicialize VideoWriter até ter o primeiro frame (evita dimensões 0 para stream)
        out = None
        writer_inited = False

        # fps fallback
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        frame_idx = 0
        reconnect_tries = 0
        window_name = "Inference (press q to quit)"

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # se for stream, tenta reconectar algumas vezes
                    if is_stream and reconnect_tries < reconnection_attempts:
                        reconnect_tries += 1
                        cap.release()
                        wait = reconnection_backoff * reconnect_tries
                        time.sleep(wait)
                        cap = _open_cap(videoInput)
                        # atualizar fps possível
                        fps = cap.get(cv2.CAP_PROP_FPS) or fps
                        continue
                    # caso não seja stream ou reconexões esgotadas -> fim
                    break

                # reset reconexão após frame bem lido
                reconnect_tries = 0
                frame_idx += 1

                # inicializar writer APENAS se for para salvar e ainda não inicializado
                if save_output and not writer_inited:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = videoOutput or "output.mp4"
                    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    writer_inited = True

                h, w = frame.shape[:2]
                # ajustar input size do YuNet para o frame atual
                try:
                    self.yunetDetector.setInputSize((w, h))
                except Exception:
                    # alguns bindings antigos podem não ter setInputSize; ignora se fail
                    pass

                # detectar faces com YuNet
                try:
                    _, faces = self.yunetDetector.detect(frame)
                except Exception:
                    faces = None

                if faces is not None:
                    # faces: array Nx? onde face[:4] é x,y,w,h
                    for face in faces:
                        x, y, fw, fh = map(int, face[:4])
                        x1 = max(0, x - 10)
                        y1 = max(0, y - 10)
                        x2 = min(w, x + fw + 10)
                        y2 = min(h, y + fh + 10)

                        ROI = frame[y1:y2, x1:x2]
                        if ROI.size == 0:
                            continue

                        # preprocess + inferência ONNX
                        try:
                            blob = preprocessForModel(ROI)  # espera formato compatível com seu modelo
                            embedding = self.session.run([self.outputName], {self.inputName: blob})[0].flatten()
                            embedding = l2_normalize(embedding)
                        except Exception as e:
                            # falha na inferência do embedding -> marca unknown
                            embedding = None

                        predictedName = "Unknown"
                        bestScore = 0.0

                        if embedding is not None and databaseEmbedding.size > 0:
                            # produto escalar (similaridade) — databaseEmbedding shape: (N, D)
                            try:
                                scores = np.dot(databaseEmbedding, embedding)
                                bestIndex = int(np.argmax(scores))
                                bestScore = float(scores[bestIndex])
                                if bestScore >= recognizerScoreThreshold and bestIndex < len(databaseNames):
                                    predictedName = databaseNames[bestIndex]
                                else:
                                    predictedName = "Unknown"
                            except Exception:
                                predictedName = "Unknown"
                                bestScore = 0.0

                        color = colorRecognized if bestScore >= recognizerScoreThreshold else colorUnrecognized
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{predictedName} {bestScore:.2f}" if drawConfidence else predictedName
                        cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # atualizar progress bar
                pbar.update(1)

                # mostrar frame (se permitido)
                if display:
                    try:
                        cv2.imshow(window_name, frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error:
                        # ambiente sem display (headless) -> fallback
                        display = False

                # salvar frame (se solicitado)
                if save_output and writer_inited and out is not None:
                    out.write(frame)

        finally:
            pbar.close()
            try:
                cap.release()
            except Exception:
                pass
            if writer_inited and out is not None:
                try:
                    out.release()
                except Exception:
                    pass
            if display:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
