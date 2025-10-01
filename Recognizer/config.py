import os

datasetPath = 'dataset'
modelsPath = 'models/'
recognitionModel = "recognition_resnet27.onnx"
detectionModel = "face_detection_yunet_2023mar.onnx"
videoInput = "input.mp4"
videoOutput = "output.mp4"
detectorScoreThreshold = 0.7
recognizerScoreThreshold = 0.5
colorRecognized = (0, 255, 0)
colorUnrecognized = (0, 200, 200)
NPZ_CACHE = "bank_cache.npz"
JSON_DB = "database.json"
draw_confidence = True
DETECTOR_PATH = os.path.join(modelsPath, detectionModel)
RECOGNIZER_PATH = os.path.join(modelsPath, recognitionModel)
