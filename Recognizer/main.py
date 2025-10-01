import config
from inference import Inferror
import cv2
import argparse
import os

def parseArgs():
    parser = argparse.ArgumentParser(description="Exemplo mínimo: parse e print")
    parser.add_argument("input", help="Path where the video is located.")
    parser.add_argument("output", nargs="?", default="output.mp4", help="Path where the video with inference will be saved.")
    parser.add_argument("threshold", nargs="?", default=0.6, help="Threshold from recognition model.")
    parser.add_argument("--detector", "-d", help="Detection model path.", default=None)
    parser.add_argument("--recognizer", "-r", help="Recognition model path.", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    inferror = Inferror(detectorPath = args.detector or config.DETECTOR_PATH,
                        recognitionPath = args.recognizer or config.RECOGNIZER_PATH,
                        recognizerScoreThreshold = args.threshold)

    inferror.gettingInference(videoInput = args.input, videoOutput = args.output)