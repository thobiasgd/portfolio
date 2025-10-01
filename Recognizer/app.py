import argparse
import os
import config

def parseArgs():
    p = argparse.ArgumentParser(description="Tool: train DB or run inference")
    sub = p.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="Build databse (scan dataset and save database.json)")
    t.add_argument("dataset", help="Dataset folder (dataset/person/*.jpg)")
    t.add_argument("-d", "--detector", default=None, help="Detector path (optional)")

    i = sub.add_parser("infer", help="Run inference in video")
    i.add_argument("input", help="Input video")
    i.add_argument("output", nargs="?", default="output.mp4", help="Output video")
    i.add_argument("-d", "--detector", default=None, help="Detector (optional)")
    i.add_argument("-r", "--recognizer", default=None, help="Recognizer (optional)")
    i.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold limit")

    return p.parse_args()

if __name__ == "__main__":
    args = parseArgs()

    if args.mode == "train":
        from databaseEmbeddingGenerator import buildDatabase
        det_path = args.detector or getattr(config, "DETECTOR_PATH", None)
        buildDatabase(datasetPath=args.dataset,
                      detectorPath=det_path,
                      detectorScoreThreshold=getattr(config, "detectorScoreThreshold", 0.6))

    else:  # infer
        from inference import Inferror
        import os
        det_path = args.detector or (os.path.join(config.modelsPath, config.detectionModel) if hasattr(config, "modelsPath") else None)
        rec_path = args.recognizer or (os.path.join(config.modelsPath, config.recognitionModel) if hasattr(config, "modelsPath") else None)

        inf = Inferror(detectorPath=det_path, recognitionPath=rec_path, recognizerScoreThreshold=args.threshold)
        inf.gettingInference(videoInput=args.input, videoOutput=args.output)
