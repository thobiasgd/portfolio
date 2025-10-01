import argparse
import os
import config

def parseArgs():
    p = argparse.ArgumentParser(description="Tool: train DB or run inference")
    sub = p.add_subparsers(dest="mode", required=True)

    # ----------------- TRAIN -----------------
    t = sub.add_parser("train", help="Build database (scan dataset and save database.json)")
    t.add_argument("dataset", help="Dataset folder (dataset/person/*.jpg)")
    t.add_argument("-d", "--detector", default=None, help="Detector path (optional)")

    # ----------------- INFER (file) -----------------
    i = sub.add_parser("infer", help="Run inference on a video file")
    i.add_argument("input", help="Input video file path")
    i.add_argument("output", nargs="?", default="output.mp4", help="Output video file (will be written unless --no-save)")
    i.add_argument("-d", "--detector", default=None, help="Detector (optional)")
    i.add_argument("-r", "--recognizer", default=None, help="Recognizer (optional)")
    i.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold limit for recognition")
    i.add_argument("--no-save", action="store_true", help="Do not save output file (infer will run and optionally display)")
    i.add_argument("--display", action="store_true", help="Show output window while running (default: False for non-interactive)")
    i.add_argument("--backend", default=None, help="Optional cv2 backend hint (e.g. 'ffmpeg')")

    # ----------------- STREAM (RTSP / file as live) -----------------
    s = sub.add_parser("stream", help="Run inference on a live stream (file path or rtsp://...)")
    s.add_argument("source", help="Video source (file path or rtsp URL)")
    s.add_argument("-d", "--detector", default=None, help="Detector model path (optional)")
    s.add_argument("-r", "--recognizer", default=None, help="Recognizer model path (optional)")
    s.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold limit for recognition")
    s.add_argument("--no-save", action="store_true", help="Do not save output (default for stream)")
    s.add_argument("--headless", action="store_true", help="Run without display (implies --no-save)")
    s.add_argument("--display", action="store_true", help="Show output window while running (if not headless)")
    s.add_argument("--backend", default=None, help="Optional cv2 backend hint (e.g. 'ffmpeg')")

    return p.parse_args()

if __name__ == "__main__":
    args = parseArgs()

    # ------------- TRAIN -------------
    if args.mode == "train":
        from databaseEmbeddingGenerator import buildDatabase
        det_path = args.detector or getattr(config, "DETECTOR_PATH", None)
        buildDatabase(datasetPath=args.dataset,
                      detectorPath=det_path,
                      detectorScoreThreshold=getattr(config, "detectorScoreThreshold", 0.6))

    elif args.mode == "infer":
        from inference import Inferror
        import os

        det_path = args.detector or getattr(config, "DETECTOR_PATH", None)
        if det_path is None and hasattr(config, "modelsPath") and hasattr(config, "detectionModel"):
            det_path = os.path.join(getattr(config, "modelsPath"), getattr(config, "detectionModel"))

        rec_path = args.recognizer or getattr(config, "RECOGNITION_PATH", None)
        if rec_path is None and hasattr(config, "modelsPath") and hasattr(config, "recognitionModel"):
            rec_path = os.path.join(getattr(config, "modelsPath"), getattr(config, "recognitionModel"))

        inf = Inferror(detectorPath=det_path,
                       recognitionPath=rec_path,
                       detectorScoreThreshold=getattr(config, "detectorScoreThreshold", 0.6),
                       recognizerScoreThreshold=args.threshold)

        save_output = not args.no_save
        display = bool(args.display)
        backend = args.backend

        inf.gettingInference(videoInput=args.input,
                             videoOutput=args.output if save_output else None,
                             is_stream=False,
                             save_output=save_output,
                             display=display,
                             backend=backend)

    elif args.mode == "stream":
        from inference import Inferror
        import os

        det_path = args.detector or getattr(config, "DETECTOR_PATH", None)
        if det_path is None and hasattr(config, "modelsPath") and hasattr(config, "detectionModel"):
            det_path = os.path.join(getattr(config, "modelsPath"), getattr(config, "detectionModel"))

        rec_path = args.recognizer or getattr(config, "RECOGNIZER_PATH", None)
        if rec_path is None and hasattr(config, "modelsPath") and hasattr(config, "recognitionModel"):
            rec_path = os.path.join(getattr(config, "modelsPath"), getattr(config, "recognitionModel"))

        inf = Inferror(detectorPath=det_path,
                    recognitionPath=rec_path,
                    detectorScoreThreshold=getattr(config, "detectorScoreThreshold", 0.6),
                    recognizerScoreThreshold=args.threshold)

        if args.headless:
            save_output = False
            display = False
        else:
            save_output = False         
            display = True              

        backend = args.backend

        inf.gettingInference(videoInput=args.source,
                            videoOutput=None if not save_output else "stream_output.mp4",
                            is_stream=True,
                            save_output=save_output,
                            display=display,
                            backend=backend)

