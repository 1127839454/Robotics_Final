import sys
import os

YOLOV5_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")


if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)


from voice_video_recognition import main

if __name__ == "__main__":
    print("[INFO] Launching voice + vision system from gshcode_test.py...")
    # main()
    weights_path = os.path.join(PARENT_DIR, 'model', 'converted_best.pt')
    model_video, device, imgsz = load_yolo_model(weights_path)
    model_video.eval()
    cam = Camera()
    cam.start_stream()

    vosk_model_path = os.path.join(PARENT_DIR, 'model', 'vosk-model-small-en-us-0.15')
    model_voice = Model(vosk_model_path)
    recognizer = KaldiRecognizer(model_voice, 16000)

    try:
        while True:
            result_voice = voice_recognition(recognizer)
            if not result_voice:
                print("[WARN] No valid voice command.")
                continue

            if result_voice["type"] == "command" and result_voice["name"] == "goback":
                direction = result_voice["direction"]
                print(f"[INFO] Got go-back command, direction: {direction}")

            elif result_voice["type"] == "object":
                object_name = result_voice["name"]
                class_index = object_to_class_index.get(object_name, None)
                print(f"[INFO] Looking for object: {object_name}")

                result_video = []

                while not result_video:
                    result_video, frame = detect_objects(
                        cam,
                        model_video,
                        device,
                        imgsz,
                        [class_index] if class_index is not None else None
                    )

                    cv2.imshow("Detection", frame)
                    cv2.waitKey(1)
                    print(result_video)

                for det in result_video:
                    x1, y1 = map(int, det["bbox"][0])
                    x2, y2 = map(int, det["bbox"][2])
                    conf = det["confidence"]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{det["class"]} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Detection", frame)
                cv2.waitKey(0)

                print("[RESULT]", result_video)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print("[INFO] Cleaning up camera resources...")
        cam.stop_stream()
        cam.close()
        print("[INFO] Done.")
        cv2.destroyAllWindows()
