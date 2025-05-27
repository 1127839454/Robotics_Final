import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from models.common import DetectMultiBackend
from utils.torch_utils import select_device


from Server.camera import Camera


from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import numpy as np
import cv2
import torch
import usb.core
import time


# Create a queue to store incoming audio data
q = queue.Queue()
object_to_class_index = {"redBall": 0, "cup": 1}
voice_aliases = {
    "ball": "redBall",
    "red ball": "redBall",
    "red buoy": "redBall",
    "buoy": "redBall"
}
# Callback function to feed microphone data into the queue
def callback(indata, frames, time, status):
    if status:
        print("Microphone status:", status)
    q.put(bytes(indata))
    
def load_yolo_model(weights_path='model/best.pt', device='cpu'):
    device = select_device(device)
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    stride = model.stride
    imgsz = check_img_size(640, s=stride)
    return model, device, imgsz

def extract_object(sentence):
    sentence = sentence.lower()
    if "find" not in sentence:
        return None

    # Split by "find" and take the part after it
    parts = sentence.split("find", 1)
    after_find = parts[1].strip()  # e.g., "my phone"

    # Remove common words like "my", "the", "a"
    stop_words = {"my", "the", "a", "an", "me", "to", "please"}
    words = after_find.split()

    # Return the first non-stop word as the object
    for word in words:
        if word not in stop_words:
            return word

    return None  # If no object found

def get_respeaker_direction(dev):
    if dev:
        try:
            ret = dev.ctrl_transfer(0xC0, 0x0A, 0, 0, 1)
            return int(ret[0])
        except Exception as e:
            print(f"[DOA Error] {e}")
    return None

def is_come_back_command(sentence,dev):
    sentence = sentence.lower()
    come_back_keywords = ["come back", "return", "go back", "back to me", "come to me"]

    for phrase in come_back_keywords:
        if phrase in sentence:
            direction = get_respeaker_direction(dev)
            return True, direction
    return False,None

# Load the Vosk model
def load_vosk_model(model_path="model/vosk-model-small-en-us-0.15"):
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        exit(1)
    try:
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 16000)
        print("Vosk model loaded successfully.")
        return recognizer
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

#Start microphone and recognize speech
def start_microphone(recognizer):
    print("Say something! (Press Ctrl+C to stop)")
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    print("?You said:", result["text"])
                    return result["text"]
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Runtime error: {e}")

def voice_recognition(recognizer):
    result = start_microphone(recognizer)
    object_name = extract_object(result)
    if object_name:
        standard_name = voice_aliases.get(object_name.lower())
    else:
        standard_name = None

    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018) #方位
    go_back = is_come_back_command(result,dev)
    if standard_name:
        print(f"Extracted object: {standard_name}")
        return {"type": "object", "name": standard_name}
    elif go_back[0]:
        print("Obtain the return instruction")
        return {"type": "command", "name": "goback", "direction": go_back[1]}
    else:
        print("No object found.")
        return None

def detect_objects(cam, model, device, imgsz, target_classes=None):
    detections_list = []

    start_time = time.time()

    frame_bytes = cam.get_frame()
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        return [], None

    img = letterbox(frame, new_shape=imgsz)[0]
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.45, classes=target_classes)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            det = det.cpu().numpy()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(float, xyxy)
                detection_info = {
                    'class': int(cls),
                    'confidence': float(conf),
                    'bbox': [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2)
                    ]
                }
                detections_list.append(detection_info)

    # frame rate
    elapsed = time.time() - start_time
    time.sleep(max(0, 1.0 - elapsed))

    return detections_list, frame


def main():
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




if __name__ == "__main__":
    main()

   
'''
    while True:
        result_video, frame = detect_objects(
            cam,
            model_video,
            device,
            imgsz,
            [class_index] if class_index is not None else None
        )

        cv2.imshow("Detection", frame)
        cv2.waitKey(1)
        
        if result_video:
            for det in result_video:
                x1, y1 = map(int, det["bbox"][0])
                x2, y2 = map(int, det["bbox"][2])
                conf = det["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'{det["class"]} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Detection", frame)
            cv2.waitKey(1)
            print("[RESULT]", result_video)

        print("[VOICE] Say 'finish' to stop detection...")
        finish_text = start_microphone(recognizer)
        if the progress finished :
            break
'''
