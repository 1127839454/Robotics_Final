import os
import sys
import time
import cv2
import numpy as np
import torch

# Add project directories to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Import YOLOv5 utilities
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# Import custom camera interface
from Server.camera import Camera

# Mapping from object names to class indices
object_to_class_index = {"redBall": 0, "cup": 1}

def load_yolo_model(weights_path='model/converted_best.pt', device='cpu'):
    """
    Load the YOLOv5 model with given weights and device.
    Returns model, device, and image size configuration.
    """
    device = select_device(device)
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    stride = model.stride
    imgsz = check_img_size(640, s=stride)
    return model, device, imgsz


def detect_objects(cam, model, device, imgsz, target_classes=None, conf_thres=0.6, iou_thres=0.45):
    """
    Capture a frame from camera, run YOLOv5 inference, and return detections.
    Each detection is a dict: {'class', 'confidence', 'bbox':[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]}.
    """
    start_time = time.time()

    # Get a frame (JPEG bytes) and decode
    frame_bytes = cam.get_frame()
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        return [], None

    # Preprocess image
    img = letterbox(frame, new_shape=imgsz)[0]
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=target_classes)

    detections_list = []
    # Postprocess detections
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            det = det.cpu().numpy()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(float, xyxy)
                detections_list.append({
                    'class': int(cls),
                    'confidence': float(conf),
                    'bbox': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                })

    # Frame rate control (~1 FPS)
    elapsed = time.time() - start_time
    time.sleep(max(0, 1.0 - elapsed))

    return detections_list, frame
    
def main():
    # Load vision model
    weights_path = os.path.join(PARENT_DIR, 'model', 'converted_best.pt')
    model_video, device, imgsz = load_yolo_model(weights_path)

    # Start camera stream
    cam = Camera()
    cam.start_stream()

    # Simulate voice command (replace actual recognition)
    # e.g., pretend user said "find ball"
    result_voice = {"type": "object", "name": "redBall"}
    print(f"[INFO] Simulated voice command: find {result_voice['name']}")

    if result_voice["type"] == "object":
        object_name = result_voice["name"]
        class_index = object_to_class_index.get(object_name)
        print(f"[INFO] Looking for object: {object_name} (class {class_index})")

        # Loop until at least one detection
        result_video = []
        frame = None
        while not result_video:
            result_video, frame = detect_objects(
                cam, model_video, device, imgsz,
                [class_index] if class_index is not None else None
            )
            # Display intermediate frame
            if frame is not None:
                cv2.imshow("Detection", frame)
                cv2.waitKey(1)
                print(f"[DEBUG] Detections: {result_video}")

        # Draw final detections
        for det in result_video:
            (x1, y1), (_, _), (x2, y2), _ = det['bbox']
            conf = det['confidence']
            cls_id = det['class']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{cls_id} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Detection", frame)
        cv2.waitKey(0)
        print("[RESULT]", result_video)

    # Cleanup
    cam.stop_stream()
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
