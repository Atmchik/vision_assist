"Simple vision helper for blind or low-vision users."

#Captures video frames from a webcam or IP camera, detects objects using a
#pretrained SSDLite MobileNet V3 model and provides voice feedback when
#objects are too close. Additionally attempts to keep the user centered on
#a walkway using basic edge detection.

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import pyttsx3
import threading
import queue
import time

# Change this to an IP camera URL if required. For Android IP Webcam try:
#   http://PHONE_IP:8080/video
VIDEO_SOURCE = 0 #'https://192.168.10.3:8080/video'  # 0 means default local webcam

def parse_args():
    """Return command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Vision assist demo")
    parser.add_argument(
        "--source",
        default=VIDEO_SOURCE,
        help="Camera index or URL to read frames from",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=320,
        help="Resize frame to this resolution for detection",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of frames to skip between detections",
    )
    parser.add_argument(
        "--no-buffer",
        action="store_true",
        help="Reduce camera latency by minimizing internal buffering",
    )

    return parser.parse_args()



# Load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    print("Using CPU")
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()
model.to(DEVICE)
if DEVICE == "cuda":
    model.half()

# COCO dataset class names for the detection model
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

engine = pyttsx3.init()

_speech_queue: queue.Queue = queue.Queue()


def _speech_worker() -> None:
    while True:
        text = _speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        _speech_queue.task_done()


speech_thread = threading.Thread(target=_speech_worker, daemon=True)
speech_thread.start()

def speak(text: str) -> None:
    _speech_queue.put(text)


# Track last time each object class was announced
_last_spoken: dict[str, float] = {}
SPEAK_INTERVAL = 1.5  # seconds


def get_distance(box, frame_width):
    """Rudimentary distance approximation based on box width."""
    box_width = box[2] - box[0]
    relative_width = box_width / frame_width
    # The closer the object, the bigger the box width ratio
    distance = 1.0 / (relative_width + 1e-6)
    return distance


def analyze_walkway(frame):
    """Return deviation of the walkway center from frame center."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    bottom = edges[int(height * 0.6) :, :]
    moments = cv2.moments(bottom)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
    else:
        cx = width // 2
    return cx - width // 2


def main(args=None):
    args = parse_args() if args is None else args
    cap = cv2.VideoCapture(args.source)
    if args.no_buffer:
        # Minimize latency by keeping only the newest frame in the buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open video source. If using an IP webcam, ensure the URL"
            " points directly to the video stream, e.g. http://ip:port/video"
        )
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if args.skip_frames and frame_count % (args.skip_frames + 1) != 0:
                cv2.imshow("Vision Assist", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (args.resize, args.resize))
            img_tensor = F.to_tensor(resized).to(DEVICE)
            if DEVICE == "cuda":
                img_tensor = img_tensor.half()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
                detections = model([img_tensor])[0]
            scale_x = frame.shape[1] / args.resize
            scale_y = frame.shape[0] / args.resize

            width = frame.shape[1]
            walkway_offset = analyze_walkway(frame)
            now = time.time()
            last_path = _last_spoken.get("_path", 0)
            if now - last_path > SPEAK_INTERVAL:
                if walkway_offset > 50:
                    speak("Move right to stay on the path")
                    _last_spoken["_path"] = now
                elif walkway_offset < -50:
                    speak("Move left to stay on the path")
                    _last_spoken["_path"] = now

            for box, score, label in zip(
                detections["boxes"], detections["scores"], detections["labels"]
            ):
                if score < 0.5:
                    continue
                box = box.to("cpu").numpy()
                box = [
                    int(box[0] * scale_x),
                    int(box[1] * scale_y),
                    int(box[2] * scale_x),
                    int(box[3] * scale_y),
                ]
                class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
                dist = get_distance(box, width)
                # draw the bounding box and label
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{class_name} {dist:.1f}m",
                    (box[0], max(box[1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                now = time.time()
                last = _last_spoken.get(class_name, 0)
                if now - last > SPEAK_INTERVAL:
                    speak(f"{class_name} ahead, {dist:.1f} meters")
                    _last_spoken[class_name] = now

            # Visualization for debugging (optional)
            cv2.imshow("Vision Assist", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        _speech_queue.put(None)
        speech_thread.join()


if __name__ == "__main__":
    main()
