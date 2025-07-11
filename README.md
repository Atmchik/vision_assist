# Vision Assist

This example demonstrates a simple real-time object and path detection tool intended to provide
voice feedback for visually impaired users. It relies on a pre-trained object detection model
(SSDLite MobileNet) from `torchvision` and uses OpenCV for image capture and basic walkway
estimation. Feedback is spoken aloud via `pyttsx3` so that the script can run offline.
Detected objects are highlighted with bounding boxes and announced along with an
approximate distance estimate.

The script is located in `vision_assist.py` and is meant to run on either a laptop or a mobile
Python environment (such as Termux on Android) that has access to a camera.
It automatically uses a GPU if one is available.

```
pip install torch torchvision opencv-python pyttsx3
python vision_assist.py --source 0
```

If performance is low, try reducing the detection resolution or skipping
frames. For example, process a 320×320 image every other frame:

```
python vision_assist.py --source 0 --resize 320 --skip-frames 1
```

If you notice a large delay between movement and what you see on screen, try
reducing the internal camera buffer:

```
python vision_assist.py --no-buffer
```

Use the `--source` argument to select another camera or a network stream. For
example with the Android *IP Webcam* app you might run:

```
python vision_assist.py --source http://192.168.0.20:8080/video
```

If the script fails to open the stream, double-check that the URL points
directly to the video feed (most apps expose it under `/video`).

python vision_assist.py --no-buffer
```

Use the `--source` argument to select another camera or a network stream. For
example with the Android *IP Webcam* app you might run:

```
python vision_assist.py --source http://192.168.0.20:8080/video
```

If the script fails to open the stream, double-check that the URL points
directly to the video feed (most apps expose it under `/video`).

## Running on Android

1. Install the [Termux](https://termux.dev) app.
2. Inside Termux, run `pkg install python ffmpeg` and clone or copy this
   repository.
3. Install dependencies with `pip install torch torchvision opencv-python pyttsx3`.
4. Run the script with `python vision_assist.py --source 0`.

The script will automatically use the phone's rear camera and provide the same
voice feedback as on a laptop.

## WebRTC server

To stream video from a mobile browser to the `vision_assist` model, you can run the simple WebRTC server included in this folder.
It receives a video track over WebRTC, performs object detection on the server and sends back the detected labels over a data channel.

1. Install the extra dependencies:
   ```bash
   pip install aiohttp aiortc opencv-python torch torchvision
   ```
2. Start the server:
   ```bash
   python webrtc_server.py
   ```
3. Open `http://<server-ip>:8080` on your phone and allow camera access.
   Press **Start** to begin streaming. Detected objects will appear on the page as they are recognized by the server.
