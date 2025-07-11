import asyncio
import json
from aiohttp import web

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

# Initialize object detection model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval().to(DEVICE)
if DEVICE == "cuda":
    model.half()

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

relay = MediaRelay()
pcs: set[RTCPeerConnection] = set()


def detect_objects(frame):
    """Return a list of detected object labels."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (320, 320))
    tensor = F.to_tensor(resized).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        outputs = model([tensor])[0]
    labels = []
    for score, label in zip(outputs["scores"], outputs["labels"]):
        if score >= 0.5:
            labels.append(COCO_INSTANCE_CATEGORY_NAMES[label])
    return labels


async def index(request: web.Request) -> web.Response:
    """Serve the client HTML page."""
    content = (request.app["client_html"])
    return web.Response(content_type="text/html", text=content)


async def offer(request: web.Request) -> web.Response:
    """Handle incoming WebRTC offer."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    results_channel = pc.createDataChannel("results")

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            local_track = relay.subscribe(track)

            async def run() -> None:
                while True:
                    frame = await local_track.recv()
                    img = frame.to_ndarray(format="bgr24")
                    labels = detect_objects(img)
                    if labels and results_channel.readyState == "open":
                        results_channel.send(json.dumps({"objects": labels}))
            asyncio.create_task(run())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def on_shutdown(app: web.Application) -> None:
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def main() -> None:
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    with open("client.html", "r", encoding="utf-8") as f:
        app["client_html"] = f.read()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    print("Server running on http://0.0.0.0:8080")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())