import io
from PIL import Image
from fastapi import File, FastAPI, Response
from ultralytics import YOLO
import cv2
import pandas as pd

model = YOLO("yolov8n.pt")

app = FastAPI(
    title="Custom YOLOV8 Machine Learning API",
    description="""Obtain object value out of image
                    and return image""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.post("/objectdetection")
async def detect_cars_return_img(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = model.track(source=input_image, persist=True, classes=[2, 5, 7], tracker='bytetrack.yaml')
    res_plotted = results[0].plot()
    success, encoded_image = cv2.imencode('.jpg', res_plotted)
    content = encoded_image.tobytes()

    # bird_view_raw = bird_view(results[0].id, results[0].cls, results[0].boxes.xyxy)
    # success, bird_view_encoded = cv2.imencode('.jpg', bird_view_raw)
    # bird_view_bytes = bird_view_encoded.tobytes()
    
    # print(content == file)
    return Response(content=content, media_type="image/jpg")#, Response(content=bird_view_bytes, media_type="image/jpg")


    # for r in results:
    #     boxes = r.boxes.xyxy  # Boxes object for bbox outputs
    #     conf = r.boxes.conf  # Masks object for segmenation masks outputs
    #     cls = r.boxes.cls  # Class probabilities for classification outputs
    # results_json = json.loads(str(results[0].names))
    # for obj in results_json:

    # Encoding already annotated image(BoundingBoxes, colors, classes, confidences)