import io
from PIL import Image
from fastapi import File, FastAPI, Response
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np

from BirdViewTransformer import BirdViewTransformer

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
    input_image = Image.open(io.BytesIO(file))#.convert("RGB")
    results = model.track(source=input_image, persist=True, classes=[2, 5, 7], tracker='bytetrack.yaml')
    res_plotted = results[0].plot()
    success, encoded_image = cv2.imencode('.jpg', res_plotted)
    content = encoded_image.tobytes()

    bounding_boxex = np.array(results[0].boxes.xyxy)
    labels = np.array(['car' if label == 2 else 'bus' if label == 5 else 'truck' for label in results[0].boxes.cls])
    obj_ids = np.array(results[0].boxes.id)

    input_image_np = np.array(input_image)
    IMAGE_H, IMAGE_W = input_image_np.shape[:2]
    
    points = np.array([
        [0, IMAGE_H],                # Top Left 
        [IMAGE_W, IMAGE_H],                # Top Right
        [IMAGE_W, 0],            # Bottom Right
        [0, 0],            # Bottom Left
    ])

    transformer_cam = BirdViewTransformer(
        input_image_np,
        points
    ) 

    image_normalized = transformer_cam.bird_view_transformation(
        bounding_boxex,
        labels,
        obj_ids
    )
    print(transformer_cam.bb_centres_transformed_prev)


    success, bird_view_encoded = cv2.imencode('.jpg', image_normalized)
    bird_view_bytes = bird_view_encoded.tobytes()
    
    print(transformer_cam.counter)

    #responce = str(content) + str('') + str(bird_view_bytes)+ str('}')  + str(transformer_cam.counter)
    responce = bird_view_bytes
    # print(content == file)
    return Response(content=responce, media_type="image/jpg")#, 
    #return Response(content=bird_view_bytes, media_type="image/jpg")


    # for r in results:
    #     boxes = r.boxes.xyxy  # Boxes object for bbox outputs
    #     conf = r.boxes.conf  # Masks object for segmenation masks outputs
    #     cls = r.boxes.cls  # Class probabilities for classification outputs
    # results_json = json.loads(str(results[0].names))
    # for obj in results_json:

    # Encoding already annotated image(BoundingBoxes, colors, classes, confidences)