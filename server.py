import io
from PIL import Image
from fastapi import File, FastAPI, Response
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

app = FastAPI()


@app.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = model.predict(source=input_image, classes=[2, 5, 7])
    res_plotted = results[0].plot()

    # for r in results:
    #     boxes = r.boxes.xyxy  # Boxes object for bbox outputs
    #     conf = r.boxes.conf  # Masks object for segmenation masks outputs
    #     cls = r.boxes.cls  # Class probabilities for classification outputs
    # results_json = json.loads(str(results[0].names))
    # for obj in results_json:

    # Encoding already annotated image(BoundingBoxes, colors, classes, confidences)
    success, encoded_image = cv2.imencode('.jpg', res_plotted)
    content = encoded_image.tobytes()
    print(content == file)
    return Response(content=content, media_type="image/jpg")

