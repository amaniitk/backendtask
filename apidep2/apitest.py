import requests
import cv2
import numpy as np
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

app = FastAPI()

subscription_key = "8f533c0cd1fd4d90ba297bb1c65778b6"
endpoint = "https://amank.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

class ImageRequest(BaseModel):
    image_url: str

class ExtractedText(BaseModel):
    text: str

@app.post("/extract_text", response_model=ExtractedText)
def extract_text(request: ImageRequest):
    response = requests.get(request.image_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid image URL")

    image = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, gray)

    with open(temp_image_path, "rb") as image_stream:
        ocr_result = computervision_client.read_in_stream(image_stream, raw=True)

    operation_location = ocr_result.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    time.sleep(5) 
    result = computervision_client.get_read_result(operation_id)

    if result.status != "succeeded":
        raise HTTPException(status_code=500, detail="OCR operation failed")

    extracted_text = ""
    for line in result.analyze_result.read_results[0].lines:
        extracted_text += line.text + "\n"

    return ExtractedText(text=extracted_text.strip())
