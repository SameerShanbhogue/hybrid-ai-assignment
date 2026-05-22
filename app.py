from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from hybrid_inference import predict_hybrid
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="Hybrid AI Decision API", description="Complaint Escalation Risk System")

class ComplaintInput(BaseModel):
    customer_id: str
    complaint_text: str
    customer_tenure: float
    previous_complaints: int
    image_base64: Optional[str] = None

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Hybrid AI System Running"}

@app.post("/predict")
def predict(payload: ComplaintInput):
    # Build input dict for hybrid inference
    input_data = {
        "text": payload.complaint_text,
        "tabular": {
            "customer_tenure": payload.customer_tenure,
            "previous_complaints": payload.previous_complaints
        },
        "text_sequence": payload.complaint_text,
        "image_available": payload.image_base64 is not None
    }
    
    # Decode image if provided
    if payload.image_base64:
        try:
            img_data = base64.b64decode(payload.image_base64)
            img = Image.open(BytesIO(img_data)).resize((64, 64))
            img_array = np.array(img)
            if img_array.shape[-1] == 4:  # RGBA to RGB
                img_array = img_array[:, :, :3]
            input_data["image_array"] = img_array
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    result = predict_hybrid(input_data)
    
    # Add customer_id to response
    full_response = {
        "customer_id": payload.customer_id,
        **result
    }
    return full_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)