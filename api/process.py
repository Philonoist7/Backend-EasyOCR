import os
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral

# Initialize FastAPI app
app = FastAPI()

# --- Your CORS settings ---
origins = [
    "https://philonoist7.github.io",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Mistral API Key securely from environment variables
try:
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
except Exception as e:
    print(f"ERROR: Could not initialize Mistral client. Is MISTRAL_API_KEY set? Error: {e}")
    client = None

@app.post("/api/process")
async def process_pdf(pdf_file: UploadFile = File(...)):
    if not client:
        raise HTTPException(status_code=500, detail="Mistral client not initialized. Check server logs.")
    
    # UPDATED: Stricter check for PDF only
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")

    try:
        file_bytes = await pdf_file.read()
        b64_string = base64.b64encode(file_bytes).decode('utf-8')

        # Hardcode the document type for security and clarity
        document_url = f"data:application/pdf;base64,{b64_string}"

        resp = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": document_url 
            }
        )
        
        full_markdown = "\n\n".join([page.markdown for page in resp.pages])

        return {"markdown_content": full_markdown}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during OCR processing: {e}")
