import os
import json
import tempfile
import time
import logging
import httpx
from fastapi import FastAPI
from pymongo import MongoClient
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from datetime import datetime
import pytz
from anthropic import Anthropic
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from pydantic import BaseModel

class BatchRequest(BaseModel):
    batchname: list[str]
    status: str

# Load environment variables
load_dotenv()

# Setting API keys and endpoints
form_recognizer_endpoint = os.getenv("AZURE_OCR_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_OCR_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")

# Initialize clients
document_analysis_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint, credential=AzureKeyCredential(form_recognizer_key)
)
anthropic_client = Anthropic(api_key=anthropic_api_key)
mongo_client = MongoClient(mongo_uri)
db = mongo_client[database_name]
collection = db[collection_name]

# Scheduler configuration
executors = {'default': ThreadPoolExecutor(1)}
scheduler = BackgroundScheduler(executors=executors, job_defaults={'coalesce': False, 'max_instances': 1})

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def chunk_document(document_path, chunk_size=1024 * 1024):  # 1 MB chunks
    with open(document_path, 'rb') as file:
        while chunk := file.read(chunk_size):
            yield chunk

def analyze_document_chunked(document_path):
    extracted_data = []
    for chunk in chunk_document(document_path):
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", document=chunk)
        result = poller.result()
        extracted_data.extend([{"page": " ".join([line.content for line in page.lines])} for page in result.pages])
    return extracted_data

def get_claude_response(prompt):
    for attempt in range(3):  # Retry logic for Claude API
        try:
            message = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            logging.warning(f"Claude API call failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    logging.error("Failed to get a valid response from Claude after 3 attempts.")
    return None

def process_ocr_output(ocr_output):
    corrected_output_parts = []
    for page in ocr_output:
        prompt = f"Fix the errors in this OCR output and return it in JSON format:\n{page['page']}"
        response = get_claude_response(prompt)
        if response:
            corrected_output_parts.append({"page": response})
        else:
            corrected_output_parts.append({"page": "Error: Could not process this page."})
    return corrected_output_parts

def get_metadata(object_id, collection, fields):
    document = collection.find_one({"_id": ObjectId(object_id)})
    if not document:
        raise ValueError(f"Document not found: {object_id}")

    json_data = document.get("json_data", [])
    combined_content = " ".join([list(page.values())[0] for page in json_data])

    # Construct the fields_to_extract section
    fields_to_extract = "\n".join(fields.keys())

    prompt = f"""
    You are tasked with extracting specific metadata fields from a document. Your goal is to accurately extract all required fields from the given document text.

    Here is the full text of the document:
    <document_text>
    {combined_content}
    </document_text>

    Please extract information for the following fields:

    <fields_to_extract>
    {fields_to_extract}
    </fields_to_extract>

    Please follow these instructions to extract the required information:

    1. Carefully read through the entire document text.
    2. For each field listed above, search for relevant information in the document.
    3. Extract the exact text that corresponds to each field.
    4. If a field's information is not found or is unclear, mark it as "" respectively.
    5. Provide a confidence score (0-100) for each extracted field, where 0 means no confidence and 100 means absolute certainty.
    6. Ensure that any dates are formatted as "MM/DD/CCYY". If the date is not in this format, convert it before extracting.

    Format your output as follows:
    <extracted_fields>
    Field Name: Extracted Value
    Confidence: [0-100]

    Field Name: Extracted Value
    Confidence: [0-100]

    ...
    </extracted_fields>

    Guidelines for handling missing or unclear information:
    - If a date is not explicitly stated but can be inferred from context, extract it and note "Inferred" in parentheses after the date.
    - For numeric fields (e.g., loan amounts), extract the full number including cents if available.
    - For names, extract full names as they appear in the document.
    - If a field has multiple relevant entries, include all of them separated by semicolons.

    Remember to be as accurate and thorough as possible in your extraction. Your goal is to provide a complete and reliable set of metadata for this document.
    """

    try:
        response = get_claude_response(prompt)
        return parse_claude_response(response)
    except Exception as e:
        logging.error(f"Failed to get or parse Claude response for metadata: {e}")
        return None

def parse_claude_response(response):
    start = response.find("<extracted_fields>") + len("<extracted_fields>")
    end = response.find("</extracted_fields>")
    extracted_fields = response[start:end].strip().split("\n\n")
    parsed_data = {}
    for field in extracted_fields:
        lines = field.split("\n")
        if len(lines) >= 2:
            field_name, value = lines[0].split(": ", 1)
            confidence = 0
            if len(lines) > 1 and lines[1].startswith("Confidence:"):
                try:
                    confidence = int(lines[1].split(": ", 1)[1])
                except ValueError:
                    logging.warning(f"Could not parse confidence for field: {field_name}")
            parsed_data[field_name] = {"value": value, "confidence": confidence}
    return parsed_data

def get_metadata_for_final_assignment(object_id, collection):
    fields = {
        "Record Type `Z`": "",
        "Document Type (Must be populated with one of the valid codes)": "",
        "FIPS Code": "",
        "MERS Indicator (ASSIGNEE)": "",
        "RECORD ID M = MAIN Record (default value for all non-addendum records); A = APN Addendum; D= DOT Addendum": "",
        "Assignment Recording Date": "",
        "Assignment EFFECTIVE or CONTRACT Date": "",
        "Assignment Document Number": "",
        "Assignment Book Number": "",
        "Assignment Page Number": "",
        "Multiple Page Image Flag": "",
        "LPS Image Identifier": "",
        "Original Deed of Trust (`DOT`) Recording Date": "",
        "Original Deed of Trust (`DOT`) Contract Date": "",
        "Original Deed of Trust (`DOT`) Document Number": "", 
        "Original Deed of Trust (`DOT`) Book Number": "",
        "Original Deed of Trust (`DOT`) Page Number": "",
        "Original Beneficiary/Lender/Mortgagee/In Favor of/Made By": "",
        "Original Loan Amount": "",
        "Assignor Name(s)": "",
        "Loan Number": "",
        "Assignee(s)": "",
        "MERS (MIN) Number": "",
        "MERS NUMBER PASS VALIDATION": "",
        "Assignee / Pool #": "",
        "MSP Servicer Number and Loan Number": "",
        "Borrower Name(s)": "",
        "Assessor Parcel Number (APN, PIN, PID)": "",
        "Multiple APN Flag": "",
        "Tax Acct ID": "",
        "Full Street Address": "",
        "Unit #": "",
        "City Name": "",
        "State": "",
        "Zip": "",
        "Zip + 4": "",
        "Data Entry Date": "",
        "Data Entry Operator Code": "",
        "Vendor Source Code": ""
    }
    return get_metadata(object_id, collection, fields)

def get_metadata_for_final_release(object_id, collection):
    fields = {
        "Record Type": "",
        "Document Type (Must be populated with one of the valid codes)": "",
        "FIPS Code": "",
        "RECORD ID M": "",
        "Release Recording Date": "",
        "Release Contract Date or Effective Date": "",
        "(Mortgage) Payoff Date (P.O. Date)": "",
        "Release Document Number (Instrument, Reception No)": "",
        "Release Book Number (Folio, Liber,Volume)": "",
        "Release Page Number": "",
        "Multiple Page Image Flag": "",
        "LPS Image Identifier": "",
        "Original Deed of Trust (`DOT`) Recording Date": "",
        "Original Deed of Trust (`DOT`) Contract Date": "",
        "Original Deed of Trust (`DOT`) Document Number": "",
        "Original Deed of Trust (`DOT`) Book Number": "",
        "Original Deed of Trust (`DOT`) Page Number": "",
        "Original Beneficiary/Lender/Mortgagee": "",
        "Original Loan Amount": "",
        "Loan Number": "",
        "Current Beneficiary/Lender/Mortgagee": "",
        "MERS (MIN) Number": "",
        "MERS NUMBER PASS VALIDATION": "",
        "MSP Servicer Number and Loan Number": "",
        "Current Lender `Pool #`": "",
        "Borrower Name(s)/Corporation(s)": "",
        "Borrower Mail: Full Street Address": "",
        "Borrower Mail: Unit #": "",
        "Borrower Mail: City Name": "",
        "Borrower Mail: State": "",
        "Borrower Mail: Zip": "",
        "Borrower Mail: Zip + 4": "",
        "Assessor Parcel Number": "",
        "Multiple APN Code": "",
        "Tax Acct ID": "",
        "Full Street Address": "",
        "Unit #": "",
        "City Name": "",
        "State": "",
        "Zip": "",
        "Zip + 4": "",
        "Data Entry Date": "",
        "Data Entry Operator Code": "",
        "Vendor Source Code": ""
    }
    return get_metadata(object_id, collection, fields)

def process_document(doc):
    try:
        collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "processing"}})
        temp_file_path = os.path.join(tempfile.gettempdir(), secure_filename(os.path.basename(doc["image"])))

        response = httpx.get(doc["image"])
        response.raise_for_status()
        with open(temp_file_path, 'wb') as file:
            file.write(response.content)

        extracted_data = analyze_document_chunked(temp_file_path)
        processed_data = process_ocr_output(extracted_data)

        collection.update_one({"_id": doc["_id"]}, {"$set": {"ocr_output": extracted_data, "json_data": processed_data}})
        
        processed_date = datetime.now(pytz.timezone('UTC')).isoformat()
        object_id = doc["_id"]
        
        final_assignment = get_metadata_for_final_assignment(object_id, collection)
        final_release = get_metadata_for_final_release(object_id, collection)

        max_attempts = 5
        attempt = 0
        
        while final_release is None and attempt < max_attempts:
            final_release = get_metadata_for_final_release(object_id, collection)
            if final_release is None:
                logging.warning(f"Attempt {attempt + 1}: final_release is None, retrying...")
            attempt += 1

        attempt = 0
        while final_assignment is None and attempt < max_attempts:
            final_assignment = get_metadata_for_final_assignment(object_id, collection)
            if final_assignment is None:
                logging.warning(f"Attempt {attempt + 1}: final_assignment is None, retrying...")
            attempt += 1

        if final_assignment is None and final_release is None:
            logging.error(f"Both final_assignment and final_release are None for document {doc['_id']}. Marking as failed.")
            collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed", "error": "Both final_assignment and final_release are None"}})
            return  # exit early if both are None

        update_data = {
            "status": "processed",
            "processed_date": processed_date,
            "final_release": final_release,
            "final_assignment": final_assignment
        }

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_data}
        )
    except Exception as e:
        logging.error(f"Processing failed for document {doc['_id']}: {e}")
        collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed", "error": str(e)}})
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def process_documents():
    documents = collection.find({"status": {"$in": ["processing", "notprocessed"]}}).limit(10)  # Process max 10 docs per run
    for doc in documents:
        try:
            process_document(doc)
            time.sleep(1)  # Add a small delay between processing documents
        except Exception as e:
            logging.error(f"Error processing document {doc['_id']}: {e}")
            collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed", "error": str(e)}})

@app.get("/")
def read_root():
    return {"status": "success"}

@app.post("/process")
def process_route():
    process_documents()
    return JSONResponse(content={"message": "Documents processed successfully"}, status_code=200)

@app.post("/process_again")
async def update_status(request: BatchRequest):
    batchname = request.batchname
    status = request.status
    modified_count = 0
    for name in batchname:
        result = collection.update_many(
            {"batchname": name, "status": status},
            {"$set": {"status": "notprocessed"}}
        )
        modified_count += result.modified_count
    if modified_count > 0:
        return JSONResponse(content={"message": f"{modified_count} images updated successfully."}, status_code=200)
    else:
        return JSONResponse(content={"message": "No images found to update."}, status_code=404)

# Scheduler setup
scheduler.add_job(process_documents, 'interval', seconds=15)  # Run every 15 seconds
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

