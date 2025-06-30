import boto3
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime

# AWS clients (specify region if needed)
s3_client = boto3.client('s3', region_name='us-east-1')  # Replace with your bucket's region
textract_client = boto3.client('textract', region_name='us-east-1')  # Replace with your region

# Initialize Chroma (local database)
chroma_client = chromadb.PersistentClient(path="C:/IntelliDoc/chroma_db")
collection = chroma_client.get_or_create_collection(name="intellidoc_embeddings")

# Initialize Hugging Face model
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_document(bucket, file_name):
    try:
        # First check if the file exists in S3
        s3_client.head_object(Bucket=bucket, Key=file_name)
        print(f"File {file_name} found in bucket {bucket}")
        
        # Check file type - Textract supports PDF, PNG, JPEG, TIFF
        supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
        file_extension = '.' + file_name.split('.')[-1].lower()
        
        if file_extension not in supported_extensions:
            print(f"Skipping {file_name} - unsupported file type {file_extension}")
            return
        
        # Extract text using Textract
        response = textract_client.detect_document_text(
            Document={'S3Object': {'Bucket': bucket, 'Name': file_name}}
        )
        
        text = ""
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text += item['Text'] + "\n"
        
        if not text.strip():
            print(f"No text extracted from {file_name}")
            return
            
        print(f"Extracted text from {file_name}: {text[:100]}...")
        
        # Split text into sentences for embeddings
        sentences = text.split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            print(f"No valid sentences found in {file_name}")
            return
        
        # Generate embeddings
        embeddings = model.encode(sentences, convert_to_numpy=True).tolist()
        
        # Store embeddings in Chroma
        collection.add(
            documents=sentences,
            embeddings=embeddings,
            ids=[f"{file_name}_{i}" for i in range(len(sentences))],
            metadatas=[{"file_name": file_name, "timestamp": datetime.utcnow().isoformat()} for _ in sentences]
        )
        
        print(f"Stored {len(sentences)} embeddings for {file_name} in Chroma")
        
    except s3_client.exceptions.NoSuchKey:
        print(f"Error: File {file_name} not found in bucket {bucket}")
        return
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return

def process_all_files_in_bucket(bucket, file_extension=None):
    """Process all files in an S3 bucket, optionally filtering by extension"""
    try:
        # List all objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket)
        
        if 'Contents' not in response:
            print(f"No files found in bucket {bucket}")
            return
        
        files_processed = 0
        for obj in response['Contents']:
            file_name = obj['Key']
            
            # Skip if file extension filter is specified and doesn't match
            if file_extension and not file_name.lower().endswith(file_extension.lower()):
                continue
                
            print(f"\n--- Processing file: {file_name} ---")
            process_document(bucket, file_name)
            files_processed += 1
        
        print(f"\nTotal files processed: {files_processed}")
        
    except Exception as e:
        print(f"Error listing files in bucket: {str(e)}")

def process_specific_files(bucket, file_list):
    """Process a specific list of files"""
    for file_name in file_list:
        print(f"\n--- Processing file: {file_name} ---")
        process_document(bucket, file_name)

if __name__ == "__main__":
    bucket = "intellidoc-engine-bucket01"
    
    # Option 1: Process all PDF files in the bucket
    print("Processing all PDF files in bucket...")
    process_all_files_in_bucket(bucket, ".pdf")
    
    # Option 2: Process specific files (uncomment to use)
    # specific_files = ["test.pdf", "document1.pdf", "report.pdf"]
    # print("Processing specific files...")
    # process_specific_files(bucket, specific_files)
    
    # Option 3: Process all files regardless of type (uncomment to use)
    # print("Processing all files in bucket...")
    # process_all_files_in_bucket(bucket)
