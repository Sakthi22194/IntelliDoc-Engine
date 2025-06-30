     import json
     import boto3

     # Initialize Textract client
     textract = boto3.client('textract')

     def lambda_handler(event, context):
         # Extract S3 bucket and file name from the event
         bucket = event['Records'][0]['s3']['bucket']['name']
         file_name = event['Records'][0]['s3']['object']['key']
         
         # Log the event
         print(f"Processing file {file_name} from bucket {bucket}")
         
         # Call Textract to extract text
         try:
             response = textract.detect_document_text(
                 Document={'S3Object': {'Bucket': bucket, 'Name': file_name}}
             )
             # Extract text from Textract response
             text = ""
             for item in response['Blocks']:
                 if item['BlockType'] == 'LINE':
                     text += item['Text'] + "\n"
             print(f"Extracted text: {text[:100]}...")  # Log first 100 chars
             
             return {
                 'statusCode': 200,
                 'body': json.dumps(f"Extracted text from {file_name}")
             }
         except Exception as e:
             print(f"Error processing {file_name}: {str(e)}")
             return {
                 'statusCode': 500,
                 'body': json.dumps(f"Error processing {file_name}: {str(e)}")
             }