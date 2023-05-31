from joblib import dump, load
import boto3
import os

def save_model(model, filename):
    dump(model, filename)

def load_model(filename):
    return load(filename)

def save_tokenizer(tokenizer, filename):
    dump(tokenizer, filename)

def load_tokenizer(filename):
    return load(filename)

def save_embeddings(embeddings, filename):
    dump(embeddings, filename)

def load_embeddings(filename):
    return load(filename)

def save_data(data, filename):
    dump(data, filename)

def load_data(filename):
    return load(filename)

def upload_to_s3(bucket_name, local_dir, s3_prefix):
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_region = os.environ['AWS_REGION']

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    s3 = session.resource("s3")
    bucket = s3.Bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(s3_prefix, os.path.relpath(local_path, local_dir))

            # Upload file to S3
            bucket.upload_file(local_path, s3_path)
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")

def download_from_s3(bucket_name, s3_prefix, local_dir):
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_region = os.environ['AWS_REGION']

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )
    
    s3 = session.resource("s3")
    bucket = s3.Bucket(bucket_name)

    for object_summary in bucket.objects.filter(Prefix=s3_prefix):
        s3_path = object_summary.key
        local_path = os.path.join(local_dir, os.path.relpath(s3_path, s3_prefix))

        # Create local directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file from S3
        bucket.download_file(s3_path, local_path)
        print(f"Downloaded s3://{bucket_name}/{s3_path} to {local_path}")
