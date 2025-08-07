import boto3
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import subprocess

########### CURRENTLY NOT USED (INSTEAD USING s5cmd) ###########

def download_file(s3_client, bucket_name, s3_path, local_path):
    try:
        s3_client.download_file(bucket_name, s3_path, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {str(e)}")
        return False

def download_from_s3(bucket_name, prefix, local_dir, max_workers=10):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    # Define folders to download
    folders = [
        'img',
        'txt',
        'index',
        'img_extracted',
        'embeddings',
        'embeddings_img_pg',
        'embeddings_img_extracted',
        'metadata',
    ]
    
    print("Begining Data Download")
    print(f"Bucket: {bucket_name}, Prefix: {prefix}, Local Directory: {local_dir}, Max Workers: {max_workers}")
    for folder in folders:
        print(f"\nDownloading {folder}...")
        
        s3_folder_path = f"s3://{bucket_name}/{prefix}/{folder}/"
        subprocess.run(f"s5cmd cp {s3_folder_path} {ec2_dir}".split())

        files_to_download = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_path):
            if 'Contents' not in page:
                print(f"No contents found for {s3_folder_path}. Skipping...")
                continue
            for obj in page['Contents']:
                s3_path = obj['Key']
                relative_path = s3_path[len(prefix)+1:]
                local_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                files_to_download.append((s3_path, local_path))

        print("FILES TO DOWNLOAD: ", files_to_download)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s3_path, local_path in files_to_download:
                future = executor.submit(download_file, s3_client, bucket_name, s3_path, local_path)
                futures.append((future, s3_path))
            
            with tqdm(total=len(files_to_download), desc=f"Downloading {folder}") as pbar:
                for future, s3_path in futures:
                    future.result()
                    pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description='Download files from S3 bucket')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', required=True, help='S3 prefix (folder path)')
    parser.add_argument('--local-dir', required=True, help='Local directory to save files')
    parser.add_argument('--max-workers', type=int, default=10, help='Number of parallel downloads')
    
    args = parser.parse_args()
    download_from_s3(args.bucket, args.prefix, args.local_dir, args.max_workers)

if __name__ == '__main__':
    main()
