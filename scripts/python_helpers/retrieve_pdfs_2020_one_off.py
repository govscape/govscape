import os
from warcio.archiveiterator import ArchiveIterator
import subprocess
import pandas as pd
import requests
import shutil
import multiprocessing
import boto3
import time
import io 

def main():
    df = pd.read_parquet('data/cdx_dir/pdf_metadata.parquet')
    df2 = pd.read_parquet('data/cdx_dir/pdf_warc_files.parquet', columns=['filename'])
    df2['crawl_file'] = df2['filename'].astype(str).str.split('/').str[5]
    df = df.merge(df2, left_on='warc_filename', right_on='crawl_file')
    local_dir_base = 'data/archive/2020/PDFs'
    output_bucket_name = 'bcgl-public-bucket'
    output_directory_base = 'archive/2020/PDFs'
    num_processes = multiprocessing.cpu_count() * 5
    batch_size = len(df) // num_processes + 1

    batches = [(df[i:i + batch_size], idx, output_bucket_name, output_directory_base, num_processes) for idx, i in enumerate(range(0, len(df), batch_size))]
    with multiprocessing.get_context('fork').Pool(processes=num_processes) as pool:
        pool.starmap(retrieve_and_store_pdfs, batches)

def retrieve_and_store_pdfs(file_batch, idx, output_bucket_name, output_directory, num_processes):
    s3 = boto3.client('s3')
    processed_pdfs = 0
    start_time = time.time()
    for i in range(1, len(file_batch)):
        filename = file_batch.iloc[i]['filename']
        url = file_batch.iloc[i]['url']
        digest = file_batch.iloc[i]['digest']
        length = int(file_batch.iloc[i]['length'])
        offset = int(file_batch.iloc[i]['offset'])
        s3_url = f'https://eotarchive.s3.amazonaws.com/{filename}'
        myagent = 'govscape/0.1 (PDF Retrieval Script; kdeeds@cs.washington.edu)'
        byte_range = f'bytes={offset}-{offset + length - 1}'
        object_exists = False

        if (idx == 0) and (processed_pdfs % 100 == num_processes):
            print(f'Processed {processed_pdfs} PDFs in {time.time() - start_time:.4f} seconds')
            pdf_per_second = processed_pdfs / (time.time() - start_time)
            print(f'Time Remaining: {(len(file_batch)- i) * num_processes / pdf_per_second :.4f} seconds')
            print(f'Time per PDF: {1 / pdf_per_second:.4f} seconds')
        
        try:
            s3.head_object(Bucket=output_bucket_name, Key=os.path.join(output_directory, digest + '.pdf'))
            object_exists = True
        except Exception as e:
            pass  # Object does not exist, continue to download
        if object_exists:
            processed_pdfs += num_processes
            continue
        
        # Send the HTTP GET request to the S3 URL with the specified byte range
        try:
            response = requests.get(
                s3_url,
                headers={'user-agent': myagent, 'Range': byte_range},
                stream=True
            )
            for record in ArchiveIterator(response.raw):
                if record.rec_type == 'response':
                    is_valid_pdf = (record.rec_headers.get('Content-Type') == 'application/pdf') or (".pdf" in record.rec_headers.get('WARC-Target-URI', ''))
                    if not is_valid_pdf:
                        continue
                    digest = record.rec_headers.get('WARC-Payload-Digest').split(':')[1]
                    s3.put_object(
                        Bucket=output_bucket_name,
                        Key=os.path.join(output_directory, digest + '.pdf'),
                        Body=record.content_stream().read()
                    )
                    processed_pdfs += num_processes
                        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    main()

