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
    df = pd.read_parquet('data/cdx_dir/pdf_metadata_mime_only.parquet')
    pdf_warc_files = df['filename'].dropna().unique()

    pd.DataFrame({'filename': pdf_warc_files}).to_parquet('data/cdx_dir/pdf_warc_files.parquet')

    local_dir_base = 'data/archive/2020/PDFs'
    output_bucket_name = 'bcgl-public-bucket'
    output_directory_base = 'archive/2020/PDFs'
    num_processes = multiprocessing.cpu_count()
    batch_size = len(pdf_warc_files) // num_processes + 1

    batches = [(pdf_warc_files[i:i + batch_size], f"{local_dir_base}_batch_{i}", output_bucket_name, output_directory_base) for idx, i in enumerate(range(0, len(pdf_warc_files), batch_size))]
    with multiprocessing.get_context('spawn').Pool(processes=num_processes) as pool:
        pool.starmap(retrieve_and_store_pdfs, batches)

def retrieve_and_store_pdfs(pdf_warc_files, local_dir, output_bucket_name, output_directory):
    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client('s3')
    processed_pdfs = 0
    start_time = time.time()
    for filename in pdf_warc_files:
        s3_url = f'https://eotarchive.s3.amazonaws.com/{filename}'
        myagent = 'govscape/0.1 (PDF Retrieval Script; kdeeds@cs.washington.edu)'

        try:
        # Send the HTTP GET request to the S3 URL with the specified byte range
            response = requests.get(
                s3_url,
                headers={'user-agent': myagent}
            )
            print('Downloaded WARC file:', filename)
            for record in ArchiveIterator(io.BytesIO(response.content)):
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
                    processed_pdfs += 1
                    if processed_pdfs == 1000:
                        print(f'Processed {processed_pdfs} PDFs in {time.time() - start_time:.2f} seconds')
                        processed_pdfs = 0
                        start_time = time.time()  # Reset timer after logging
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    main()