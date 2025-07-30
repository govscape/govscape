from govscape.indexing import SQLiteMetadataIndex

import pandas as pd
from urllib.parse import urlparse
import boto3

def extract_subdomain(url):
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return None
    parts = hostname.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return hostname

def main():    # Initialize the SQLite metadata index
    index = SQLiteMetadataIndex('data/index_metadata')

    # Create the metadata table if it doesn't exist
    index.build_index()

    print("Reading CDX data")
    cdx_data = pd.read_parquet('data/cdx_dir/pdf_metadata.parquet')

    print("Building Index")
    cur_batch = []
    for _, row in cdx_data.iterrows():
        cur_batch.append({
            'url': row['url'],
            'crawl_date': row['crawl_date'],
            'pdf_name': row['digest'],
            'sub_domain': extract_subdomain(row['url']),
            's3_url': f"https://bcgl-public-bucket.s3.amazonaws.com/archive/2020/PDFs/{row['digest']}.pdf"
        })
        if len(cur_batch) >= 1000:
            index.add_batch(cur_batch)
            cur_batch = []
    print("Built Index")
    index.save_index()
    s3 = boto3.client('s3')
    s3.upload_file('data/index_metadata/metadata.db', 'bcgl-public-bucket', 'prod-serving/index_metadata/metadata.db')
    
main()
