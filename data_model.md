The data in the S3 Bucket should be laid out as follows:

* archive-small/PDFs/{digest}.pdf <-- Used for testing on a small scale, holds ~15k PDFs
* archive/{year}/PDFs/{digest}.pdf
* archive/{year}/metadata/pdf_metadata.parquet
* {test,dev,prod}-serving/txt/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/img/{digest}/{digest}_{pg_no}.jpeg
* {test,dev,prod}-serving/embeddings/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/embeddings_img_pg/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/index/faiss_index.pkl
* {test,dev,prod}-serving/index_keyword/{whoosh idx files}
* {test,dev,prod}-serving/index_img_pg/faiss_index.pkl
* {test,dev,prod}-serving/index_metadata/metadata.db
* {test,dev,prod}-serving/metadata/{digest}/metadata.json

The pdf_metadata.parquet file has the following columns:

* url : The URL that the PDF was crawled from
* crawl_date : The date that the pdf was crawled as an 8 digit number (YYYYMMDD)
* digest : The hash digest of the pdf as a 32 character string
* filename : The prefix within the eotarchive bucket where the pdf's warc file can be found.
* offset : The pdf's offset into the warc file 
* length : The number of bytes corresponding to the pdf's warc record.

The metadata.db database has a table with the columns:
* id INTEGER PRIMARY KEY AUTOINCREMENT,
* url TEXT,
* crawl_date TEXT,
* pdf_name TEXT,
* sub_domain TEXT,
* page_count INTEGER <- currently this is not filled in (i.e. entirely nulls)


In a future version, we will transition to:
* archive/PDFs/{digest}.pdf
* archive/metadata/pdf_metadata.parquet
* archive/{year}/metadata/pdf_metadata.parquet
* {test,dev,prod}-serving/txt/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/img/{digest}/{digest}_{pg_no}.jpeg
* {test,dev,prod}-serving/embeddings/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/embeddings_img_pg/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/index_text_vector/faiss_index.pkl
* {test,dev,prod}-serving/index_text_keyword/{whoosh idx files}
* {test,dev,prod}-serving/index_img_pg_vector/faiss_index.pkl

Note: The PDF digest should always be a 32 character string with not "sha1:" prefix.