Remote data is split across two source.coop repositories, both accessed through the
source.coop S3 proxy (`https://data.source.coop`) as bucket `govscape`, with the
repository name as the key prefix (see `govscape/config.py`).

The PDF archive ([govscape/eota-pdf-archive](https://source.coop/govscape/eota-pdf-archive))
holds the raw inputs:

* pdfs/{digest}.pdf
* cdx/complete_cdx.parquet

The derivative data repository ([govscape/eota-derivative-data](https://source.coop/govscape/eota-derivative-data))
holds everything derived from the PDFs:

* {test,dev,prod}-serving/txt/{digest}/{digest}_{pg_no}.txt
* {test,dev,prod}-serving/img/{digest}/{digest}_{pg_no}.jpeg
* {test,dev,prod}-serving/embeddings/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/embeddings_img_pg/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/index/faiss_index.pkl
* {test,dev,prod}-serving/index/forward_index.lmdb/ (LMDB, default) or forward_index.db (SQLite)
* {test,dev,prod}-serving/index_keyword/{whoosh idx files}
* {test,dev,prod}-serving/index_img_pg/faiss_index.pkl
* {test,dev,prod}-serving/index_img_pg/forward_index.lmdb/ (LMDB, default) or forward_index.db (SQLite)
* {test,dev,prod}-serving/index_metadata/metadata.db
* {test,dev,prod}-serving/metadata/{digest}/metadata.json
* {test,dev,prod}-serving/performance/performance_{job_name}.json
* {test,dev,prod}-serving/checkpoints/checkpoint_{job_name/server_id}.json
* {test,dev,prod}-serving/blacklist.txt

The blacklist.txt file is optional. When present, it contains PDF digests (one per line) to hide from all search results and `/pages/<pdf_id>` lookups, used for privacy and copyright takedown requests. Blank lines and lines starting with `#` are ignored, so operational annotations like `# DMCA ticket-1234` are allowed.

The complete_cdx.parquet file has the following columns:

* url : The URL that the PDF was crawled from
* crawl_date : The date that the pdf was crawled as an 8 digit number (YYYYMMDD)
* digest : The hash digest of the pdf as a 32 character string
* filename : The prefix within the eotarchive bucket where the pdf's warc file can be found.
* offset : The pdf's offset into the warc file
* length : The number of bytes corresponding to the pdf's warc record.

Each vector index directory holds a forward index mapping a PDF digest to the exact vectors of its pages. It is maintained by the vector index and used for prefiltered (metadata-filtered) search. Its format is chosen with `--forward_index_type`: LMDB (default) or SQLite. The same type must be used for index building and serving.

The metadata.db database has a table with the columns:
* url TEXT,
* crawl_date TEXT,
* digest TEXT,
* pretty_name TEXT,
* sub_domain TEXT,
* page_count INTEGER

Note: The PDF digest should always be a 32 character string without a "sha1:" prefix.
