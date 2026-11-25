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
* {test,dev,prod}-serving/img/pack_{sha1}.jpegs (see "Page image packs" below; serving directories written before packs have img/{digest}/{digest}_{pg_no}.jpeg instead)
* {test,dev,prod}-serving/embeddings/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/embeddings_img_pg/{digest}/{digest}_{pg_no}.np
* {test,dev,prod}-serving/index/faiss_index.pkl
* {test,dev,prod}-serving/index/forward_index.lmdb/
* {test,dev,prod}-serving/index_keyword/tantivy/ (Tantivy, default), fts_txt.db (SQLite) or {whoosh idx files}
* {test,dev,prod}-serving/index_img_pg/faiss_index.pkl
* {test,dev,prod}-serving/index_img_pg/forward_index.lmdb/
* {test,dev,prod}-serving/index_metadata/metadata.db
* {test,dev,prod}-serving/metadata/{digest}/metadata.json
* {test,dev,prod}-serving/performance/performance_{job_name}.json
* {test,dev,prod}-serving/checkpoints/{job_name}_checkpoint.json
* {test,dev,prod}-serving/blacklist.txt

While the embedding pipeline runs, its local working directory holds two kinds of batch directory, both laid out like a serving directory. Each `download_{n}/` holds a batch of PDFs as they're extracted, with the downloaded PDFs in `PDFs/{digest}.pdf` (deleted once extracted). The extracted PDFs then move into `pages_{n}/` directories, regrouped by page count, which also hold (with OCR) `ocr_img/{digest}/{digest}_{pg_no}.png`, high-resolution renders of the pages that need OCR (deleted once OCR'd), and the per-page JPEGs that are concatenated into the batch's pack. Only the pack is uploaded, not the per-page JPEGs or the scratch directories.

The embedding pipeline splits the PDFs into shards by the leading characters of their digests (two by default, so 1,024 shards) and deals the shards out across servers. Each shard has its own checkpoint, `embedding_pipeline_shard_{prefix}_checkpoint.json`, so a run can resume with a different number of servers.

## Page image packs

Each batch of the embedding pipeline uploads its pages' JPEG images concatenated into one object, `img/pack_{sha1}.jpegs`, named after the SHA-1 of its bytes, so a pack's name always identifies the bytes its offsets point into. A run that resumes from a checkpoint may form different batches and leave earlier, unreferenced packs behind. Within a pack, PDFs are in digest order and each PDF's pages are contiguous, in page order. Each PDF's `metadata.json` records where its pages are:

```json
"page_images": {"pack": "pack_<sha1>.jpegs", "offsets": [0, 51234, 98877]}
```

There is one more offset than pages: page `i` is bytes `offsets[i]` to `offsets[i + 1] - 1` of the pack. The metadata index keeps these in its `page_images` table, the API returns each page as `{"pack", "start", "end"}` (an inclusive byte range), and the frontend fetches it with an HTTP `Range` request. To take a PDF's images down, overwrite its byte range in the pack (e.g. with zeros) and re-upload the pack; its length, and so every other PDF's offsets, stay the same.

The blacklist.txt file is optional. When present, it contains PDF digests (one per line) to hide from all search results and `/pages/<pdf_id>` lookups, used for privacy and copyright takedown requests. Blank lines and lines starting with `#` are ignored, so operational annotations like `# DMCA ticket-1234` are allowed.

The complete_cdx.parquet file has the following columns:

* url : The URL that the PDF was crawled from
* crawl_date : The date that the pdf was crawled as an 8 digit number (YYYYMMDD)
* digest : The hash digest of the pdf as a 32 character string
* filename : The prefix within the eotarchive bucket where the pdf's warc file can be found.
* offset : The pdf's offset into the warc file
* length : The number of bytes corresponding to the pdf's warc record.

Each in-memory (FAISS) vector index directory holds `forward_index.lmdb`, an LMDB store mapping a PDF digest to the exact vectors of its pages, used for prefiltered (metadata-filtered) search. Disk (IVF-PQ) indexes keep their vectors in `vectors.f16` instead, with `digest_rows.lmdb` mapping each PDF to its rows.

The metadata.db database has a table with the columns:
* url TEXT,
* crawl_date TEXT,
* digest TEXT,
* pretty_name TEXT,
* sub_domain TEXT,
* page_count INTEGER

Note: The PDF digest should always be a 32 character string without a "sha1:" prefix.
