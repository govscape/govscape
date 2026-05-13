import logging
import os
import tempfile
from multiprocessing import Pool, cpu_count

import duckdb
from govscape.data_loader import build_data_loader
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

YEARS = [2008, 2012, 2016, 2020, 2024]
CDX_GLOB_TEMPLATE = (
    "s3://eotarchive/eot-index/collections/EOT-{year}/indexes/*.gz"
)

# Regular string (not f-string) so that { } characters in the SQL are literal.
# chr(123) == '{'.  sep=chr(1) == SOH, guaranteed absent from CDX data, so
# each line becomes a single column (column0).  ? is the glob path parameter.
_INSERT_SQL = (
    "INSERT INTO pdf_entries\n"
    "SELECT\n"
    "  j.url,\n"
    "  j.filename,\n"
    "  regexp_extract(j.filename, '[0-9]{8}') AS crawl_date,\n"
    "  replace(j.digest, 'sha1:', '') AS digest,\n"
    '  j.joffset AS "offset",\n'
    "  j.length\n"
    "FROM (\n"
    "  SELECT\n"
    "    json_extract_string(jstr, '$.url')      AS url,\n"
    "    json_extract_string(jstr, '$.mime')     AS mime,\n"
    "    json_extract_string(jstr, '$.filename') AS filename,\n"
    "    json_extract_string(jstr, '$.digest')   AS digest,\n"
    "    json_extract_string(jstr, '$.status')   AS status,\n"
    "    TRY_CAST(json_extract_string(jstr, '$.offset') AS BIGINT) AS joffset,\n"
    "    TRY_CAST(json_extract_string(jstr, '$.length') AS BIGINT) AS length\n"
    "  FROM (\n"
    "    SELECT substring(column0, strpos(column0, chr(123))) AS jstr\n"
    "    FROM read_csv(?, header=false, sep=chr(1), compression='gzip', max_line_size=10000000)\n"
    "    WHERE strpos(column0, chr(123)) > 0\n"
    "  ) lines\n"
    "  WHERE json_valid(jstr)\n"
    ") j\n"
    "WHERE (j.mime = 'application/pdf' OR j.url LIKE '%.pdf%')\n"
    "  AND j.status = '200'"
)


def _setup_s3(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("LOAD httpfs")
    con.execute("SET s3_region='us-east-1'")
    con.execute("SET s3_access_key_id=''")
    con.execute("SET s3_secret_access_key=''")


def _create_pdf_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "CREATE TABLE pdf_entries ("
        "  url VARCHAR, filename VARCHAR, crawl_date VARCHAR,"
        '  digest VARCHAR, "offset" BIGINT, length BIGINT'
        ")"
    )


def _process_one_file(args: tuple[str, str]) -> tuple[str, int]:
    """Worker: process a single CDX file and write its PDF entries to a partial parquet."""
    cdx_path, output_parquet = args
    con = duckdb.connect()
    _setup_s3(con)
    _create_pdf_table(con)
    con.execute(_INSERT_SQL, [cdx_path])
    count = con.execute("SELECT count(*) FROM pdf_entries").fetchone()[0]
    con.execute(f"COPY pdf_entries TO '{output_parquet}' (FORMAT PARQUET)")
    con.close()
    return cdx_path, count


def main() -> None:
    parser = base_argument_parser(description="Process CDX files from S3 via DuckDB.")
    parser.add_argument("--output_prefix", required=True, help="S3 key prefix for upload")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=YEARS,
        help="EOT years to process (default: all five)",
    )
    parser.add_argument(
        "--max_cdx_files",
        type=int,
        default=None,
        help="Limit CDX files processed per year (useful for testing)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel worker processes (default: cpu_count())",
    )
    args = parser.parse_args()

    # Enumerate CDX file paths for the requested years.
    list_con = duckdb.connect()
    _setup_s3(list_con)
    all_paths: list[str] = []
    for year in args.years:
        glob_path = CDX_GLOB_TEMPLATE.format(year=year)
        if args.max_cdx_files is not None:
            rows = list_con.execute(
                "SELECT file FROM glob(?) LIMIT ?",
                [glob_path, args.max_cdx_files],
            ).fetchall()
        else:
            rows = list_con.execute("SELECT file FROM glob(?)", [glob_path]).fetchall()
        all_paths.extend(p[0] for p in rows)
        logging.info("Year %d: queued %d CDX files", year, len(rows))
    list_con.close()

    if not all_paths:
        logging.warning("No CDX files matched; nothing to do.")
        return

    with tempfile.TemporaryDirectory(prefix="cdx_") as work_dir:
        partials_dir = os.path.join(work_dir, "partials")
        os.makedirs(partials_dir)
        parquet_path = os.path.join(work_dir, "complete_cdx.parquet")

        worker_args = [
            (path, os.path.join(partials_dir, f"part_{i:05d}.parquet"))
            for i, path in enumerate(all_paths)
        ]
        num_workers = min(args.num_workers, len(worker_args))
        logging.info(
            "Processing %d CDX files across %d worker processes",
            len(worker_args),
            num_workers,
        )

        total = 0
        with Pool(processes=num_workers) as pool:
            for cdx_path, count in pool.imap_unordered(_process_one_file, worker_args):
                total += count
                logging.info(
                    "Finished %s: %d rows (cumulative: %d)", cdx_path, count, total
                )

        merge_con = duckdb.connect()
        merge_con.execute(
            f"COPY (SELECT * FROM '{partials_dir}/*.parquet') "
            f"TO '{parquet_path}' (FORMAT PARQUET)"
        )
        merge_count = merge_con.execute(
            f"SELECT count(*) FROM '{parquet_path}'"
        ).fetchone()[0]
        merge_con.close()
        logging.info("Merged %d rows into %s", merge_count, parquet_path)

        data_loader = build_data_loader(args.backend, args.bucket_name, args.local_base_dir)
        remote_key = os.path.join(args.output_prefix, "complete_cdx.parquet")
        data_loader.upload_file(parquet_path, remote_key)
        logging.info("Uploaded to s3://%s/%s", args.bucket_name, remote_key)


main()
