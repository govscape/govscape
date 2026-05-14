import logging
import os
import tempfile

import duckdb
from govscape.data_loader import build_data_loader
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

_HOSTNAME_EXPR = "regexp_extract(url, '^https?://([^/?#]+)', 1)"


def _run_stats(parquet_path: str) -> str:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW cdx AS SELECT *, {_HOSTNAME_EXPR} AS hostname FROM '{parquet_path}'")

    total_entries = con.execute("SELECT count(*) FROM cdx").fetchone()[0]
    distinct_digests = con.execute("SELECT count(DISTINCT digest) FROM cdx").fetchone()[0]
    distinct_urls = con.execute("SELECT count(DISTINCT url) FROM cdx").fetchone()[0]
    pdf_url_entries = con.execute(
        "SELECT count(*) FROM cdx WHERE lower(url) LIKE '%.pdf'"
    ).fetchone()[0]

    per_year = con.execute(
        "SELECT substr(crawl_date, 1, 4) AS year,"
        "  count(*) AS entries,"
        "  count(DISTINCT digest) AS distinct_digests,"
        "  count(DISTINCT url) AS distinct_urls"
        " FROM cdx"
        " GROUP BY year"
        " ORDER BY year"
    ).fetchall()

    per_subdomain = con.execute(
        "SELECT hostname,"
        "  count(*) AS entries,"
        "  count(DISTINCT digest) AS distinct_digests"
        " FROM cdx"
        " WHERE hostname <> ''"
        " GROUP BY hostname"
        " ORDER BY entries DESC"
    ).fetchall()

    con.close()

    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("CDX Descriptive Statistics")
    lines.append("=" * 60)
    lines.append("")

    lines.append("--- Overall ---")
    lines.append(f"Total entries:              {total_entries:>12,}")
    lines.append(f"Distinct PDFs (by digest):  {distinct_digests:>12,}")
    lines.append(f"Distinct URLs:              {distinct_urls:>12,}")
    lines.append(f"Entries with .pdf in URL:   {pdf_url_entries:>12,}")
    lines.append("")

    lines.append("--- Per Year ---")
    lines.append(f"{'Year':<6}  {'Entries':>12}  {'Distinct Digests':>16}  {'Distinct URLs':>13}")
    lines.append("-" * 54)
    for year, entries, digests, urls in per_year:
        lines.append(f"{year:<6}  {entries:>12,}  {digests:>16,}  {urls:>13,}")
    lines.append("")

    lines.append("--- Per Subdomain (top 100 by entry count) ---")
    lines.append(f"{'Hostname':<50}  {'Entries':>10}  {'Distinct Digests':>16}")
    lines.append("-" * 80)
    for hostname, entries, digests in per_subdomain[:100]:
        lines.append(f"{hostname:<50}  {entries:>10,}  {digests:>16,}")
    lines.append("")
    lines.append(f"(Total distinct hostnames: {len(per_subdomain):,})")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = base_argument_parser(description="Compute descriptive statistics for complete_cdx.parquet.")
    parser.add_argument("--input_prefix", required=True, help="Remote key prefix where complete_cdx.parquet lives")
    parser.add_argument("--output_prefix", required=True, help="Remote key prefix for the output stats txt file")
    args = parser.parse_args()

    data_loader = build_data_loader(args.backend, args.bucket_name, args.local_base_dir)
    remote_parquet = os.path.join(args.input_prefix, "complete_cdx.parquet")

    with tempfile.TemporaryDirectory(prefix="cdx_stats_") as work_dir:
        local_parquet = os.path.join(work_dir, "complete_cdx.parquet")
        logging.info("Downloading %s", remote_parquet)
        data_loader.download_file(remote_parquet, local_parquet)

        logging.info("Computing statistics")
        report = _run_stats(local_parquet)

        local_txt = os.path.join(work_dir, "cdx_statistics.txt")
        with open(local_txt, "w") as f:
            f.write(report)

        remote_txt = os.path.join(args.output_prefix, "cdx_statistics.txt")
        data_loader.upload_file(local_txt, remote_txt)
        logging.info("Uploaded statistics to %s", remote_txt)

    print(report)


main()
