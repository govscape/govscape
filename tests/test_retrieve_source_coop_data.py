import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

import duckdb
import pandas as pd
import requests

from scripts.data_prep import retrieve_source_coop_data as source_data


class FakeResponse(requests.Response):
    def iter_content(
        self, chunk_size: int | None = None, decode_unicode: bool = False
    ) -> Iterator[bytes]:
        assert chunk_size == 1024 * 1024
        yield b"pdf-content"

    def close(self) -> None:
        return None


def test_download_pdf_uses_archive_digest_path(tmp_path: Path) -> None:
    digest = "ABC123"
    session = requests.Session()
    response = FakeResponse()
    response.status_code = 200
    with patch.object(session, "get", return_value=response) as get:
        pdf_path = source_data._download_pdf(session, digest, tmp_path)

    get.assert_called_once_with(
        f"{source_data.PDF_BASE_URL}/{digest}.pdf",
        stream=True,
        timeout=(10, 120),
    )
    assert pdf_path == tmp_path / f"{digest}.pdf"
    assert pdf_path.read_bytes() == b"pdf-content"
    assert sorted(path.name for path in tmp_path.iterdir()) == [pdf_path.name]


def test_save_cdx_rows_selects_downloaded_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "digest": ["digest-1", "digest-2", "digest-1", "digest-3"],
            "url": ["one.pdf", "two.pdf", "other.pdf", "three.pdf"],
        }
    ).to_parquet(source_path, index=False)
    monkeypatch.setattr(source_data, "PARQUET_URL", str(source_path))

    output_path = tmp_path / "complete_cdx_sample.parquet"
    with duckdb.connect() as connection:
        source_data._save_cdx_rows(connection, ["digest-1", "digest-2"], output_path)

    with duckdb.connect() as connection:
        result = connection.execute(
            "SELECT digest, url FROM read_parquet(?) ORDER BY digest, url",
            [str(output_path)],
        ).fetchall()

    assert result == [
        ("digest-1", "one.pdf"),
        ("digest-1", "other.pdf"),
        ("digest-2", "two.pdf"),
    ]

    with (
        duckdb.connect() as connection,
        pytest.raises(ValueError, match="No CDX records found for any PDF digest"),
    ):
        source_data._save_cdx_rows(
            connection, ["missing-digest"], tmp_path / "missing.parquet"
        )


def test_main_downloads_requested_distinct_pdfs_and_matching_cdx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "digest": ["digest-3", "digest-2", "digest-1", "digest-1", "digest-4"],
            "url": ["three.pdf", "two.pdf", "one.pdf", "other.pdf", "four.pdf"],
        }
    ).to_parquet(source_path, index=False)
    monkeypatch.setattr(source_data, "PARQUET_URL", str(source_path))
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "digest-3.pdf").write_bytes(b"already present")
    (pdf_dir / "report.pdf").write_bytes(b"not an archive digest")
    cdx_dir = tmp_path / "cdx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrieve_source_coop_data.py",
            "--num_pdfs",
            "2",
            "--pdf_dir",
            str(pdf_dir),
            "--cdx_dir",
            str(cdx_dir),
        ],
    )
    response = FakeResponse()
    response.status_code = 200

    with patch.object(requests.Session, "get", return_value=response) as get:
        source_data.main()

    assert [call.args[0] for call in get.call_args_list] == [
        f"{source_data.PDF_BASE_URL}/digest-2.pdf",
        f"{source_data.PDF_BASE_URL}/digest-1.pdf",
    ]
    assert (pdf_dir / "digest-3.pdf").read_bytes() == b"already present"
    assert sorted(path.name for path in pdf_dir.iterdir()) == [
        "digest-1.pdf",
        "digest-2.pdf",
        "digest-3.pdf",
        "report.pdf",
    ]
    with duckdb.connect() as connection:
        result = connection.execute(
            "SELECT digest, url FROM read_parquet(?) ORDER BY digest, url",
            [str(cdx_dir / source_data.CDX_SAMPLE_FILENAME)],
        ).fetchall()
    assert result == [
        ("digest-1", "one.pdf"),
        ("digest-1", "other.pdf"),
        ("digest-2", "two.pdf"),
        ("digest-3", "three.pdf"),
    ]

    manifest = pd.read_csv(cdx_dir / "digests_manifest.csv")
    assert manifest.to_dict("records") == [
        {"file": "digest-1.pdf", "digest": "digest-1", "url": "one.pdf"},
        {"file": "digest-2.pdf", "digest": "digest-2", "url": "two.pdf"},
        {"file": "digest-3.pdf", "digest": "digest-3", "url": "three.pdf"},
    ]


def test_main_url_filter_limits_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "digest": ["digest-1", "digest-2", "digest-3"],
            "url": [
                "https://www.nasa.gov/one.pdf",
                "https://www.EPA.gov/two.pdf",
                "https://www.epa.gov/three.pdf",
            ],
        }
    ).to_parquet(source_path, index=False)
    monkeypatch.setattr(source_data, "PARQUET_URL", str(source_path))
    pdf_dir = tmp_path / "pdfs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrieve_source_coop_data.py",
            "--num_pdfs",
            "2",
            "--pdf_dir",
            str(pdf_dir),
            "--cdx_dir",
            str(tmp_path / "cdx"),
            "--url_filter",
            "epa.gov",
        ],
    )
    response = FakeResponse()
    response.status_code = 200

    with patch.object(requests.Session, "get", return_value=response):
        source_data.main()

    assert sorted(path.name for path in pdf_dir.iterdir()) == [
        "digest-2.pdf",
        "digest-3.pdf",
    ]
