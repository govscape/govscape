import shutil
from pathlib import Path

import pytest

from govscape.pdf_processing_pipeline import PDFProcessingPipeline
from govscape.processing import PDFExtractionStage
from govscape.processing.pdf_extraction_stage import _convert_single_pdf


@pytest.fixture()
def sample_pipeline(tmp_path):
    source_pdfs = Path(__file__).resolve().parent / "test_data" / "small" / "PDFs"
    pdf_dir = tmp_path / "pdfs"
    shutil.copytree(source_pdfs, pdf_dir)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pipeline = PDFProcessingPipeline(str(pdf_dir), str(data_dir), "Dummy", "Dummy")
    pdf_files = sorted(f.name for f in pdf_dir.glob("*.pdf"))
    return pipeline, pdf_files


def test_convert_pdf_to_txt_img_and_metadata(sample_pipeline):
    pipeline, pdf_files = sample_pipeline
    assert "govscape_intro.pdf" in pdf_files
    _convert_single_pdf(
        pipeline.txts_path,
        pipeline.img_path,
        pipeline.pdfs_path,
        pipeline.metadata_dir,
        "govscape_intro.pdf",
    )

    pdf_stem = "govscape_intro"
    text_path = Path(pipeline.txts_path) / pdf_stem / f"{pdf_stem}_0.txt"
    image_path = Path(pipeline.img_path) / pdf_stem / f"{pdf_stem}_0.jpeg"

    assert text_path.exists()
    assert image_path.exists()

    text_content = text_path.read_text(encoding="utf-8").strip().lower()
    assert "govscape" in text_content


def test_convert_pdfs_to_txt_and_img_creates_outputs(sample_pipeline):
    pipeline, pdf_files = sample_pipeline
    PDFExtractionStage(
        pdfs_path=pipeline.pdfs_path,
        txts_path=pipeline.txts_path,
        img_path=pipeline.img_path,
        metadata_dir=pipeline.metadata_dir,
        pdf_files=pdf_files,
        cpu_count=pipeline.cpu_count,
    ).run()

    txt_base = Path(pipeline.txts_path)
    img_base = Path(pipeline.img_path)

    expected_dirs = {Path(pdf).stem for pdf in pdf_files}
    txt_dirs = {p.name for p in txt_base.iterdir() if p.is_dir()}
    img_dirs = {p.name for p in img_base.iterdir() if p.is_dir()}

    assert txt_dirs == expected_dirs
    assert img_dirs == expected_dirs

    total_txt_files = 0
    total_img_files = 0

    for stem in expected_dirs:
        txt_files = list((txt_base / stem).glob("*.txt"))
        img_files = list((img_base / stem).glob("*.jpeg"))
        total_txt_files += len(txt_files)
        total_img_files += len(img_files)

    assert total_txt_files > 0
    assert total_img_files > 0


def test_process_pdfs_text_only(sample_pipeline):
    pipeline, pdf_files = sample_pipeline

    # Run the text-only portion of the pipeline; skips heavy image embedding work.
    timings = pipeline.process_pdfs(
        pdf_files,
        do_text_embedding=True,
        do_img_embedding=False,
        do_metadata_collection=False,
    )

    assert len(timings) == 3
    assert all(isinstance(value, float) for value in timings)

    txt_base = Path(pipeline.txts_path)
    img_base = Path(pipeline.img_path)

    for pdf_file in pdf_files:
        stem = Path(pdf_file).stem
        assert (txt_base / stem).exists()
        assert (img_base / stem).exists()
