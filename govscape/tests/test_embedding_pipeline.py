import shutil
from pathlib import Path

import pytest

from govscape.config import DataModel
from govscape.pdf_processing_pipeline import PDFProcessingPipeline
from govscape.processing import (
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)
from govscape.processing.pdf_extraction_stage import _convert_single_pdf


@pytest.fixture()
def sample_pipeline(tmp_path):
    source_pdfs = Path(__file__).resolve().parent / "test_data" / "small" / "PDFs"
    pdf_dir = tmp_path / "pdfs"
    shutil.copytree(source_pdfs, pdf_dir)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pipeline = PDFProcessingPipeline(str(data_dir), "Dummy", "Dummy")
    pdf_files = sorted(str(f) for f in pdf_dir.glob("*.pdf"))
    return pipeline, pdf_files


def test_convert_pdf_to_txt_img_and_metadata(sample_pipeline):
    pipeline, pdf_files = sample_pipeline
    pdf_file = next(f for f in pdf_files if Path(f).name == "govscape_intro.pdf")
    _convert_single_pdf(pipeline.data_model, pdf_file)

    pdf_stem = "govscape_intro"
    text_path = Path(pipeline.data_model.txt_directory) / pdf_stem / f"{pdf_stem}_0.txt"
    image_path = (
        Path(pipeline.data_model.image_directory) / pdf_stem / f"{pdf_stem}_0.jpeg"
    )

    assert text_path.exists()
    assert image_path.exists()

    text_content = text_path.read_text(encoding="utf-8").strip().lower()
    assert "govscape" in text_content


def test_convert_pdfs_to_txt_and_img_creates_outputs(sample_pipeline):
    pipeline, pdf_files = sample_pipeline
    PDFExtractionStage(
        data_model=pipeline.data_model,
        pdf_files=pdf_files,
        cpu_count=pipeline.cpu_count,
    ).run()

    txt_base = Path(pipeline.data_model.txt_directory)
    img_base = Path(pipeline.data_model.image_directory)

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


def test_pdf_extraction_stage_validate_raises_on_missing_file(tmp_path):
    data_model = DataModel(str(tmp_path))
    stage = PDFExtractionStage(
        data_model=data_model,
        pdf_files=[str(tmp_path / "nonexistent.pdf")],
        cpu_count=1,
    )
    with pytest.raises(ValueError, match="PDF file"):
        stage.validate()


def test_text_embedding_stage_validate_raises_on_missing_dir(tmp_path):
    data_model = DataModel(str(tmp_path / "nonexistent"))
    stage = TextEmbeddingStage(
        data_model=data_model,
        model_type="Dummy",
    )
    with pytest.raises(ValueError, match="Text input directory does not exist"):
        stage.validate()


def test_page_image_embedding_stage_validate_raises_on_missing_dir(tmp_path):
    data_model = DataModel(str(tmp_path / "nonexistent"))
    stage = PageImageEmbeddingStage(
        data_model=data_model,
        model_type="Dummy",
        cpu_count=1,
    )
    with pytest.raises(ValueError, match="Image input directory does not exist"):
        stage.validate()


def test_pdf_extraction_stage_validate_passes_with_existing_files(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()
    data_model = DataModel(str(tmp_path))
    stage = PDFExtractionStage(
        data_model=data_model,
        pdf_files=[str(pdf_file)],
        cpu_count=1,
    )
    stage.validate()  # should not raise


def test_text_embedding_stage_validate_passes_with_existing_dir(tmp_path):
    data_model = DataModel(str(tmp_path))
    (tmp_path / "txt").mkdir()
    stage = TextEmbeddingStage(
        data_model=data_model,
        model_type="Dummy",
    )
    stage.validate()  # should not raise


def test_page_image_embedding_stage_validate_passes_with_existing_dir(tmp_path):
    data_model = DataModel(str(tmp_path))
    (tmp_path / "img").mkdir()
    stage = PageImageEmbeddingStage(
        data_model=data_model,
        model_type="Dummy",
        cpu_count=1,
    )
    stage.validate()  # should not raise


def test_process_pdfs_text_only(sample_pipeline):
    pipeline, pdf_files = sample_pipeline

    # Run the text-only portion of the pipeline; skips heavy image embedding work.
    timings = pipeline.process_pdfs(
        pdf_files,
        do_text_embedding=True,
        do_img_embedding=False,
        do_metadata_collection=False,
        do_ocr=False,
    )

    assert len(timings) == 4
    assert all(isinstance(value, float) for value in timings)

    txt_base = Path(pipeline.data_model.txt_directory)
    img_base = Path(pipeline.data_model.image_directory)

    for pdf_file in pdf_files:
        stem = Path(pdf_file).stem
        assert (txt_base / stem).exists()
        assert (img_base / stem).exists()
