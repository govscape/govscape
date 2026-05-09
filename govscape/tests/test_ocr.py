# AI modified: 2026-05-09 govscape
"""Tests for OCR implementations and OCRProcessingStage."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from govscape.config import DataModel
from govscape.processing import OCRProcessingStage
from govscape.processing.ocr import (
    BaseOCR,
    EasyOCRImpl,
    OcrMyPDFImpl,
    OLMOcrImpl,
    PaddleOCRImpl,
)


class TestBaseOCR:
    """Test the abstract BaseOCR class."""

    def test_base_ocr_is_abstract(self):
        """Test that BaseOCR cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseOCR()

    def test_base_ocr_methods_are_abstract(self):
        """Test that BaseOCR methods must be implemented by subclasses."""

        class IncompleteOCR(BaseOCR):
            pass

        with pytest.raises(TypeError):
            IncompleteOCR()


class TestEasyOCRImpl:
    """Test EasyOCR implementation."""

    def test_easyocr_init(self):
        """Test EasyOCR initialization."""
        ocr = EasyOCRImpl(languages=["en", "fr"], gpu=False)
        assert ocr.languages == ["en", "fr"]
        assert ocr.gpu is False

    def test_easyocr_default_languages(self):
        """Test EasyOCR defaults to English."""
        ocr = EasyOCRImpl()
        assert ocr.languages == ["en"]

    @patch("govscape.processing.ocr.easyocr_impl.easyocr")
    def test_easyocr_validate(self, mock_easyocr):
        """Test EasyOCR validation."""
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader

        ocr = EasyOCRImpl(languages=["en"])
        ocr.validate()

        assert ocr.reader is not None
        mock_easyocr.Reader.assert_called_once()

    def test_easyocr_validate_import_error(self):
        """Test EasyOCR validation fails without easyocr package."""
        ocr = EasyOCRImpl()
        # This will fail if easyocr is not installed
        # In test environment, we may need to mock this
        with (
            patch(
                "govscape.processing.ocr.easyocr_impl.easyocr", side_effect=ImportError
            ),
            pytest.raises(ImportError),
        ):
            ocr.validate()


class TestPaddleOCRImpl:
    """Test PaddleOCR implementation."""

    def test_paddleocr_init(self):
        """Test PaddleOCR initialization."""
        ocr = PaddleOCRImpl(language="en", use_gpu=False)
        assert ocr.language == "en"
        assert ocr.use_gpu is False

    def test_paddleocr_default_language(self):
        """Test PaddleOCR defaults to English."""
        ocr = PaddleOCRImpl()
        assert ocr.language == "en"

    @patch("govscape.processing.ocr.paddleocr_impl.PaddleOCR")
    def test_paddleocr_validate(self, mock_paddleocr):
        """Test PaddleOCR validation."""
        mock_ocr = MagicMock()
        mock_paddleocr.return_value = mock_ocr

        ocr = PaddleOCRImpl(language="en")
        # We need to mock the import as well
        with patch(
            "govscape.processing.ocr.paddleocr_impl.PaddleOCR", return_value=mock_ocr
        ):
            ocr.validate()
            assert ocr.ocr is not None


class TestOLMOcrImpl:
    """Test OLMOcr implementation."""

    def test_olmocr_init(self):
        """Test OLMOcr initialization."""
        ocr = OLMOcrImpl(model_name="default")
        assert ocr.model_name == "default"

    def test_olmocr_default_model(self):
        """Test OLMOcr defaults to 'default' model."""
        ocr = OLMOcrImpl()
        assert ocr.model_name == "default"


class TestOcrMyPDFImpl:
    """Test OcrMyPDF/Tesseract implementation."""

    def test_ocrmypdf_init(self):
        """Test OcrMyPDF initialization."""
        ocr = OcrMyPDFImpl(language="eng", output_type="txt")
        assert ocr.language == "eng"
        assert ocr.output_type == "txt"

    def test_ocrmypdf_default_language(self):
        """Test OcrMyPDF defaults to English."""
        ocr = OcrMyPDFImpl()
        assert ocr.language == "eng"

    @patch("govscape.processing.ocr.ocrmypdf_impl.pytesseract")
    def test_ocrmypdf_extract_text(self, mock_pytesseract):
        """Test OcrMyPDF text extraction."""
        mock_pytesseract.image_to_string.return_value = "Hello World"

        ocr = OcrMyPDFImpl()
        ocr.validate()

        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        text = ocr.extract_text(test_image)

        assert isinstance(text, str)


class TestOCRProcessingStage:
    """Test OCRProcessingStage integration."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image directory structure
            data_model = DataModel(tmpdir)
            os.makedirs(data_model.image_directory, exist_ok=True)
            yield tmpdir, data_model

    def test_ocr_stage_initialization(self, temp_data_dir):
        """Test OCRProcessingStage initialization."""
        tmpdir, data_model = temp_data_dir

        stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="easyocr",
            languages=["en"],
            gpu=False,
        )

        assert stage.data_model == data_model
        assert isinstance(stage.ocr_engine, EasyOCRImpl)

    def test_ocr_stage_invalid_ocr_type(self, temp_data_dir):
        """Test OCRProcessingStage with invalid OCR type."""
        tmpdir, data_model = temp_data_dir

        with pytest.raises(ValueError, match="Unsupported OCR type"):
            OCRProcessingStage(
                data_model=data_model,
                ocr_type="invalid_ocr",
            )

    def test_ocr_stage_validates_image_directory(self, temp_data_dir):
        """Test OCRProcessingStage validates image directory exists."""
        tmpdir, data_model = temp_data_dir

        # Remove the image directory
        import shutil

        shutil.rmtree(data_model.image_directory)

        stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="easyocr",
        )

        with pytest.raises(ValueError, match="Image input directory does not exist"):
            stage.validate()

    def test_ocr_stage_build_engines(self, temp_data_dir):
        """Test building different OCR engines."""
        tmpdir, data_model = temp_data_dir

        # Test EasyOCR
        stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="easyocr",
            languages=["en"],
        )
        assert isinstance(stage.ocr_engine, EasyOCRImpl)

        # Test PaddleOCR
        stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="paddleocr",
            language="en",
        )
        assert isinstance(stage.ocr_engine, PaddleOCRImpl)

        # Test OLMOcr
        stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="olmocr",
            model_name="default",
        )
        assert isinstance(stage.ocr_engine, OLMOcrImpl)

        # Test OcrMyPDF
        stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="ocrmypdf",
            language="eng",
        )
        assert isinstance(stage.ocr_engine, OcrMyPDFImpl)

    def test_ocr_stage_creates_txt_directory(self, temp_data_dir):
        """Test that OCRProcessingStage creates txt directory."""
        tmpdir, data_model = temp_data_dir

        stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="easyocr",
        )

        # Mock the OCR engine to avoid actual OCR
        with (
            patch.object(stage.ocr_engine, "validate"),
            patch.object(stage.ocr_engine, "extract_text", return_value="test"),
        ):
            # Create a test image
            digest = "abc123def456abc123def456abc123def45"
            img_dir = os.path.join(data_model.image_directory, digest)
            os.makedirs(img_dir, exist_ok=True)

            # Create a dummy image file
            import cv2

            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_dir, f"{digest}_0.jpeg"), dummy_image)

            # Run stage
            stage.run()

            # Check that txt directory was created
            assert os.path.exists(data_model.txt_directory)

            # Check that txt file was created
            txt_file = data_model.txt_page_path(digest, 0)
            assert os.path.exists(txt_file)

            # Check content
            with open(txt_file) as f:
                content = f.read()
                assert content == "test"
