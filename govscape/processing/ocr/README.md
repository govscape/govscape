# OCR Processing Module

This module provides OCR (Optical Character Recognition) functionality to extract text from PDF page images. It supports multiple OCR engines and integrates seamlessly with the processing pipeline.

## Architecture


### Core Components

#### BaseOCR (Abstract Class)
All OCR implementations inherit from `BaseOCR` and must implement:
- `extract_text(images: list[np.ndarray]) -> list[str]`: Extract text from a
  batch of page images, returning one string per image in order
- `validate() -> None`: Validate installation and initialize the engine

#### OCR Implementations

1. **EasyOCR** (`easyocr_impl.py`)
   - Supports 80+ languages
   - GPU acceleration available
   - Default language: English
   - Good balance of speed and accuracy

2. **PaddleOCR** (`paddleocr_impl.py`)
   - Fast and accurate multilingual OCR
   - Supports document structure analysis
   - Efficient on both CPU and GPU
   - Default language: English

3. **OLMOcr** (`olmocr_impl.py`)
   - Open-source language model-based OCR
   - Customizable model selection
   - Good for specialized documents

4. **OcrMyPDF** (`ocrmypdf_impl.py`)
   - Built on Tesseract OCR engine
   - Integrates with ocrmypdf library
   - Simpler setup, widely supported languages

#### OCRProcessingStage
Integrates OCR into the processing pipeline:
- Reads images from `{image_directory}/{digest}/{digest}_{pg_no}.jpeg`
- Applies OCR extraction
- Saves text to `{txt_directory}/{digest}/{digest}_{pg_no}.txt`
- Follows the protocol defined in `DATA_MODEL.md`

## Dependencies

OCR engine backends are declared as optional extras in `pyproject.toml`, so you
only install what you use. Each implementation guards its import and raises a
clear error if its backend is missing.

```bash
poetry install --extras ocr        # all engines
poetry install --extras easyocr    # a single engine
```

Available extras: `easyocr`, `paddleocr`, `olmocr`, `ocrmypdf`, and `ocr` (all).

The `ocrmypdf` backend additionally requires the Tesseract system binary, which
is not a Python package:

```bash
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS:         brew install tesseract
```

## Configuration

### OCRProcessingStage Parameters

- `data_model` (DataModel): Required. Defines the directory structure
- `ocr_type` (str): Type of OCR engine ('easyocr', 'paddleocr', 'olmocr', 'ocrmypdf')
- `**ocr_kwargs`: Engine-specific parameters

### EasyOCR Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="easyocr",
    languages=["en", "fr"],  # Support multiple languages
    gpu=True,  # Use GPU if available
)
```

### PaddleOCR Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="paddleocr",
    language="en",  # Single language
    use_gpu=True,  # Use GPU if available
)
```

### OLMOcr Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="olmocr",
    model_name="default",  # Model variant
)
```

### OcrMyPDF Options
```python
OCRProcessingStage(
    data_model=data_model,
    ocr_type="ocrmypdf",
    language="eng",  # Tesseract language code
    output_type="txt",  # Output format
)
```

## Output Format

Text extraction results are saved following the `DATA_MODEL.md` protocol:
```
{txt_directory}/{digest}/{digest}_{pg_no}.txt
```

Example:
```
txt/
├── abc123def456abc123def456abc123def45/
│   ├── abc123def456abc123def456abc123def45_0.txt
│   ├── abc123def456abc123def456abc123def45_1.txt
│   └── abc123def456abc123def456abc123def45_2.txt
└── xyz789abc123xyz789abc123xyz789abc123/
    ├── xyz789abc123xyz789abc123xyz789abc123_0.txt
    └── xyz789abc123xyz789abc123xyz789abc123_1.txt
```

## Extending the Module

To add a new OCR engine:

1. Create a new file in `ocr/` directory
2. Implement the `BaseOCR` abstract class
3. Add the import and mapping in `ocr/__init__.py`
4. Add the mapping in `_build_ocr_engine()` function in `ocr_processing_stage.py`

Example:
```python
# ocr/tesseract_impl.py
from .base_ocr import BaseOCR


class TesseractImpl(BaseOCR):
    def __init__(self, language: str = "eng"):
        self.language = language
        self.ocr = None

    def validate(self) -> None:
        # Initialize Tesseract
        pass

    def extract_text(self, images: list[np.ndarray]) -> list[str]:
        # Extract text from each image, returning one string per image
        pass
```

## Testing

Test the OCR implementations:
```bash
# Run OCR tests
poetry run pytest tests/test_ocr.py -v

# Run specific OCR implementation test
poetry run pytest tests/test_ocr.py::test_easyocr -v
```

## References

- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [OcrMyPDF Documentation](https://ocrmypdf.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
