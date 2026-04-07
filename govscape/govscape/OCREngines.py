import easyocr
import ocrmypdf
from paddleocr import PaddleOCR
from olmocr.pipeline import process_pdf
from base import OCREngine # Assuming the code above is in base.py

class EasyOCREngine(OCREngine):
    def __init__(self):
        super().__init__("EasyOCR")
        self.reader = easyocr.Reader(['en'])

    def process(self, input_path: str) -> str:
        raw_result = self.reader.readtext(input_path, detail=0)
        return "\n\n".join(raw_result)

class PaddleOCREngine(OCREngine):
    def __init__(self):
        super().__init__("PaddleOCR")
        self.engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def process(self, input_path: str) -> str:
        paddle_raw = self.engine.ocr(input_path, cls=True)
        text_only = [line[1][0] for res in paddle_raw for line in res] if paddle_raw[0] else []
        return "\n\n".join(text_only)

class OCRmyPDFEngine(OCREngine):
    def __init__(self):
        super().__init__("OCRmyPDF")

    def process(self, input_path: str) -> str:
        import os
        sidecar_txt = "sidecar_output.txt"
        ocrmypdf.ocr(input_path, "temp.pdf", sidecar=sidecar_txt, skip_text=True, quiet=True)
        with open(sidecar_txt, 'r', encoding='utf-8') as f:
            return f.read()

class OlmOCREngine(OCREngine):
    def __init__(self):
        super().__init__("olmOCR")

    def process(self, input_path: str) -> str:
        olm_results = process_pdf(input_path)
        return "\n".join([page.natural_language for page in olm_results])