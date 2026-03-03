from abc import ABC, abstractmethod
import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from docling.document_converter import DocumentConverter
from pathlib import Path
import tempfile
import json

class MetadataExtractor(ABC):
    @abstractmethod
    def extract(self, pdf_bytes: bytes) -> str | None:
        pass

class PDFMetadataExtractor(MetadataExtractor):
    def extract(self, pdf_bytes: bytes) -> str | None:
        # pdf = pypdfium2.PdfDocument(pdf_path)
        # creation_date = pdf.get_metadata_value("CreationDate")
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        creation_date = reader.metadata.get("/CreationDate")
        return self.parse_date(creation_date)

class DoclingQwenMetadataExtractor(MetadataExtractor):
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            load_in_8bit=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.converter = DocumentConverter()
        # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
    def extract(self, pdf_bytes: bytes) -> str | None:
        text = parse_pdf(pdf_bytes)
        text = text[:2000] + "\n...\n" + text[-2000:]
        return run_qwen(text)

    def parse_pdf(self, pdf_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file_path = Path(tmp_file.name)
        result = self.converter.convert(tmp_file_path)
        text = result.document.export_to_text()
        return text

    def run_qwen(self, text):
        prompt = f"""
        You are an information extraction system.

        Extract the document creation date from the text below.

        Rules:
        - Return ONLY valid JSON: {{"creation_date": "MM/DD/YYYY"}}
        - No explanations.
        - If no date is found, return {{"creation_date": null}}.

        Text:
        {text}
        """
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=30)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"creation_date": None}
        del model_inputs, generated_ids, text
        torch.cuda.empty_cache()
