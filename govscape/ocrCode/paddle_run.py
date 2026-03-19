import os
import logging

# 1. FORCE THE LEGACY ENGINE GLOBALLY
# This specific key is what PaddleOCR 3.4.0 checks to determine its architecture.
# Setting it to '0' or 'False' disables the broken PaddleX pipeline.
os.environ['USE_PADDLEX'] = '0' 

# 2. Other stability flags
os.environ['FLAGS_use_onednn'] = '0'
os.environ['GLOG_minloglevel'] = '3'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import paddle
from paddleocr import PaddleOCR

# 3. Set device to CPU globally
paddle.set_device('cpu')

# Silence python-level logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Path setup
input_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\testFiles"
output_dir = r"C:\Users\anjal\OneDrive\UWSeattle\2026WinterQ\494\OCRComparisons\MarkDownsPaddleOcr"
os.makedirs(output_dir, exist_ok=True)

# 4. INITIALIZATION WITH NO ARGUMENTS
# The library will now see the 'USE_PADDLEX' environment variable 
# and switch to the stable engine automatically.
ocr = PaddleOCR()

for file in os.listdir(input_dir):
    if file.lower().endswith(".pdf"):
        print(f"Working on: {file}")
        try:
            img_path = os.path.join(input_dir, file)
            result = ocr.ocr(img_path)
            
            if not result or result[0] is None:
                continue
                
            md_path = os.path.join(output_dir, file.replace(".pdf", ".md"))
            with open(md_path, "w", encoding="utf-8") as f:
                for page in result:
                    if page is None: continue
                    for line in page:
                        # line[1][0] is the recognized text
                        f.write(f"{line[1][0]} ")
                    f.write("\n\n")
            print(f"Done: {file}")
            
        except Exception as e:
            print(f"Error: {e}")