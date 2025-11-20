from PyPDF2 import PdfReader
import io
def metadata(bytes):
    reader = PdfReader(io.BytesIO(bytes))
    info = reader.metadata
    title = info.title
    return title or None
def get_pdf_title(bytes):
    try:
        #try getting title from metadata first
        title = metadata(bytes)
        if title:
            return title
        #if no title in metadata, try getting from first 2 pages
        pdf = PdfReader(io.BytesIO(bytes))
        lines = []
        for i in pdf.pages[:2]: # get all lines
            text = i.extract_text()
            if text:
                lines.extend(text.split("\n"))
        for i in lines:
            num_percentage = sum(p.isnumeric(p) for p in i/len(i)) #check how much of the line is numbers (to see if likely date)
            if num_percentage > 0.3:
                continue 
            return "_".join(i.strip().split(" "))
        if lines and len(lines[0].strip()) > 0: # return first line
            return "_".join(lines[0].strip().split(" "))
        for i in pdf.pages: # last resort, get first non empty line from all pages
            text = i.extract_text()
            if text:
                for line in text.split("\n"):
                    if line.strip():
                        return "_".join(line.strip().split(" "))
    except:
        pass
    return None
    
    