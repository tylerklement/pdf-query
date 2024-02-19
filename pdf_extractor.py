import io
import os
from PIL import Image

import fitz  # PyMuPDF
import pytesseract
from tqdm import tqdm


class PDFExtractor:
    def __init__(self, tesseract_executable_loc):
        pytesseract.pytesseract.tesseract_cmd = tesseract_executable_loc

    def extract_pdf_text(self, pdf_path):
        """Attempt to extract text directly; use OCR if necessary."""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                # Attempt direct text extraction from the page
                page_text = page.get_text()
                if len(page_text.strip()) > 50:  # Threshold for direct text extraction
                    text += page_text
                else:
                    # If direct text extraction yields little text, use OCR
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")  # Convert the pixmap to bytes
                    img = Image.open(io.BytesIO(img_bytes))
                    text += pytesseract.image_to_string(img)
        return text

    def load_pdfs_as_text(self, folder_path):
        pdf_texts = {}  # Dictionary to store file names and their texts
        for file in tqdm(os.listdir(folder_path), 'Reading PDFs'):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file)
                text = self.extract_pdf_text(pdf_path)
                pdf_texts[file] = text
        return pdf_texts
