import fitz
import re
import os
import easyocr
import numpy as np
from PIL import Image
from config import Config

pdf_dirs = Config.PDF_DIR
output_dir = Config.TXT_DIR
os.makedirs(output_dir, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=True)

def extract_text_ocr(pdf_path, skip_first_page=False):
    doc = fitz.open(pdf_path)
    full_text = ""

    for i, page in enumerate(doc):
        if skip_first_page and i == 0:
            continue

        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        results = reader.readtext(np.array(img), detail=0)
        page_text = " ".join(results)
        page_text = clean_text(page_text)
        full_text += f"\n[Page {i + 1}]\n{page_text}"

    return full_text.strip()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[👉●•👻😎]', '', text)
    text = re.sub(r'\b\d{4}/\d{1,2}/\d{1,2}\b', '', text)
    text = re.sub(r'\s*,?\s*(Chien-Nan Liu|NCTUEE|Chien-Nan Liu, NCTUEE)', '', text)
    text = re.sub(r'\b\d+-\d+\b', '', text)
    
    return text.strip()

def extract_text(pdf_path, skip_first_page=False):
    doc = fitz.open(pdf_path)
    full_text = ""
    for i, page in enumerate(doc):
        if skip_first_page and i == 0:
            continue  
        page_text = page.get_text()
        if page_text:
            page_text = clean_text(page_text)
            full_text += f"\n[Page {i + 1}]\n" + page_text
    return full_text

def save_text(text, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved cleaned text: {out_path}")

if __name__ == "__main__":
    root_out_dir = output_dir

    for pdf_dir in pdf_dirs:
        category = os.path.basename(os.path.normpath(pdf_dir))  
        out_dir = os.path.join(root_out_dir, category)
        skip_first = category in ["Computer", "Physics"]  

        for file in os.listdir(pdf_dir):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, file)
                print("Extracting:", pdf_path)
                if category == "Physics":
                    text = extract_text_ocr(pdf_path, skip_first_page=skip_first)
                else:
                    text = extract_text(pdf_path, skip_first_page=skip_first)
                filename = os.path.splitext(file)[0] + ".txt"
                save_text(text, out_dir, filename)
