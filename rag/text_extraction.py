import os
import re
import sys
import argparse
import pdfplumber
from collections import Counter

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
try:
    from config import Config
except ImportError:
    print("錯誤: 找不到 config.py，請確認檔案結構。")
    sys.exit(1)

# 設定路徑
pdf_dirs = Config.PDF_DIR
test_output_dir = os.path.join(current_script_dir, "result")

def get_header_footer_blacklist(pdf_path, threshold=5):
    candidates = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                h, w = page.height, page.width
                # 掃描上下各 10%
                for bbox in [(0, 0, w, h * 0.1), (0, h * 0.9, w, h)]:
                    crop = page.within_bbox(bbox)
                    text = crop.extract_text()
                    if text:
                        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
                        candidates.extend(lines)
    except Exception as e:
        print(f"黑名單掃描失敗: {e}")
    return {text for text, count in Counter(candidates).items() if count >= threshold}

def clean_text(text, blacklist=None):
    """
    執行正則清理與黑名單過濾
    """
    if not text:
        return ""

    # 依照黑名單過濾
    if blacklist:
        lines = text.split('\n')
        # 如果該行文字存在於黑名單中，就移除
        lines = [line for line in lines if line.strip() not in blacklist]
        text = '\n'.join(lines)
    
    return text.strip()

def extract_all_text(pdf_path, skip_first_page=False):
    """
    使用 pdfplumber 提取所有文字並應用過濾
    """
    blacklist = get_header_footer_blacklist(pdf_path)
    if blacklist:
        print(f"  [黑名單命中]: {blacklist}")

    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if skip_first_page and i == 0:
                    continue
                
                page_content = page.extract_text()
                if page_content:
                    cleaned_page = clean_text(page_content, blacklist)
                    full_text += f"\n[Page {i + 1}]\n{cleaned_page}"
    except Exception as e:
        print(f"  讀取 PDF 失敗 {pdf_path}: {e}")
        
    return full_text.strip()

def save_text(text, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [已儲存]: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF 轉純文字工具")
    parser.add_argument("--file", type=str, help="指定單一 PDF 檔案路徑進行測試")
    parser.add_argument("--category", type=str, help="指定特定分類資料夾 (例如: Computer)")
    args = parser.parse_args()

    # 執行模式判斷
    if args.file:
        # 模式 1: 測試單一檔案
        pdf_path = args.file
        if os.path.exists(pdf_path):
            print(f"正在測試單一檔案: {pdf_path}")
            # 簡單判斷是否需跳過第一頁 (基於路徑關鍵字)
            skip = any(cat in pdf_path for cat in ["Computer", "Physics"])
            result_text = extract_all_text(pdf_path, skip_first_page=skip)
            
            save_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
            save_text(result_text, test_output_dir, save_name)
        else:
            print(f"找不到檔案: {pdf_path}")
    else:
        # 模式 2: 批次處理
        for pdf_dir in pdf_dirs:
            category = os.path.basename(os.path.normpath(pdf_dir))
            
            if args.category and args.category != category:
                continue
                
            print(f"\n--- 正在處理分類: {category} ---")
            out_dir = os.path.join(test_output_dir, category)
            skip_first = category in ["Computer", "Physics"]

            if not os.path.exists(pdf_dir):
                continue

            for file in os.listdir(pdf_dir):
                if file.lower().endswith(".pdf"):
                    p_path = os.path.join(pdf_dir, file)
                    print(f"處理中: {file}")
                    
                    text = extract_all_text(p_path, skip_first_page=skip_first)
                    save_text(text, out_dir, file.replace(".pdf", ".txt"))