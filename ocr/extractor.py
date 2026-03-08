"""
ocr/extractor.py
Extracts text from images and PDFs using Tesseract OCR.
Supports English, Hebrew, and Arabic.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.
    Steps: grayscale → denoise → threshold → deskew
    """
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Deskew: find and correct rotation
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 10:  # Only correct small skews
            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return thresh


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using Tesseract.
    Auto-detects and handles English, Hebrew, and Arabic.
    """
    processed = preprocess_image(image_path)

    # Try multilingual OCR first (English + Hebrew + Arabic)
    try:
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, lang="eng+heb+ara", config=config)
        if len(text.strip()) > 20:
            return text.strip()
    except pytesseract.TesseractError:
        pass

    # Fallback: English only
    try:
        text = pytesseract.image_to_string(processed, lang="eng", config="--oem 3 --psm 6")
        return text.strip()
    except pytesseract.TesseractError as e:
        return f"OCR Error: {str(e)}. Make sure Tesseract is installed."


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    First tries native text extraction (fast), then falls back to OCR per page.
    """
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc):
        # Try native text extraction first
        text = page.get_text("text").strip()

        if len(text) > 50:
            full_text.append(text)
        else:
            # Fall back to rendering page as image and running OCR
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            ocr_text = pytesseract.image_to_string(gray, lang="eng+heb+ara", config="--oem 3 --psm 6")
            if ocr_text.strip():
                full_text.append(ocr_text.strip())

    doc.close()
    return "\n\n".join(full_text)
