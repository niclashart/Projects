import re
import os
import logging
from typing import List
from langchain.schema.document import Document
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class TechnicalDocumentProcessor:
    """Spezialisierte Vorverarbeitung für technische Dokumente."""
    
    def __init__(self):
        # Technische Keywords für bessere Extraktion
        self.technical_patterns = {
            'cpu': r'(?i)(cpu|processor|intel|amd|core i[357]|ryzen)',
            'memory': r'(?i)(ram|memory|gb|tb|ddr[3-5])',
            'ports': r'(?i)(usb|hdmi|vga|displayport|ethernet|audio)',
            'resolution': r'(?i)(\d{3,4}x\d{3,4}|4k|full hd|hd)',
            'specs': r'(?i)(specification|spezifikation|technical data)'
        }
    
    def extract_text_with_ocr(self, pdf_path: str):
        """Extrahiert Text aus PDFs mit OCR-Fallback."""
        try:
            # Versuche zuerst normalen Text zu extrahieren
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Falls wenig Text gefunden, verwende OCR
            if len(text.strip()) < 100:
                logger.info(f"Wenig Text gefunden, verwende OCR für {pdf_path}")
                images = convert_from_path(pdf_path, dpi=300)
                text = ""
                for i, image in enumerate(images):
                    ocr_text = pytesseract.image_to_string(image, lang='deu+eng')
                    text += f"\n--- Seite {i+1} ---\n{ocr_text}"
            
            return text
        except Exception as e:
            logger.error(f"OCR-Fehler für {pdf_path}: {e}")
            return ""
    
    def extract_technical_sections(self, documents: List[Document]) -> List[Document]:
        """Extrahiert technische Abschnitte aus Dokumenten."""
        processed_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Erkenne technische Abschnitte
            technical_score = self.calculate_technical_score(content)
            
            if technical_score > 0.3:  # Threshold für technische Relevanz
                # Bereinige Text
                cleaned_content = self.clean_technical_text(content)
                
                # Erstelle neues Dokument mit Metadaten
                new_doc = Document(
                    page_content=cleaned_content,
                    metadata={
                        **doc.metadata,
                        'technical_score': technical_score,
                        'processed': True
                    }
                )
                processed_docs.append(new_doc)
        
        return processed_docs
    
    def calculate_technical_score(self, text: str) -> float:
        """Berechnet technischen Relevanz-Score."""
        score = 0
        text_lower = text.lower()
        
        for category, pattern in self.technical_patterns.items():
            matches = len(re.findall(pattern, text))
            score += matches * 0.1
        
        # Normalisiere auf 0-1
        return min(score, 1.0)
    
    def clean_technical_text(self, text: str) -> str:
        """Bereinigt technischen Text."""
        # Entferne übermäßige Whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Entferne Sonderzeichen außer technisch relevanten
        text = re.sub(r'[^\w\s\-\.\(\)\+\=\:\;\/]', '', text)
        
        return text.strip()