# Virtual Environment erstellen (empfohlen)
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# oder
rag_env\Scripts\activate     # Windows

# Dependencies installieren
pip install -r requirements.txt

# System Dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng
sudo apt-get install poppler-utils

# Für macOS mit Homebrew
brew install tesseract tesseract-lang poppler

# Für Windows: Tesseract von GitHub herunterladen
# https://github.com/UB-Mannheim/tesseract/wiki