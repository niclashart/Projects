import logging

logger = logging.getLogger(__name__)

class MultilingualHandler:
    """Handler für Mehrsprachigkeit in technischen Dokumenten."""
    
    def __init__(self):
        self.translation_available = False
        try:
            from transformers import MarianMTModel, MarianTokenizer
            # Marian MT Modelle für DE <-> EN
            self.de_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
            self.de_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
            self.en_de_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
            self.en_de_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
            self.translation_available = True
            logger.info("Übersetzungsmodelle erfolgreich geladen")
        except Exception as e:
            logger.warning(f"Übersetzungsmodelle konnten nicht geladen werden: {e}")
            logger.info("System läuft ohne Übersetzungsfunktion")
    
    def translate_text(self, text: str, source_lang: str = 'de', target_lang: str = 'en'):
        """Übersetzt Text zwischen Deutsch und Englisch."""
        if not self.translation_available:
            logger.warning("Übersetzung nicht verfügbar - Text wird unverändert zurückgegeben")
            return text
        
        try:
            if source_lang == 'de' and target_lang == 'en':
                tokens = self.de_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = self.de_en_model.generate(**tokens, max_length=512)
                return self.de_en_tokenizer.decode(translated[0], skip_special_tokens=True)
            elif source_lang == 'en' and target_lang == 'de':
                tokens = self.en_de_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = self.en_de_model.generate(**tokens, max_length=512)
                return self.en_de_tokenizer.decode(translated[0], skip_special_tokens=True)
            else:
                return text
        except Exception as e:
            logger.error(f"Übersetzungsfehler: {e}")
            return text
    
    def translate_query(self, query: str, target_lang: str = 'en'):
        """Übersetzt Benutzeranfrage für besseres Retrieval."""
        return self.translate_text(query, 'de', target_lang)
    
    def translate_answer(self, answer: str, target_lang: str):
        """Übersetzt Antwort in gewünschte Sprache."""
        return self.translate_text(answer, 'en', target_lang)