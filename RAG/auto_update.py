import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging

class DatabaseAutoUpdater:
    """Automatische Überwachung und Update der Vector Database."""
    
    def __init__(self, data_path: str = "./data", metadata_file: str = "./data/.metadata.json"):
        self.data_path = Path(data_path)
        self.metadata_file = Path(metadata_file)
        self.logger = logging.getLogger(__name__)
        
    def get_file_hash(self, file_path: Path) -> str:
        """Berechnet MD5-Hash einer Datei."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Fehler beim Hash-Berechnung für {file_path}: {e}")
            return ""
    
    def get_current_files_status(self) -> Dict[str, Dict]:
        """Gibt den aktuellen Status aller PDF-Dateien zurück."""
        files_status = {}
        
        if not self.data_path.exists():
            return files_status
        
        for pdf_file in self.data_path.glob("*.pdf"):
            try:
                stat = pdf_file.stat()
                files_status[pdf_file.name] = {
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "hash": self.get_file_hash(pdf_file),
                    "path": str(pdf_file)
                }
            except Exception as e:
                self.logger.error(f"Fehler beim Lesen von {pdf_file}: {e}")
        
        return files_status
    
    def load_stored_metadata(self) -> Dict:
        """Lädt gespeicherte Metadaten."""
        if not self.metadata_file.exists():
            return {"files": {}, "last_update": None, "database_version": 1}
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Metadaten: {e}")
            return {"files": {}, "last_update": None, "database_version": 1}
    
    def save_metadata(self, files_status: Dict, database_version: int = None):
        """Speichert aktuelle Metadaten."""
        try:
            # Stelle sicher, dass der Ordner existiert
            self.metadata_file.parent.mkdir(exist_ok=True)
            
            stored_metadata = self.load_stored_metadata()
            if database_version is None:
                database_version = stored_metadata.get("database_version", 1) + 1
            
            metadata = {
                "files": files_status,
                "last_update": datetime.now().isoformat(),
                "database_version": database_version,
                "total_files": len(files_status)
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Metadaten gespeichert: {len(files_status)} Dateien")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Metadaten: {e}")
    
    def check_for_changes(self) -> Tuple[bool, List[str], List[str], List[str]]:
        """
        Prüft auf Änderungen in den PDF-Dateien.
        
        Returns:
            (has_changes, new_files, modified_files, deleted_files)
        """
        current_files = self.get_current_files_status()
        stored_metadata = self.load_stored_metadata()
        stored_files = stored_metadata.get("files", {})
        
        new_files = []
        modified_files = []
        deleted_files = []
        
        # Prüfe auf neue und geänderte Dateien
        for filename, current_info in current_files.items():
            if filename not in stored_files:
                new_files.append(filename)
            else:
                stored_info = stored_files[filename]
                # Prüfe Hash und Größe
                if (current_info["hash"] != stored_info.get("hash", "") or 
                    current_info["size"] != stored_info.get("size", 0)):
                    modified_files.append(filename)
        
        # Prüfe auf gelöschte Dateien
        for filename in stored_files:
            if filename not in current_files:
                deleted_files.append(filename)
        
        has_changes = bool(new_files or modified_files or deleted_files)
        
        if has_changes:
            self.logger.info(f"Änderungen erkannt - Neu: {len(new_files)}, Geändert: {len(modified_files)}, Gelöscht: {len(deleted_files)}")
        
        return has_changes, new_files, modified_files, deleted_files
    
    def should_rebuild_database(self) -> Tuple[bool, str]:
        """
        Entscheidet ob die Database neu erstellt werden soll.
        
        Returns:
            (should_rebuild, reason)
        """
        # Prüfe ob Vector Store existiert
        vector_store_path = Path("./chroma")
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            return True, "Vector Store existiert nicht"
        
        # Prüfe auf Dateiänderungen
        has_changes, new_files, modified_files, deleted_files = self.check_for_changes()
        
        if has_changes:
            reasons = []
            if new_files:
                reasons.append(f"{len(new_files)} neue Datei(en): {', '.join(new_files[:3])}")
            if modified_files:
                reasons.append(f"{len(modified_files)} geänderte Datei(en): {', '.join(modified_files[:3])}")
            if deleted_files:
                reasons.append(f"{len(deleted_files)} gelöschte Datei(en): {', '.join(deleted_files[:3])}")
            
            reason = "Änderungen erkannt - " + "; ".join(reasons)
            return True, reason
        
        return False, "Keine Änderungen"
    
    def update_after_rebuild(self):
        """Aktualisiert Metadaten nach Database-Rebuild."""
        current_files = self.get_current_files_status()
        self.save_metadata(current_files)
        self.logger.info("Metadaten nach Database-Rebuild aktualisiert")
    
    def get_change_summary(self) -> Dict:
        """Gibt eine Zusammenfassung der Änderungen zurück."""
        has_changes, new_files, modified_files, deleted_files = self.check_for_changes()
        
        return {
            "has_changes": has_changes,
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "total_current_files": len(self.get_current_files_status()),
            "summary": self._generate_change_summary(new_files, modified_files, deleted_files)
        }
    
    def _generate_change_summary(self, new_files: List[str], modified_files: List[str], deleted_files: List[str]) -> str:
        """Generiert eine benutzerfreundliche Zusammenfassung."""
        if not any([new_files, modified_files, deleted_files]):
            return "Keine Änderungen"
        
        parts = []
        if new_files:
            parts.append(f"📄 {len(new_files)} neue Dateien")
        if modified_files:
            parts.append(f"✏️ {len(modified_files)} geänderte Dateien")
        if deleted_files:
            parts.append(f"🗑️ {len(deleted_files)} gelöschte Dateien")
        
        return " | ".join(parts)
