import os
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FileManager:
    """Gestionnaire avancé des fichiers avec archivage et nettoyage automatique"""
    
    def __init__(self, base_folders: Dict[str, str]):
        self.folders = base_folders
        self.archive_folder = base_folders.get('ARCHIVE_FOLDER', 'archive')
        
        # Créer tous les dossiers nécessaires
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
    
    def archive_session_files(self, session_id: str, session_date: datetime = None) -> bool:
        """Archive tous les fichiers d'une session"""
        try:
            if session_date is None:
                session_date = datetime.now()
            
            # Créer le dossier d'archive avec la date
            archive_date_folder = os.path.join(
                self.archive_folder, 
                session_date.strftime('%Y-%m-%d')
            )
            session_archive_folder = os.path.join(archive_date_folder, session_id)
            os.makedirs(session_archive_folder, exist_ok=True)
            
            files_archived = 0
            
            # Archiver les fichiers de chaque dossier
            for folder_type, folder_path in self.folders.items():
                if folder_type == 'ARCHIVE_FOLDER':
                    continue
                    
                # Chercher les fichiers de cette session
                session_files = self._find_session_files(folder_path, session_id)
                
                if session_files:
                    # Créer un sous-dossier par type
                    type_folder = os.path.join(session_archive_folder, folder_type.lower())
                    os.makedirs(type_folder, exist_ok=True)
                    
                    for file_path in session_files:
                        try:
                            filename = os.path.basename(file_path)
                            archive_path = os.path.join(type_folder, filename)
                            shutil.move(file_path, archive_path)
                            files_archived += 1
                            logger.info(f"Fichier archivé: {file_path} -> {archive_path}")
                        except Exception as e:
                            logger.error(f"Erreur archivage fichier {file_path}: {e}")
            
            # Créer un fichier de métadonnées
            self._create_archive_metadata(session_archive_folder, session_id, files_archived)
            
            logger.info(f"Session {session_id} archivée: {files_archived} fichiers")
            return True
            
        except Exception as e:
            logger.error(f"Erreur archivage session {session_id}: {e}")
            return False
    
    def _find_session_files(self, folder_path: str, session_id: str) -> List[str]:
        """Trouve tous les fichiers d'une session dans un dossier"""
        session_files = []
        try:
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if session_id in filename:
                        file_path = os.path.join(folder_path, filename)
                        if os.path.isfile(file_path):
                            session_files.append(file_path)
        except Exception as e:
            logger.error(f"Erreur recherche fichiers session {session_id} dans {folder_path}: {e}")
        
        return session_files
    
    def _create_archive_metadata(self, archive_folder: str, session_id: str, files_count: int):
        """Crée un fichier de métadonnées pour l'archive"""
        try:
            metadata = {
                'session_id': session_id,
                'archived_at': datetime.now().isoformat(),
                'files_count': files_count,
                'archive_version': '1.0'
            }
            
            metadata_file = os.path.join(archive_folder, 'metadata.json')
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erreur création métadonnées archive {session_id}: {e}")
    
    def cleanup_old_files(self, days_old: int = 7) -> Dict[str, int]:
        """Nettoie les fichiers anciens (non archivés)"""
        cleanup_stats = {}
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for folder_type, folder_path in self.folders.items():
            if folder_type == 'ARCHIVE_FOLDER':
                continue
                
            cleaned_count = 0
            try:
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, filename)
                        if os.path.isfile(file_path):
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mtime < cutoff_date:
                                os.remove(file_path)
                                cleaned_count += 1
                                logger.info(f"Fichier ancien supprimé: {file_path}")
                                
                cleanup_stats[folder_type] = cleaned_count
                
            except Exception as e:
                logger.error(f"Erreur nettoyage dossier {folder_path}: {e}")
                cleanup_stats[folder_type] = 0
        
        total_cleaned = sum(cleanup_stats.values())
        logger.info(f"Nettoyage terminé: {total_cleaned} fichiers supprimés")
        
        return cleanup_stats
    
    def get_folder_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retourne les statistiques des dossiers"""
        stats = {}
        
        for folder_type, folder_path in self.folders.items():
            try:
                if os.path.exists(folder_path):
                    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                    total_size = sum(
                        os.path.getsize(os.path.join(folder_path, f)) 
                        for f in files
                    )
                    
                    stats[folder_type] = {
                        'files_count': len(files),
                        'total_size_mb': round(total_size / (1024 * 1024), 2),
                        'path': folder_path
                    }
                else:
                    stats[folder_type] = {
                        'files_count': 0,
                        'total_size_mb': 0,
                        'path': folder_path
                    }
                    
            except Exception as e:
                logger.error(f"Erreur calcul stats dossier {folder_path}: {e}")
                stats[folder_type] = {
                    'files_count': 0,
                    'total_size_mb': 0,
                    'path': folder_path,
                    'error': str(e)
                }
        
        return stats
    
    def restore_session_from_archive(self, session_id: str, archive_date: str = None) -> bool:
        """Restaure une session depuis l'archive"""
        try:
            # Trouver le dossier d'archive
            if archive_date:
                archive_path = os.path.join(self.archive_folder, archive_date, session_id)
            else:
                # Chercher dans tous les dossiers de date
                archive_path = None
                for date_folder in os.listdir(self.archive_folder):
                    potential_path = os.path.join(self.archive_folder, date_folder, session_id)
                    if os.path.exists(potential_path):
                        archive_path = potential_path
                        break
            
            if not archive_path or not os.path.exists(archive_path):
                logger.error(f"Archive non trouvée pour session {session_id}")
                return False
            
            # Restaurer les fichiers
            restored_count = 0
            for folder_type in os.listdir(archive_path):
                type_archive_path = os.path.join(archive_path, folder_type)
                if os.path.isdir(type_archive_path) and folder_type != 'metadata.json':
                    target_folder = self.folders.get(folder_type.upper())
                    if target_folder:
                        for filename in os.listdir(type_archive_path):
                            source_path = os.path.join(type_archive_path, filename)
                            target_path = os.path.join(target_folder, filename)
                            shutil.copy2(source_path, target_path)
                            restored_count += 1
                            logger.info(f"Fichier restauré: {source_path} -> {target_path}")
            
            logger.info(f"Session {session_id} restaurée: {restored_count} fichiers")
            return True
            
        except Exception as e:
            logger.error(f"Erreur restauration session {session_id}: {e}")
            return False