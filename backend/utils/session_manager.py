import uuid
import time
import threading
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Gestionnaire de sessions avec nettoyage automatique"""
    
    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        self.sessions: Dict[str, dict] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self._lock = threading.RLock()
        
        # Démarrage du nettoyage automatique
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self._cleanup_thread.start()
    
    def create_session(self, **initial_data) -> str:
        """Crée une nouvelle session"""
        with self._lock:
            # Nettoyage si nécessaire
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_oldest_sessions(keep=self.max_sessions - 1)
            
            session_id = str(uuid.uuid4())[:8]
            
            # Éviter les collisions (très improbable mais sécurisé)
            while session_id in self.sessions:
                session_id = str(uuid.uuid4())[:8]
            
            self.sessions[session_id] = {
                'id': session_id,
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'status': 'created',
                **initial_data
            }
            
            logger.info(f"Session créée: {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Récupère une session et met à jour l'accès"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session['last_accessed'] = datetime.now()
            return session
    
    def update_session(self, session_id: str, **updates) -> bool:
        """Met à jour une session"""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(updates)
                self.sessions[session_id]['last_accessed'] = datetime.now()
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Supprime une session"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Session supprimée: {session_id}")
                return True
            return False
    
    def list_sessions(self, include_expired: bool = False) -> list:
        """Liste les sessions actives"""
        with self._lock:
            sessions = []
            cutoff_time = datetime.now() - timedelta(seconds=self.session_timeout)
            
            for session_id, session_data in self.sessions.items():
                if include_expired or session_data['last_accessed'] > cutoff_time:
                    # Copie sécurisée sans les données sensibles
                    session_copy = {
                        'id': session_id,
                        'status': session_data.get('status', 'unknown'),
                        'created_at': session_data['created_at'].isoformat(),
                        'last_accessed': session_data['last_accessed'].isoformat(),
                        'original_file': session_data.get('original_file', ''),
                        'stats': session_data.get('stats', {})
                    }
                    sessions.append(session_copy)
            
            return sorted(sessions, key=lambda x: x['created_at'], reverse=True)
    
    def _cleanup_expired_sessions(self):
        """Nettoyage automatique des sessions expirées (thread en arrière-plan)"""
        while True:
            try:
                time.sleep(300)  # Vérification toutes les 5 minutes
                
                with self._lock:
                    cutoff_time = datetime.now() - timedelta(seconds=self.session_timeout)
                    expired_sessions = [
                        sid for sid, data in self.sessions.items()
                        if data['last_accessed'] < cutoff_time
                    ]
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        logger.info(f"Session expirée supprimée: {session_id}")
                        
            except Exception as e:
                logger.error(f"Erreur nettoyage sessions: {e}")
    
    def _cleanup_oldest_sessions(self, keep: int):
        """Supprime les sessions les plus anciennes"""
        if len(self.sessions) <= keep:
            return
        
        # Tri par dernière utilisation
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Suppression des plus anciennes
        to_remove = len(self.sessions) - keep
        for i in range(to_remove):
            session_id = sorted_sessions[i][0]
            del self.sessions[session_id]
            logger.info(f"Session ancienne supprimée: {session_id}")
    
    def get_stats(self) -> dict:
        """Statistiques des sessions"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(seconds=self.session_timeout)
            active_sessions = sum(
                1 for data in self.sessions.values()
                if data['last_accessed'] > cutoff_time
            )
            
            return {
                'total_sessions': len(self.sessions),
                'active_sessions': active_sessions,
                'expired_sessions': len(self.sessions) - active_sessions,
                'max_sessions': self.max_sessions,
                'session_timeout': self.session_timeout
            }