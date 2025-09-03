import logging
import traceback
import re
from typing import Dict, Any, Optional
from flask import current_app

logger = logging.getLogger(__name__)

class ErrorSanitizer:
    """Classe pour sanitiser les messages d'erreur avant de les exposer"""
    
    # Patterns à masquer dans les messages d'erreur
    SENSITIVE_PATTERNS = [
        r'(/[a-zA-Z0-9_\-./]+)',  # Chemins de fichiers
        r'(C:\\[a-zA-Z0-9_\-\\.:]+)',  # Chemins Windows
        r'(File "[^"]+", line \d+)',  # Références de fichiers Python
        r'(at 0x[0-9a-fA-F]+)',  # Adresses mémoire
        r'(password[=:]\s*[^\s]+)',  # Mots de passe
        r'(token[=:]\s*[^\s]+)',  # Tokens
        r'(key[=:]\s*[^\s]+)',  # Clés
    ]
    
    # Messages d'erreur génériques pour remplacer les détails techniques
    GENERIC_MESSAGES = {
        'FileNotFoundError': 'Fichier non trouvé',
        'PermissionError': 'Permissions insuffisantes',
        'ValueError': 'Valeur invalide',
        'TypeError': 'Type de données incorrect',
        'KeyError': 'Clé manquante',
        'IndexError': 'Index hors limites',
        'AttributeError': 'Attribut manquant',
        'ImportError': 'Module non disponible',
        'ConnectionError': 'Erreur de connexion',
        'TimeoutError': 'Délai d\'attente dépassé',
    }
    
    @classmethod
    def sanitize_error_message(cls, error: Exception, include_type: bool = True) -> str:
        """Sanitise un message d'erreur pour l'exposition publique"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # En mode développement, on peut être plus verbeux
        if current_app and current_app.config.get('DEBUG', False):
            return cls._sanitize_debug_message(error_message, error_type, include_type)
        
        # En production, utiliser des messages génériques
        generic_message = cls.GENERIC_MESSAGES.get(error_type)
        if generic_message:
            return f"{error_type}: {generic_message}" if include_type else generic_message
        
        # Pour les erreurs inconnues, sanitiser le message
        sanitized = cls._remove_sensitive_info(error_message)
        return f"{error_type}: {sanitized}" if include_type else sanitized
    
    @classmethod
    def _sanitize_debug_message(cls, message: str, error_type: str, include_type: bool) -> str:
        """Sanitise un message pour le mode debug (moins restrictif)"""
        # Masquer seulement les informations vraiment sensibles
        sensitive_patterns = [
            r'(password[=:]\s*[^\s]+)',
            r'(token[=:]\s*[^\s]+)',
            r'(key[=:]\s*[^\s]+)',
        ]
        
        sanitized = message
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[MASKED]', sanitized, flags=re.IGNORECASE)
        
        return f"{error_type}: {sanitized}" if include_type else sanitized
    
    @classmethod
    def _remove_sensitive_info(cls, message: str) -> str:
        """Supprime les informations sensibles d'un message"""
        sanitized = message
        
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[MASKED]', sanitized, flags=re.IGNORECASE)
        
        # Limiter la longueur du message
        if len(sanitized) > 200:
            sanitized = sanitized[:200] + "..."
        
        return sanitized or "Erreur interne"

class APIErrorHandler:
    """Gestionnaire d'erreurs pour l'API"""
    
    @staticmethod
    def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """Gère une erreur et retourne une réponse API standardisée"""
        error_id = APIErrorHandler._generate_error_id()
        
        # Logger l'erreur complète pour le debug
        logger.error(
            f"Erreur {error_id} dans {context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )
        
        # Sanitiser le message pour l'utilisateur
        user_message = ErrorSanitizer.sanitize_error_message(error, include_type=False)
        
        return {
            'error': user_message,
            'error_id': error_id,
            'context': context if context else 'unknown'
        }
    
    @staticmethod
    def handle_validation_error(errors: list, context: str = "validation") -> Dict[str, Any]:
        """Gère les erreurs de validation"""
        error_id = APIErrorHandler._generate_error_id()
        
        logger.warning(f"Erreurs de validation {error_id}: {errors}")
        
        return {
            'error': 'Erreurs de validation détectées',
            'error_id': error_id,
            'context': context,
            'details': errors[:10]  # Limiter à 10 erreurs max
        }
    
    @staticmethod
    def _generate_error_id() -> str:
        """Génère un ID unique pour l'erreur"""
        import uuid
        return str(uuid.uuid4())[:8]

# Décorateur pour gérer les erreurs automatiquement
def handle_api_errors(context: str = ""):
    """Décorateur pour gérer automatiquement les erreurs dans les endpoints"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = APIErrorHandler.handle_error(e, context or func.__name__)
                
                # Déterminer le code de statut HTTP approprié
                status_code = 500
                if isinstance(e, (ValueError, TypeError)):
                    status_code = 400
                elif isinstance(e, FileNotFoundError):
                    status_code = 404
                elif isinstance(e, PermissionError):
                    status_code = 403
                
                from flask import jsonify
                return jsonify(error_response), status_code
        
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator