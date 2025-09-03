import time
from collections import defaultdict, deque
from typing import Dict, Tuple
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)

class SimpleRateLimiter:
    """Rate limiter simple basé sur la mémoire"""
    
    def __init__(self):
        # Stockage des requêtes par IP
        self.requests: Dict[str, deque] = defaultdict(deque)
        # Configuration par défaut
        self.default_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'upload_per_minute': 5,  # Limite spéciale pour les uploads
        }
    
    def is_allowed(self, client_ip: str, endpoint_type: str = 'default') -> Tuple[bool, Dict]:
        """Vérifie si la requête est autorisée"""
        current_time = time.time()
        
        # Nettoyer les anciennes requêtes
        self._cleanup_old_requests(client_ip, current_time)
        
        # Obtenir les limites pour ce type d'endpoint
        limits = self._get_limits_for_endpoint(endpoint_type)
        
        # Vérifier les limites
        requests_last_minute = self._count_requests_in_window(client_ip, current_time, 60)
        requests_last_hour = self._count_requests_in_window(client_ip, current_time, 3600)
        
        # Enregistrer la requête actuelle
        self.requests[client_ip].append(current_time)
        
        # Vérifier les limites
        if requests_last_minute >= limits['per_minute']:
            return False, {
                'error': 'Trop de requêtes par minute',
                'retry_after': 60,
                'limit': limits['per_minute'],
                'remaining': 0
            }
        
        if requests_last_hour >= limits['per_hour']:
            return False, {
                'error': 'Trop de requêtes par heure',
                'retry_after': 3600,
                'limit': limits['per_hour'],
                'remaining': 0
            }
        
        return True, {
            'limit_minute': limits['per_minute'],
            'remaining_minute': limits['per_minute'] - requests_last_minute - 1,
            'limit_hour': limits['per_hour'],
            'remaining_hour': limits['per_hour'] - requests_last_hour - 1
        }
    
    def _cleanup_old_requests(self, client_ip: str, current_time: float):
        """Nettoie les requêtes anciennes (plus d'1 heure)"""
        if client_ip in self.requests:
            # Garder seulement les requêtes de la dernière heure
            cutoff_time = current_time - 3600
            while (self.requests[client_ip] and 
                   self.requests[client_ip][0] < cutoff_time):
                self.requests[client_ip].popleft()
    
    def _count_requests_in_window(self, client_ip: str, current_time: float, window_seconds: int) -> int:
        """Compte les requêtes dans une fenêtre de temps"""
        if client_ip not in self.requests:
            return 0
        
        cutoff_time = current_time - window_seconds
        return sum(1 for req_time in self.requests[client_ip] if req_time >= cutoff_time)
    
    def _get_limits_for_endpoint(self, endpoint_type: str) -> Dict[str, int]:
        """Obtient les limites pour un type d'endpoint"""
        if endpoint_type == 'upload':
            return {
                'per_minute': self.default_limits['upload_per_minute'],
                'per_hour': self.default_limits['requests_per_hour']
            }
        else:
            return {
                'per_minute': self.default_limits['requests_per_minute'],
                'per_hour': self.default_limits['requests_per_hour']
            }
    
    def get_client_ip(self) -> str:
        """Obtient l'IP du client en tenant compte des proxies"""
        # Vérifier les en-têtes de proxy
        if request.headers.get('X-Forwarded-For'):
            # Prendre la première IP (client original)
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        else:
            return request.remote_addr or 'unknown'

# Instance globale
rate_limiter = SimpleRateLimiter()

def apply_rate_limit(endpoint_type: str = 'default'):
    """Décorateur pour appliquer le rate limiting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            client_ip = rate_limiter.get_client_ip()
            
            is_allowed, info = rate_limiter.is_allowed(client_ip, endpoint_type)
            
            if not is_allowed:
                logger.warning(f"Rate limit dépassé pour {client_ip} sur {endpoint_type}")
                response = jsonify({
                    'error': info['error'],
                    'retry_after': info['retry_after']
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(info['retry_after'])
                return response
            
            # Ajouter les en-têtes de rate limiting à la réponse
            response = func(*args, **kwargs)
            
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit-Minute'] = str(info['limit_minute'])
                response.headers['X-RateLimit-Remaining-Minute'] = str(info['remaining_minute'])
                response.headers['X-RateLimit-Limit-Hour'] = str(info['limit_hour'])
                response.headers['X-RateLimit-Remaining-Hour'] = str(info['remaining_hour'])
            
            return response
        
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator