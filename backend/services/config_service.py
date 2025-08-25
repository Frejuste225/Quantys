import yaml
import os
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ConfigService:
    """Service de gestion de la configuration externe"""
    
    def __init__(self, config_path: str = 'config/sage_mappings.yaml'):
        self.config_path = config_path
        self._config = None
        self.load_config()
    
    def load_config(self):
        """Charge la configuration depuis le fichier YAML"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"Configuration chargée depuis {self.config_path}")
            else:
                logger.warning(f"Fichier de configuration non trouvé: {self.config_path}")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Erreur chargement configuration: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut si le fichier n'existe pas"""
        return {
            'sage_x3': {
                'columns': {
                    'TYPE_LIGNE': 0,
                    'NUMERO_SESSION': 1,
                    'NUMERO_INVENTAIRE': 2,
                    'RANG': 3,
                    'SITE': 4,
                    'QUANTITE': 5,
                    'QUANTITE_REELLE_IN_INPUT': 6,
                    'INDICATEUR_COMPTE': 7,
                    'CODE_ARTICLE': 8,
                    'EMPLACEMENT': 9,
                    'STATUT': 10,
                    'UNITE': 11,
                    'VALEUR': 12,
                    'ZONE_PK': 13,
                    'NUMERO_LOT': 14,
                },
                'validation': {
                    'required_line_types': ['E', 'L', 'S'],
                    'min_columns': 15,
                    'max_file_size_mb': 16
                },
                'processing': {
                    'aggregation_keys': ['CODE_ARTICLE', 'STATUT', 'EMPLACEMENT', 'ZONE_PK', 'UNITE'],
                    'distribution_strategies': ['FIFO', 'LIFO']
                },
                'lot_patterns': {
                    'cpku_pattern': r'CPKU\d{3}(\d{2})(\d{2})\d{4}',
                    'inventory_date_pattern': r'(\d{2})(\d{2})INV'
                }
            }
        }
    
    def get_sage_columns(self) -> Dict[str, int]:
        """Retourne le mapping des colonnes Sage X3"""
        return self._config.get('sage_x3', {}).get('columns', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Retourne la configuration de validation"""
        return self._config.get('sage_x3', {}).get('validation', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Retourne la configuration de traitement"""
        return self._config.get('sage_x3', {}).get('processing', {})
    
    def get_lot_patterns(self) -> Dict[str, str]:
        """Retourne les patterns pour l'extraction des dates de lot"""
        return self._config.get('sage_x3', {}).get('lot_patterns', {})
    
    def get_lot_priority(self) -> List[str]:
        """Retourne l'ordre de priorité des types de lots"""
        return self._config.get('sage_x3', {}).get('lot_priority', ['type1', 'type2', 'type3', 'legacy', 'unknown'])
    
    def reload_config(self):
        """Recharge la configuration depuis le fichier"""
        self.load_config()

# Instance globale
config_service = ConfigService()