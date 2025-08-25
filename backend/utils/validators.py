import os
import pandas as pd
from typing import Tuple, Union, List
from werkzeug.utils import secure_filename
import magic
import logging

logger = logging.getLogger(__name__)

class FileValidator:
    """Validateur de fichiers avec sécurité renforcée"""
    
    ALLOWED_MIME_TYPES = {
        'text/csv': ['.csv'],
        'application/vnd.ms-excel': ['.xls'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
        'text/plain': ['.csv'],  # Parfois les CSV sont détectés comme text/plain
        'application/zip': ['.xlsx'],  # Les fichiers XLSX sont des archives ZIP
        'application/x-zip-compressed': ['.xlsx']  # Variante de détection ZIP
    }
    
    @staticmethod
    def validate_file_security(file, max_size: int) -> Tuple[bool, str]:
        """Validation sécurisée du fichier"""
        try:
            # Vérification de la taille
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > max_size:
                return False, f"Fichier trop volumineux ({file_size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB)"
            
            if file_size == 0:
                return False, "Fichier vide"
            
            # Vérification du nom de fichier
            if not file.filename:
                return False, "Nom de fichier manquant"
            
            filename = secure_filename(file.filename)
            if not filename:
                return False, "Nom de fichier invalide"
            
            # Vérification de l'extension
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext:
                return False, "Extension de fichier manquante"
            
            # Vérification du type MIME (si python-magic est disponible)
            try:
                file_content = file.read(1024)  # Lire les premiers 1024 bytes
                file.seek(0)  # Remettre le curseur au début
                
                mime_type = magic.from_buffer(file_content, mime=True)
                
                # Validation spéciale pour les fichiers XLSX
                if file_ext == '.xlsx':
                    # Les fichiers XLSX peuvent être détectés comme ZIP ou comme leur type MIME correct
                    allowed_xlsx_mimes = [
                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        'application/zip',
                        'application/x-zip-compressed'
                    ]
                    if mime_type not in allowed_xlsx_mimes:
                        return False, f"Type MIME non autorisé pour fichier XLSX: {mime_type}"
                elif mime_type not in FileValidator.ALLOWED_MIME_TYPES:
                    if mime_type not in FileValidator.ALLOWED_MIME_TYPES:
                        return False, f"Type de fichier non autorisé: {mime_type}"
                    else:
                        # Vérification normale pour les autres types
                        allowed_extensions = FileValidator.ALLOWED_MIME_TYPES[mime_type]
                        if file_ext not in allowed_extensions:
                            return False, f"Extension {file_ext} non compatible avec le type {mime_type}"
                        
            except ImportError:
                logger.warning("python-magic non disponible, validation MIME ignorée")
                # Fallback sur l'extension uniquement
                allowed_extensions = {'.csv', '.xlsx', '.xls'}
                if file_ext not in allowed_extensions:
                    return False, f"Extension {file_ext} non autorisée"
            except Exception as e:
                logger.warning(f"Erreur lors de la détection MIME: {e}")
                # Fallback sur l'extension uniquement en cas d'erreur
                allowed_extensions = {'.csv', '.xlsx', '.xls'}
                if file_ext not in allowed_extensions:
                    return False, f"Extension {file_ext} non autorisée"
            
            return True, "Fichier valide"
            
        except Exception as e:
            logger.error(f"Erreur validation fichier: {str(e)}")
            return False, f"Erreur de validation: {str(e)}"

class DataValidator:
    """Validateur de données métier"""
    
    @staticmethod
    def validate_sage_structure(df: pd.DataFrame, required_columns: dict) -> Tuple[bool, str]:
        """Valide la structure des données Sage X3"""
        try:
            # Vérification du nombre de colonnes
            max_col_needed = max(required_columns.values())
            if df.shape[1] <= max_col_needed:
                return False, f"Nombre de colonnes insuffisant. Minimum {max_col_needed + 1} colonnes requises, {df.shape[1]} trouvées"
            
            # Vérification des données quantité
            qty_col = required_columns['QUANTITE']
            quantities = pd.to_numeric(df.iloc[:, qty_col], errors='coerce')
            
            if quantities.isna().any():
                invalid_count = quantities.isna().sum()
                return False, f"{invalid_count} valeurs de quantité invalides détectées"
            
            if (quantities < 0).any():
                negative_count = (quantities < 0).sum()
                return False, f"{negative_count} quantités négatives détectées"
            
            # Vérification des codes articles
            article_col = required_columns['CODE_ARTICLE']
            articles = df.iloc[:, article_col].astype(str)
            
            if articles.str.strip().eq('').any():
                empty_count = articles.str.strip().eq('').sum()
                return False, f"{empty_count} codes articles vides détectés"
            
            return True, "Structure valide"
            
        except Exception as e:
            return False, f"Erreur de validation des données: {str(e)}"
    
    @staticmethod
    def validate_template_completion(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
        """Valide le fichier template complété"""
        errors = []
        
        # Colonnes requises
        required_columns = {'Numéro Session', 'Numéro Inventaire', 'Code Article', 'Quantité Théorique', 'Quantité Réelle', 'Numéro Lot'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            errors.append(f"Colonnes manquantes: {', '.join(missing_columns)}")
        
        if 'Quantité Réelle' in df.columns:
            # Conversion et validation des quantités réelles
            real_qty = pd.to_numeric(df['Quantité Réelle'], errors='coerce')
            
            # Vérification des valeurs manquantes
            missing_qty = real_qty.isna()
            if missing_qty.any():
                missing_info = df.loc[missing_qty, ['Code Article', 'Numéro Inventaire', 'Numéro Lot']].apply(
                    lambda x: f"{x['Code Article']} - Lot: {x['Numéro Lot']} (Inv: {x['Numéro Inventaire']})", axis=1
                ).tolist()
                errors.append(f"Quantités réelles manquantes pour: {', '.join(map(str, missing_info[:5]))}")
                if len(missing_info) > 5:
                    errors.append(f"... et {len(missing_info) - 5} autres articles")
            
            # Vérification des valeurs négatives
            negative_qty = real_qty < 0
            if negative_qty.any():
                negative_info = df.loc[negative_qty, ['Code Article', 'Numéro Inventaire', 'Numéro Lot']].apply(
                    lambda x: f"{x['Code Article']} - Lot: {x['Numéro Lot']} (Inv: {x['Numéro Inventaire']})", axis=1
                ).tolist()
                errors.append(f"Quantités négatives pour: {', '.join(map(str, negative_info[:5]))}")
        
        is_valid = len(errors) == 0
        message = "Template valide" if is_valid else "Erreurs détectées"
        
        return is_valid, message, errors