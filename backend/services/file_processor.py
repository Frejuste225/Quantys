import os
import pandas as pd
import openpyxl
from datetime import datetime, date
import re
import logging
from typing import Tuple, Dict, List, Union
from utils.validators import FileValidator, DataValidator
from services.config_service import config_service

logger = logging.getLogger(__name__)

class FileProcessorService:
    """Service pour le traitement des fichiers Sage X3"""
    
    def __init__(self):
        # Configuration des colonnes Sage X3 depuis le fichier externe
        self.SAGE_COLUMNS = config_service.get_sage_columns()
        self.SAGE_COLUMN_NAMES_ORDERED = list(self.SAGE_COLUMNS.keys())
        self.validation_config = config_service.get_validation_config()
        self.processing_config = config_service.get_processing_config()
        self.lot_patterns = config_service.get_lot_patterns()
        
        logger.info(f"FileProcessorService initialisé avec {len(self.SAGE_COLUMN_NAMES_ORDERED)} colonnes attendues")
        logger.info(f"Colonnes: {self.SAGE_COLUMN_NAMES_ORDERED}")
    
    def reload_config(self):
        """Recharge la configuration depuis le fichier externe"""
        config_service.reload_config()
        self.SAGE_COLUMNS = config_service.get_sage_columns()
        self.SAGE_COLUMN_NAMES_ORDERED = list(self.SAGE_COLUMNS.keys())
        self.validation_config = config_service.get_validation_config()
        self.processing_config = config_service.get_processing_config()
        self.lot_patterns = config_service.get_lot_patterns()
        logger.info("Configuration rechargée depuis le fichier externe")
    
    def detect_file_format(self, filepath: str) -> Tuple[bool, str, Dict]:
        """Détecte automatiquement le format du fichier et sa structure"""
        try:
            file_extension = os.path.splitext(filepath)[1].lower()
            
            if file_extension == '.csv':
                return self._detect_csv_format(filepath)
            elif file_extension in ['.xlsx', '.xls']:
                return self._detect_xlsx_format(filepath)
            else:
                return False, "Extension non supportée", {}
                
        except Exception as e:
            logger.error(f"Erreur détection format: {e}")
            return False, str(e), {}
    
    def _detect_csv_format(self, filepath: str) -> Tuple[bool, str, Dict]:
        """Détecte le format d'un fichier CSV"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()[:10] if line.strip()]
            
            format_info = {
                'total_lines': len(lines),
                'e_lines': [i for i, line in enumerate(lines) if line.startswith('E;')],
                'l_lines': [i for i, line in enumerate(lines) if line.startswith('L;')],
                's_lines': [i for i, line in enumerate(lines) if line.startswith('S;')],
                'columns_per_line': []
            }
            
            for i, line in enumerate(lines):
                cols = len(line.split(';'))
                format_info['columns_per_line'].append(cols)
                logger.info(f"Ligne {i+1}: {cols} colonnes - {line[:100]}...")
            
            return True, "Format détecté", format_info
            
        except Exception as e:
            return False, str(e), {}
    
    def _detect_xlsx_format(self, filepath: str) -> Tuple[bool, str, Dict]:
        """Détecte le format d'un fichier XLSX"""
        try:
            df = pd.read_excel(filepath, header=None, dtype=str, engine='openpyxl')
            
            format_info = {
                'total_rows': len(df),
                'total_cols': len(df.columns),
                'sample_data': []
            }
            
            for i, row in df.head(10).iterrows():
                row_data = [str(val).strip() if pd.notna(val) else '' for val in row.values]
                format_info['sample_data'].append({
                    'row': i+1,
                    'columns': len([x for x in row_data if x]),
                    'first_col': row_data[0] if row_data else '',
                    'data': row_data[:5]
                })
                logger.info(f"Ligne {i+1}: {len(row_data)} colonnes - Première: '{row_data[0] if row_data else ''}' - Données: {row_data[:5]}")
            
            return True, "Format détecté", format_info
            
        except Exception as e:
            return False, str(e), {}
    
    def validate_and_process_sage_file(self, filepath: str, file_extension: str, 
                                     session_creation_timestamp: datetime) -> Tuple[bool, Union[str, pd.DataFrame], List[str], Union[date, None]]:
        """
        Valide et traite un fichier Sage X3
        """
        try:
            # Validation sécurisée du fichier
            # Validation de l'existence du fichier
            if not os.path.exists(filepath):
                return False, "Fichier non trouvé", [], None
            
            # Validation de la taille du fichier
            file_size = os.path.getsize(filepath)
            max_size = 16 * 1024 * 1024  # 16MB
            if file_size > max_size:
                return False, f"Fichier trop volumineux ({file_size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB)", [], None
            
            if file_size == 0:
                return False, "Fichier vide", [], None
            
            headers = []
            data_rows = []
            original_s_lines_raw = []
            first_s_line_numero_inventaire = None
            
            expected_num_cols_for_data = len(self.SAGE_COLUMN_NAMES_ORDERED)
            
            if file_extension == '.csv':
                success, data, headers, inventory_date = self._process_csv_file(
                    filepath, expected_num_cols_for_data, session_creation_timestamp
                )
            elif file_extension in ['.xlsx', '.xls']:
                success, data, headers, inventory_date = self._process_xlsx_file(
                    filepath, expected_num_cols_for_data, session_creation_timestamp
                )
            else:
                return False, "Extension de fichier non supportée", [], None
            
            if not success:
                return False, data, [], None
            
            # Validation des données métier
            is_valid, validation_msg = DataValidator.validate_sage_structure(data, self.SAGE_COLUMNS)
            if not is_valid:
                return False, validation_msg, [], None
            
            return True, data, headers, inventory_date
            
        except Exception as e:
            logger.error(f"Erreur traitement fichier: {str(e)}", exc_info=True)
            return False, str(e), [], None
    
    def _process_csv_file(self, filepath: str, expected_cols: int, 
                         session_timestamp: datetime) -> Tuple[bool, Union[str, pd.DataFrame], List[str], Union[date, None]]:
        """Traite un fichier CSV"""
        headers = []
        data_rows = []
        original_s_lines_raw = []
        first_s_line_numero_inventaire = None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('E;') or line.startswith('L;'):
                        headers.append(line)
                    elif line.startswith('S;'):
                        parts = line.split(';')
                        
                        if len(parts) < expected_cols:
                            return False, f"Ligne {i+1} : Format invalide. {expected_cols} colonnes requises.", [], None
                        
                        if first_s_line_numero_inventaire is None:
                            first_s_line_numero_inventaire = parts[self.SAGE_COLUMNS['NUMERO_INVENTAIRE']]
                        
                        processed_parts = parts[:expected_cols]
                        if len(processed_parts) < expected_cols:
                            processed_parts.extend([''] * (expected_cols - len(processed_parts)))
                        
                        data_rows.append(processed_parts)
                        original_s_lines_raw.append(';'.join(processed_parts))
            
            if not data_rows:
                return False, "Aucune donnée S; trouvée", [], None
            
            # Créer le DataFrame
            df = pd.DataFrame(data_rows, columns=self.SAGE_COLUMN_NAMES_ORDERED)
            df = self._process_dataframe(df, original_s_lines_raw)
            
            # Extraire la date d'inventaire
            inventory_date = self._extract_inventory_date(first_s_line_numero_inventaire, session_timestamp)
            
            return True, df, headers, inventory_date
            
        except Exception as e:
            logger.error(f"Erreur traitement CSV: {e}")
            return False, str(e), [], None
    
    def _process_xlsx_file(self, filepath: str, expected_cols: int, 
                          session_timestamp: datetime) -> Tuple[bool, Union[str, pd.DataFrame], List[str], Union[date, None]]:
        """Traite un fichier XLSX"""
        headers = []
        data_rows = []
        original_s_lines_raw = []
        first_s_line_numero_inventaire = None
        
        try:
            # Lecture du fichier Excel avec gestion d'erreurs améliorée
            try:
                temp_df = pd.read_excel(filepath, header=None, dtype=str, engine='openpyxl')
            except Exception as e:
                logger.error(f"Erreur lecture Excel avec openpyxl: {e}")
                # Fallback avec xlrd pour les anciens formats
                try:
                    temp_df = pd.read_excel(filepath, header=None, dtype=str, engine='xlrd')
                except Exception as e2:
                    logger.error(f"Erreur lecture Excel avec xlrd: {e2}")
                    return False, f"Impossible de lire le fichier Excel: {str(e)}", [], None
            
            logger.info(f"Fichier Excel lu avec succès. Dimensions: {temp_df.shape}")
            logger.info(f"Premières lignes du fichier:")
            for i, row in temp_df.head(5).iterrows():
                logger.info(f"Ligne {i}: {list(row.values)}")
            
            for i, row_series in temp_df.iterrows():
                parts = [str(val).strip() if pd.notna(val) else '' for val in row_series.iloc[:max(self.SAGE_COLUMNS.values()) + 1]]
                
                if not parts:
                    continue
                
                line_type = parts[self.SAGE_COLUMNS['TYPE_LIGNE']] if len(parts) > self.SAGE_COLUMNS['TYPE_LIGNE'] else ''
                logger.debug(f"Ligne {i+1}: Type='{line_type}', Colonnes={len(parts)}, Contenu: {parts[:5]}...")
                
                if line_type in ['E', 'L']:
                    headers.append(';'.join(parts))
                elif line_type == 'S':
                    if len(parts) < expected_cols:
                        logger.error(f"Ligne {i+1} (S;): Format invalide. {expected_cols} colonnes requises, {len(parts)} trouvées.")
                        logger.error(f"Contenu de la ligne: {parts}")
                        return False, f"Ligne {i+1} (S;): Format invalide. {expected_cols} colonnes requises, {len(parts)} trouvées.", [], None
                    
                    processed_parts = parts[:expected_cols]
                    if len(processed_parts) < expected_cols:
                        processed_parts.extend([''] * (expected_cols - len(processed_parts)))
                    
                    if first_s_line_numero_inventaire is None:
                        first_s_line_numero_inventaire = processed_parts[self.SAGE_COLUMNS['NUMERO_INVENTAIRE']]
                    
                    data_rows.append(processed_parts)
                    original_s_lines_raw.append(';'.join(processed_parts))
            
            if not data_rows:
                return False, "Aucune donnée S; trouvée dans le fichier XLSX", [], None
            
            logger.info(f"Traitement terminé. {len(data_rows)} lignes de données S; trouvées.")
            
            # Créer le DataFrame
            df = pd.DataFrame(data_rows, columns=self.SAGE_COLUMN_NAMES_ORDERED)
            df = self._process_dataframe(df, original_s_lines_raw)
            
            # Extraire la date d'inventaire
            inventory_date = self._extract_inventory_date(first_s_line_numero_inventaire, session_timestamp)
            
            return True, df, headers, inventory_date
            
        except Exception as e:
            logger.error(f"Erreur traitement XLSX: {e}")
            return False, str(e), [], None
    
    def _process_dataframe(self, df: pd.DataFrame, original_lines: List[str]) -> pd.DataFrame:
        """Traite le DataFrame après création"""
        # Conversion des types
        df['QUANTITE'] = pd.to_numeric(df['QUANTITE'], errors='coerce')
        
        # Extraction des dates de lot
        df['Date_Lot'] = df['NUMERO_LOT'].apply(self._extract_date_from_lot)
        
        # Ajout des lignes originales
        df['original_s_line_raw'] = original_lines
        
        return df
    
    def _extract_date_from_lot(self, lot_number: str) -> Union[datetime, None]:
        """Extrait une date d'un numéro de lot Sage X3"""
        if pd.isna(lot_number):
            return None
        
        # Pattern depuis la configuration
        pattern = self.lot_patterns.get('cpku_pattern', r'CPKU\d{3}(\d{2})(\d{2})\d{4}')
        match = re.search(pattern, str(lot_number))
        if match:
            try:
                month = int(match.group(1))
                year = int(match.group(2)) + 2000
                return datetime(year, month, 1)
            except ValueError:
                logger.warning(f"Date invalide dans le lot: {lot_number}")
        return None
    
    def _extract_inventory_date(self, numero_inventaire: str, session_timestamp: datetime) -> Union[date, None]:
        """Extrait la date d'inventaire du numéro d'inventaire"""
        if not numero_inventaire:
            return None
        
        # Pattern depuis la configuration
        pattern = self.lot_patterns.get('inventory_date_pattern', r'(\d{2})(\d{2})INV')
        match = re.search(pattern, numero_inventaire)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = session_timestamp.year
                return date(year, month, day)
            except ValueError:
                logger.warning(f"Date invalide dans le numéro d'inventaire: {numero_inventaire}")
        return None
    
    def aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrège les données par clés métier"""
        try:
            if df.empty:
                raise ValueError("DataFrame vide pour l'agrégation")
            
            # Clés d'agrégation depuis la configuration
            aggregation_keys = self.processing_config.get('aggregation_keys', [
                'CODE_ARTICLE', 'STATUT', 'EMPLACEMENT', 'ZONE_PK', 'UNITE'
            ])
            
            aggregated = df.groupby(aggregation_keys).agg(
                Quantite_Theorique_Totale=('QUANTITE', 'sum'),
                Numero_Session=('NUMERO_SESSION', 'first'),
                Numero_Inventaire=('NUMERO_INVENTAIRE', 'first'),
                Site=('SITE', 'first'),
                Date_Min=('Date_Lot', lambda x: min(d for d in x if d is not None) if any(d for d in x if d is not None) else None)
            ).reset_index()
            
            return aggregated.sort_values('Date_Min', na_position='last')
            
        except Exception as e:
            logger.error(f"Erreur d'agrégation: {str(e)}", exc_info=True)
            raise
    
    def generate_template(self, aggregated_df: pd.DataFrame, session_id: str, output_folder: str) -> str:
        """Génère un template Excel pour la saisie"""
        try:
            if aggregated_df.empty:
                raise ValueError(f"Aucune donnée agrégée pour la session {session_id}")
            
            # Récupérer les métadonnées
            session_num = aggregated_df['Numero_Session'].iloc[0]
            inventory_num = aggregated_df['Numero_Inventaire'].iloc[0]
            site_code = aggregated_df['Site'].iloc[0]
            
            template_data = {
                'Numéro Session': [session_num] * len(aggregated_df),
                'Numéro Inventaire': [inventory_num] * len(aggregated_df),
                'Code Article': aggregated_df['CODE_ARTICLE'],
                'Statut Article': aggregated_df['STATUT'],
                'Quantité Théorique': 0,
                'Quantité Réelle': 0,
                'Unites': aggregated_df['UNITE'],
                'Depots': aggregated_df['ZONE_PK'],
                'Emplacements': aggregated_df['EMPLACEMENT'],
            }
            
            template_df = pd.DataFrame(template_data)
            
            # Construction du nom de fichier
            filename = f"{site_code}_{inventory_num}_{session_id}.xlsx"
            filepath = os.path.join(output_folder, filename)
            
            # Écriture Excel avec formatage
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='Inventaire')
                
                worksheet = writer.sheets['Inventaire']
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value)) for cell in column if cell.value is not None)
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
            
            return filepath
            
        except Exception as e:
            logger.error(f"Erreur génération template: {str(e)}", exc_info=True)
            raise
    
    def validate_completed_template(self, filepath: str) -> Tuple[bool, str, List[str]]:
        """Valide le fichier template complété"""
        try:
            df = pd.read_excel(filepath)
            return DataValidator.validate_template_completion(df)
        except Exception as e:
            logger.error(f"Erreur validation template: {e}")
            return False, str(e), []