import os
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import openpyxl
from datetime import datetime, date
import uuid
from werkzeug.utils import secure_filename
import logging
import re
import io
from typing import Tuple, Dict, List, Union
import threading

# Importation de PyMySQL
import pymysql.cursors
from pymysql.connections import Connection
from pymysql.err import OperationalError, ProgrammingError

app = Flask(__name__)
# Exposer l'en-tête Content-Disposition pour le frontend
CORS(app, expose_headers=['Content-Disposition']) 

# Configuration
class Config:
    def __init__(self):
        self.UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
        # CORRECTION DE LA TYPO ICI
        self.PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', 'processed') 
        self.FINAL_FOLDER = os.getenv('FINAL_FOLDER', 'final')
        self.ARCHIVE_FOLDER = os.getenv('ARCHIVE_FOLDER', 'archive')
        self.LOG_FOLDER = os.getenv('LOG_FOLDER', 'logs')
        self.MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB
        
        # Configuration MySQL
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
        self.MYSQL_USER = os.getenv('MYSQL_USER', 'root')
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'root') # REMPLACEZ PAR UN MOT DE PASSE SÉCURISÉ EN PROD
        self.MYSQL_DB_NAME = os.getenv('MYSQL_DB_NAME', 'moulinette')

        # Créer les répertoires
        for folder in [self.UPLOAD_FOLDER, self.PROCESSED_FOLDER, 
                      self.FINAL_FOLDER, self.ARCHIVE_FOLDER, self.LOG_FOLDER]:
            os.makedirs(folder, exist_ok=True)

config = Config()
app.config.from_object(config)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_FOLDER, 'inventory_processor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pool de connexions MySQL (simple pour l'exemple, un pool plus robuste serait nécessaire en prod)
_db_connection_local = threading.local()

def get_db_connection():
    """Obtient une nouvelle connexion à la base de données MySQL."""
    if not hasattr(_db_connection_local, "connection") or not _db_connection_local.connection.open:
        try:
            _db_connection_local.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                port=config.MYSQL_PORT,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DB_NAME,
                cursorclass=pymysql.cursors.DictCursor # Retourne les résultats sous forme de dictionnaires
            )
            logger.info(f"Nouvelle connexion MySQL établie pour le thread {threading.get_ident()}")
        except OperationalError as e:
            logger.error(f"Erreur de connexion à MySQL: {e}", exc_info=True)
            raise ConnectionError(f"Impossible de se connecter à la base de données MySQL: {e}")
    return _db_connection_local.connection

@app.teardown_appcontext
def close_db_connection(exception):
    """Ferme la connexion à la base de données à la fin de la requête."""
    if hasattr(_db_connection_local, "connection") and _db_connection_local.connection.open:
        _db_connection_local.connection.close()
        logger.info(f"Connexion MySQL fermée pour le thread {threading.get_ident()}")


def create_tables_if_not_exists():
    """Crée les tables MySQL si elles n'existent pas."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Table `sessions`
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `sessions` (
                    `sessionId` VARCHAR(8) PRIMARY KEY,
                    `originalFilePath` VARCHAR(255) NOT NULL,
                    `headerLines` JSON,
                    `timestamp` DATETIME NOT NULL,
                    `status` VARCHAR(50) NOT NULL,
                    `templateFilePath` VARCHAR(255),
                    `completedFilePath` VARCHAR(255),
                    `finalFilePath` VARCHAR(255),
                    `totalDiscrepancy` FLOAT DEFAULT 0,
                    `adjustedItemsCount` INT DEFAULT 0,
                    `strategyUsed` VARCHAR(50),
                    `inventoryDate` DATE
                );
            """)
            # Table `inventoryLines`
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `inventoryLines` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `sessionId` VARCHAR(8) NOT NULL,
                    `originalLineIndex` INT NOT NULL,
                    `originalPartsJson` JSON NOT NULL,
                    `quantiteStockOriginal` FLOAT NOT NULL,
                    `codeArticle` VARCHAR(255) NOT NULL,
                    `numeroLot` VARCHAR(255),
                    `numeroSession` VARCHAR(255),
                    `numeroInventaire` VARCHAR(255),
                    `site` VARCHAR(255),
                    `statut` VARCHAR(50),
                    `unite` VARCHAR(50),
                    `emplacement` VARCHAR(255),
                    `zonePk` VARCHAR(255),
                    `dateLot` DATETIME,
                    `quantiteCorrigee` FLOAT,
                    FOREIGN KEY (`sessionId`) REFERENCES `sessions`(`sessionId`) ON DELETE CASCADE
                );
            """)
            # Table `aggregatedArticles`
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `aggregatedArticles` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `sessionId` VARCHAR(8) NOT NULL,
                    `codeArticle` VARCHAR(255) NOT NULL,
                    `statut` VARCHAR(50),
                    `emplacement` VARCHAR(255),
                    `zonePk` VARCHAR(255),
                    `unite` VARCHAR(50),
                    `quantiteTheoriqueTotale` FLOAT NOT NULL,
                    `numeroSession` VARCHAR(255),
                    `numeroInventaire` VARCHAR(255),
                    `site` VARCHAR(255),
                    `dateMin` DATETIME,
                    `quantiteReelleSaisie` FLOAT,
                    `ecartCalcule` FLOAT,
                    FOREIGN KEY (`sessionId`) REFERENCES `sessions`(`sessionId`) ON DELETE CASCADE,
                    UNIQUE KEY `idx_unique_aggregation` (`sessionId`, `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`)
                );
            """)
        conn.commit()
        logger.info("Tables MySQL vérifiées/créées avec succès.")
    except (OperationalError, ProgrammingError) as e:
        logger.error(f"Erreur lors de la création des tables MySQL: {e}", exc_info=True)
        raise ConnectionError(f"Impossible de créer les tables MySQL: {e}")
    finally:
        if conn and conn.open: 
            pass 

# Exécuter la création des tables au démarrage de l'application
with app.app_context():
    create_tables_if_not_exists()


class SageX3Processor:
    """Classe principale pour le traitement des fichiers d'inventaire Sage X3"""
    
    def __init__(self):
        self._lock = threading.Lock() # Pour les opérations concurrentes sur les fichiers et la DB

        # Configuration des colonnes Sage X3 (indices basés sur 0)
        self.SAGE_COLUMNS = {
            'TYPE_LIGNE': 0,
            'NUMERO_SESSION': 1,
            'NUMERO_INVENTAIRE': 2, # Contient la date de l'inventaire
            'RANG': 3,
            'SITE': 4,
            'QUANTITE': 5,
            'INDICATEUR_COMPTE': 6,
            'CODE_ARTICLE': 7,
            'EMPLACEMENT': 8, 
            'STATUT': 9,      
            'UNITE': 10,      
            'VALEUR': 11,
            'ZONE_PK': 12,    
            'NUMERO_LOT': 13,
        }
        # Liste des noms de colonnes dans l'ordre pour la reconstruction du fichier Sage X3
        self.SAGE_COLUMN_NAMES_ORDERED = [
            'TYPE_LIGNE', 'NUMERO_SESSION', 'NUMERO_INVENTAIRE', 'RANG', 'SITE',
            'QUANTITE', 'INDICATEUR_COMPTE', 'CODE_ARTICLE', 'EMPLACEMENT',
            'STATUT', 'UNITE', 'VALEUR', 'ZONE_PK', 'NUMERO_LOT'
        ]

    def get_num_inventory_lines(self, session_id: str) -> int:
        """Helper pour obtenir le nombre de lignes S; pour une session"""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM `inventoryLines` WHERE `sessionId` = %s", (session_id,))
                result = cursor.fetchone()
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Erreur lors du comptage des lignes d'inventaire: {e}", exc_info=True)
            return 0

    def _extract_inventory_date_from_num_inventaire(self, numero_inventaire: str, session_creation_timestamp: datetime) -> Union[date, None]:
        """
        Extrait la date (jour, mois) du numéro d'inventaire et utilise l'année de création de la session.
        Ex: ABJ012507INV00000002 -> 25/07/<session_creation_year>
        """
        # Regex pour capturer DDMM avant 'INV'
        match = re.search(r'(\d{2})(\d{2})INV', numero_inventaire)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                # Utilise l'année de la création de la session pour la date de l'inventaire
                year = session_creation_timestamp.year
                return date(year, month, day)
            except ValueError:
                logger.warning(f"Date invalide (jour/mois) dans le numéro d'inventaire: {numero_inventaire}")
        return None

    def extract_date_from_lot(self, lot_number: str) -> Union[datetime, None]:
        """Extrait une date d'un numéro de lot Sage X3"""
        if pd.isna(lot_number):
            return None
            
        # Pattern pour les lots de format CPKU###MMYY####
        match = re.search(r'CPKU\d{3}(\d{2})(\d{2})\d{4}', str(lot_number))
        if match:
            try:
                month = int(match.group(1))
                year = int(match.group(2)) + 2000
                return datetime(year, month, 1)
            except ValueError:
                logger.warning(f"Date invalide dans le lot: {lot_number}")
        return None
    
    def validate_sage_file(self, filepath: str, session_id: str, session_creation_timestamp: datetime) -> Tuple[bool, str, List[str], Union[date, None]]:
        """
        Valide la structure d'un fichier Sage X3 et insère les données dans la table inventoryLines.
        Retourne True/False, un message, les lignes d'en-tête, et la date d'inventaire extraite.
        """
        conn = get_db_connection()
        try:
            headers = []
            s_lines_data_to_insert = [] 
            first_s_line_numero_inventaire = None

            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('E;') or line.startswith('L;'):
                        headers.append(line)
                    elif line.startswith('S;'):
                        parts = line.split(';')
                        max_expected_col_index = max(self.SAGE_COLUMNS.values())
                        if len(parts) <= max_expected_col_index:
                            return False, f"Ligne {i+1} : Format de colonnes invalide. Minimum {max_expected_col_index + 1} colonnes requises.", [], None
                        
                        quantite_stock = pd.to_numeric(parts[self.SAGE_COLUMNS['QUANTITE']], errors='coerce')
                        if pd.isna(quantite_stock):
                            return False, f"Ligne {i+1} : Valeur de quantité invalide '{parts[self.SAGE_COLUMNS['QUANTITE']]}'.", [], None
                        
                        date_lot = self.extract_date_from_lot(parts[self.SAGE_COLUMNS['NUMERO_LOT']])

                        if first_s_line_numero_inventaire is None: # Capture le NUMERO_INVENTAIRE de la première ligne S;
                            first_s_line_numero_inventaire = parts[self.SAGE_COLUMNS['NUMERO_INVENTAIRE']]
                        
                        s_lines_data_to_insert.append((
                            session_id,
                            i, # originalLineIndex
                            json.dumps(parts), # originalPartsJson
                            float(quantite_stock), # quantiteStockOriginal
                            parts[self.SAGE_COLUMNS['CODE_ARTICLE']], # codeArticle
                            parts[self.SAGE_COLUMNS['NUMERO_LOT']], # numeroLot
                            parts[self.SAGE_COLUMNS['NUMERO_SESSION']], # numeroSession
                            parts[self.SAGE_COLUMNS['NUMERO_INVENTAIRE']], # numeroInventaire
                            parts[self.SAGE_COLUMNS['SITE']], # site
                            parts[self.SAGE_COLUMNS['STATUT']], # statut
                            parts[self.SAGE_COLUMNS['UNITE']], # unite
                            parts[self.SAGE_COLUMNS['EMPLACEMENT']], # emplacement
                            parts[self.SAGE_COLUMNS['ZONE_PK']], # zonePk
                            date_lot, # dateLot
                            None # quantiteCorrigee initialement NULL
                        ))

            if not s_lines_data_to_insert:
                return False, "Aucune donnée 'S;' trouvée dans le fichier.", [], None
            
            # Extraire la date d'inventaire après avoir lu la première ligne S;
            inventory_date = self._extract_inventory_date_from_num_inventaire(first_s_line_numero_inventaire, session_creation_timestamp)
            if inventory_date is None:
                return False, "Impossible d'extraire une date d'inventaire valide du numéro d'inventaire.", [], None

            with conn.cursor() as cursor:
                # --- CORRECTION ICI : Insérer dans `sessions` d'abord ---
                insert_session_query = """
                    INSERT INTO `sessions` (
                        `sessionId`, `originalFilePath`, `headerLines`, 
                        `timestamp`, `status`, `inventoryDate`
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_session_query, (
                    session_id,
                    filepath,
                    json.dumps(headers),
                    session_creation_timestamp,
                    'uploaded',
                    inventory_date
                ))
                logger.info(f"Session {session_id} insérée dans la table `sessions`.")

                # --- Ensuite, insérer dans `inventoryLines` ---
                insert_query = """
                    INSERT INTO `inventoryLines` (
                        `sessionId`, `originalLineIndex`, `originalPartsJson`, 
                        `quantiteStockOriginal`, `codeArticle`, `numeroLot`, `numeroSession`, 
                        `numeroInventaire`, `site`, `statut`, `unite`, `emplacement`, 
                        `zonePk`, `dateLot`, `quantiteCorrigee`
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(insert_query, s_lines_data_to_insert)
                logger.info(f"{len(s_lines_data_to_insert)} lignes 'S;' insérées pour la session {session_id}.")

            conn.commit()
            
            return True, "Fichier validé et données importées dans MySQL.", headers, inventory_date
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur traitement du fichier complété: {str(e)}", exc_info=True)
            raise
    
    def distribute_discrepancies(self, session_id: str, strategy: str = 'FIFO') -> pd.DataFrame:
        """
        Répartit les écarts selon la stratégie spécifiée et met à jour les quantités corrigées dans MySQL.
        Cette fonction opère sur des sous-ensembles de données pour la scalabilité.
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                select_session_query = "SELECT `status` FROM `sessions` WHERE `sessionId` = %s;"
                cursor.execute(select_session_query, (session_id,))
                session_status = cursor.fetchone()
                if not session_status or session_status['status'] not in ['completedFileProcessed', 'discrepanciesDistributed', 'finalFileGenerated']:
                    raise ValueError("Session invalide ou fichier complété non traité.")
                
                # Récupérer tous les articles agrégés avec leurs écarts calculés
                select_aggregated_with_discrepancy = """
                    SELECT `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`, `ecartCalcule`
                    FROM `aggregatedArticles`
                    WHERE `sessionId` = %s AND `ecartCalcule` != 0;
                """
                cursor.execute(select_aggregated_with_discrepancy, (session_id,))
                articles_with_discrepancy = cursor.fetchall()
                
                if not articles_with_discrepancy:
                    logger.info(f"Aucun écart à distribuer pour la session {session_id}.")
                    update_session_query = """
                        UPDATE `sessions` SET `status` = 'discrepanciesDistributed', `strategyUsed` = %s, `adjustedItemsCount` = 0
                        WHERE `sessionId` = %s;
                    """
                    cursor.execute(update_session_query, (strategy, session_id))
                    conn.commit()
                    return pd.DataFrame() 

                adjusted_items_count = 0
                
                for aggregated_item in articles_with_discrepancy:
                    code_article = aggregated_item['codeArticle']
                    statut = aggregated_item['statut']
                    emplacement = aggregated_item['emplacement']
                    zone_pk = aggregated_item['zonePk']
                    unite = aggregated_item['unite']
                    ecart = float(aggregated_item['ecartCalcule'])

                    # Construire le filtre pour récupérer les lignes d'inventaire spécifiques à cette combinaison
                    filter_conditions = [
                        "`sessionId` = %s",
                        "`codeArticle` = %s",
                        "`statut` = %s",
                        "`emplacement` = %s",
                        "`zonePk` = %s",
                        "`unite` = %s"
                    ]
                    filter_params = [session_id, code_article, statut, emplacement, zone_pk, unite]
                    
                    # Déterminer l'ordre de tri pour les lots
                    order_by_clause = "ORDER BY `dateLot` ASC" if strategy == 'FIFO' else "ORDER BY `dateLot` DESC"
                    
                    select_relevant_lots_query = f"""
                        SELECT `id`, `quantiteStockOriginal`, `quantiteCorrigee`, `dateLot`
                        FROM `inventoryLines`
                        WHERE {' AND '.join(filter_conditions)}
                        {order_by_clause};
                    """
                    cursor.execute(select_relevant_lots_query, filter_params)
                    relevant_lots = cursor.fetchall()

                    if not relevant_lots:
                        logger.warning(f"Aucun lot trouvé pour {code_article}/{statut}/{emplacement}/{zone_pk}/{unite} malgré un écart.")
                        continue

                    update_lot_quantities = [] # Liste pour les mises à jour en masse

                    if ecart > 0:  # Écart positif: il manque des articles (Théorique > Réel)
                        remaining_discrepancy = ecart
                        for lot in relevant_lots:
                            if remaining_discrepancy <= 0:
                                break
                            current_qty = float(lot['quantiteCorrigee'] if lot['quantiteCorrigee'] is not None else lot['quantiteStockOriginal'])
                            ajust = min(current_qty, remaining_discrepancy)
                            
                            new_qty = current_qty - ajust
                            update_lot_quantities.append((new_qty, lot['id']))
                            remaining_discrepancy -= ajust
                            adjusted_items_count += 1
                    
                    elif ecart < 0:  # Écart négatif: il y a plus d'articles que prévu (Réel > Théorique)
                        amount_to_add = abs(ecart)
                        lot_to_adjust = relevant_lots[0] # Appliquer au premier lot selon le tri
                        current_qty = float(lot_to_adjust['quantiteCorrigee'] if lot_to_adjust['quantiteCorrigee'] is not None else lot_to_adjust['quantiteStockOriginal'])
                        
                        new_qty = current_qty + amount_to_add
                        update_lot_quantities.append((new_qty, lot_to_adjust['id']))
                        adjusted_items_count += 1
                    
                    if update_lot_quantities:
                        update_query = "UPDATE `inventoryLines` SET `quantiteCorrigee` = %s WHERE `id` = %s;"
                        cursor.executemany(update_query, update_lot_quantities)
            
                # Mettre à jour le statut de la session
                update_session_query = """
                    UPDATE `sessions` SET `status` = 'discrepanciesDistributed', `strategyUsed` = %s, `adjustedItemsCount` = %s
                    WHERE `sessionId` = %s;
                """
                cursor.execute(update_session_query, (strategy, adjusted_items_count, session_id))
            conn.commit()

            # Retourner un DataFrame des articles avec écarts pour l'aperçu frontend
            return pd.DataFrame(articles_with_discrepancy)
    
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur distribution des écarts: {str(e)}", exc_info=True)
            raise
    
    def generate_final_file(self, session_id: str) -> str:
        """
        Génère le fichier final pour l'export Sage X3 à partir des données corrigées dans MySQL.
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                select_session_query = "SELECT `headerLines`, `status` FROM `sessions` WHERE `sessionId` = %s;"
                cursor.execute(select_session_query, (session_id,))
                session_doc = cursor.fetchone()
                if not session_doc or session_doc['status'] not in ['discrepanciesDistributed', 'finalFileGenerated']:
                    raise ValueError("Session invalide ou écarts non distribués.")
                
                header_lines = json.loads(session_doc['headerLines'])
                
                # Récupérer toutes les lignes d'inventaire corrigées depuis MySQL, triées par originalLineIndex
                select_lines_query = """
                    SELECT `originalPartsJson`, `quantiteStockOriginal`, `quantiteCorrigee`, `originalLineIndex`
                    FROM `inventoryLines`
                    WHERE `sessionId` = %s
                    ORDER BY `originalLineIndex` ASC;
                """
                cursor.execute(select_lines_query, (session_id,))
                inventory_lines_docs = cursor.fetchall()
                
                reconstructed_lines = []
                for doc in inventory_lines_docs:
                    original_parts = list(json.loads(doc['originalPartsJson'])) # Crée une copie modifiable
                    
                    # Utilise quantiteCorrigee si elle existe (non NULL), sinon quantiteStockOriginal
                    corrected_qty = int(doc['quantiteCorrigee']) if doc['quantiteCorrigee'] is not None else int(doc['quantiteStockOriginal']) 

                    # Assurez-vous d'avoir suffisamment de parties avant de modifier l'index 5
                    if len(original_parts) > self.SAGE_COLUMNS['QUANTITE']:
                        original_parts[self.SAGE_COLUMNS['QUANTITE']] = str(corrected_qty)
                        reconstructed_lines.append(';'.join(original_parts))
                    else:
                        logger.warning(f"Ligne originale trop courte pour l'index quantité: {original_parts}. Ligne non modifiée.")
                        reconstructed_lines.append(';'.join(json.loads(doc['originalPartsJson']))) # Ajouter la ligne originale non modifiée

                # Concaténer en-têtes et lignes de données
                final_content = header_lines + reconstructed_lines
                
                # Génération du nom de fichier final
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"sage_x3_inventaire_corrige_{session_id}_{timestamp}.csv"
                filepath = os.path.join(config.FINAL_FOLDER, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for line in final_content:
                        f.write(line + '\n')
                
                # Mettre à jour le chemin du fichier final dans la session MySQL
                update_session_query = """
                    UPDATE `sessions` SET `finalFilePath` = %s, `status` = 'finalFileGenerated' WHERE `sessionId` = %s;
                """
                cursor.execute(update_session_query, (filepath, session_id))
            conn.commit()
            
            return filepath
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur génération fichier final: {str(e)}", exc_info=True)
            raise

# Initialisation du processeur
processor = SageX3Processor()

# Endpoints API
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Endpoint pour l'upload initial"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > config.MAX_FILE_SIZE:
        return jsonify({'error': 'Fichier trop volumineux'}), 413
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Seuls les fichiers CSV sont acceptés'}), 400
    
    session_id = str(uuid.uuid4())[:8]
    filepath = None 
    session_creation_timestamp = datetime.now() # Capture le timestamp de création de la session
    try:
        filename_on_disk = secure_filename(f"{session_id}_{file.filename}")
        filepath = os.path.join(config.UPLOAD_FOLDER, filename_on_disk)
        file.save(filepath)
        
        # Validation et importation des données dans MySQL
        is_valid, message, headers, inventory_date = processor.validate_sage_file(filepath, session_id, session_creation_timestamp)
        if not is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            # Nettoyage des données MySQL si la validation échoue
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM `sessions` WHERE `sessionId` = %s", (session_id,))
                cursor.execute("DELETE FROM `inventoryLines` WHERE `sessionId` = %s", (session_id,))
            conn.commit()
            return jsonify({'error': message}), 400
        
        # Agrégation des données depuis MySQL
        aggregated_preview_df = processor.aggregate_data(session_id)
        
        # Génération du template Excel
        template_file_path = processor.generate_template(session_id)
        
        # Récupérer les stats pour la réponse
        total_quantity = float(aggregated_preview_df['quantiteTheoriqueTotale'].sum()) if not aggregated_preview_df.empty else 0
        
        # Mettre à jour la session avec la quantité totale agrégée
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # totalDiscrepancy est utilisé ici pour le total initial de la quantité théorique agrégée
            cursor.execute("UPDATE `sessions` SET `totalDiscrepancy` = %s WHERE `sessionId` = %s", (total_quantity, session_id)) 
        conn.commit()

        return jsonify({
            'success': True,
            'sessionId': session_id,
            'templateUrl': f"/api/download/template/{session_id}",
            'stats': {
                'nbArticles': len(aggregated_preview_df), 
                'totalQuantity': total_quantity,
                'nbLots': processor.get_num_inventory_lines(session_id), # CORRECTION : Utiliser la méthode de la classe
                'inventoryDate': inventory_date.isoformat() if inventory_date else None 
            }
        })
    
    except Exception as e:
        logger.error(f"Erreur upload: {str(e)}", exc_info=True)
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        # Nettoyage complet en cas d'erreur
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            cursor.execute("DELETE FROM `inventoryLines` WHERE `sessionId` = %s", (session_id,))
            cursor.execute("DELETE FROM `aggregatedArticles` WHERE `sessionId` = %s", (session_id,))
        conn.commit()
        return jsonify({'error': 'Erreur interne du serveur lors de l\'upload initial'}), 500

@app.route('/api/process', methods=['POST'])
def process_completed_file_route():
    """Endpoint pour traiter le fichier complété, calculer les écarts et générer le fichier final."""
    if 'file' not in request.files or 'sessionId' not in request.form: # sessionId dans le form
        return jsonify({'error': 'Paramètres manquants'}), 400
    
    try:
        session_id = request.form['sessionId'] # Récupération en camelCase
        file = request.files['file']
        strategy = request.form.get('strategy', 'FIFO')
        
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({'error': 'Seuls les fichiers Excel sont acceptés'}), 400
        
        filename_on_disk = secure_filename(f"completed_{session_id}_{file.filename}")
        filepath = os.path.join(config.PROCESSED_FOLDER, filename_on_disk)
        file.save(filepath)
        
        # Traitement du fichier complété et mise à jour des écarts
        processed_summary_df = processor.process_completed_file(session_id, filepath)
        
        # Distribution des écarts et mise à jour des quantités corrigées
        distributed_summary_df = processor.distribute_discrepancies(session_id, strategy)
        
        # Génération du fichier final
        final_file_path = processor.generate_final_file(session_id)
        
        # Récupérer le document de session mis à jour pour les stats
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT `totalDiscrepancy`, `adjustedItemsCount`, `strategyUsed` FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            session_stats = cursor.fetchone()

        return jsonify({
            'success': True,
            'finalUrl': f"/api/download/final/{session_id}",
            'stats': {
                'totalDiscrepancy': session_stats.get('totalDiscrepancy', 0) if session_stats else 0,
                'adjustedItems': session_stats.get('adjustedItemsCount', 0) if session_stats else 0, 
                'strategyUsed': session_stats.get('strategyUsed', 'N/A') if session_stats else 'N/A'
            }
        })
    
    except ValueError as e:
        logger.error(f"Erreur de validation/logique: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur traitement du fichier complété: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/distribute/<strategy>', methods=['POST'])
def redistribute(strategy: str):
    """Endpoint pour re-répartir avec une autre stratégie (agit sur les données en MySQL)."""
    if 'sessionId' not in request.form:
        return jsonify({'error': 'Session ID manquant'}), 400
    
    if strategy not in ['FIFO', 'LIFO']:
        return jsonify({'error': 'Stratégie non supportée'}), 400
    
    try:
        session_id = request.form['sessionId']
        
        # Répartition avec nouvelle stratégie
        distributed_summary_df = processor.distribute_discrepancies(session_id, strategy)
        final_file_path = processor.generate_final_file(session_id)
        
        # Récupérer la session mise à jour pour les stats
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT `totalDiscrepancy`, `adjustedItemsCount`, `strategyUsed` FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            session_stats = cursor.fetchone()

        return jsonify({
            'success': True,
            'finalUrl': f"/api/download/final/{session_id}",
            'strategyUsed': session_stats.get('strategyUsed', 'N/A') if session_stats else 'N/A',
            'adjustedItems': session_stats.get('adjustedItemsCount', 0) if session_stats else 0
        })
    
    except ValueError as e:
        logger.error(f"Erreur de validation/logique redistribution: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur redistribution: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/download/<file_type>/<session_id>', methods=['GET'])
def download_file(file_type: str, session_id: str):
    """Endpoint de téléchargement unifié."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT `templateFilePath`, `finalFilePath` FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            session_data = cursor.fetchone()
        
        if not session_data:
            return jsonify({'error': 'Session invalide ou non trouvée'}), 404
        
        filepath = None
        download_name = None
        mimetype = None

        if file_type == 'template':
            filepath = session_data.get('templateFilePath')
            if not filepath:
                return jsonify({'error': 'Chemin du template non trouvé pour cette session.'}), 404
            download_name = os.path.basename(filepath)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif file_type == 'final':
            filepath = session_data.get('finalFilePath') 
            if not filepath:
                return jsonify({'error': 'Fichier final non généré'}), 404
            download_name = os.path.basename(filepath)
            mimetype = 'text/csv'
        else:
            return jsonify({'error': 'Type de fichier invalide'}), 400
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Fichier non trouvé sur le serveur.'}), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=download_name,
            mimetype=mimetype
        )
    
    except Exception as e:
        logger.error(f"Erreur téléchargement: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """Liste les sessions existantes avec leurs statuts et statistiques."""
    try:
        sessions_list = []
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT `sessionId`, `status`, `timestamp`, `originalFilePath`, 
                       `totalDiscrepancy`, `adjustedItemsCount`, `strategyUsed`, `inventoryDate`
                FROM `sessions`
                ORDER BY `timestamp` DESC;
            """)
            session_docs = cursor.fetchall()

            for doc in session_docs:
                nb_articles = 0
                total_quantity = 0
                
                cursor.execute("""
                    SELECT COUNT(*) AS nbArticlesAgg, SUM(`quantiteTheoriqueTotale`) AS totalQtyAgg
                    FROM `aggregatedArticles`
                    WHERE `sessionId` = %s;
                """, (doc['sessionId'],))
                agg_stats = cursor.fetchone()
                if agg_stats:
                    nb_articles = agg_stats['nbArticlesAgg'] if agg_stats['nbArticlesAgg'] is not None else 0
                    total_quantity = float(agg_stats['totalQtyAgg']) if agg_stats['totalQtyAgg'] is not None else 0

                sessions_list.append({
                    'id': doc['sessionId'],
                    'status': doc.get('status', 'unknown'),
                    'created': doc.get('timestamp').isoformat() if doc.get('timestamp') else None,
                    'originalFile': os.path.basename(doc.get('originalFilePath', '')),
                    'stats': {
                        'nbArticles': nb_articles,
                        'totalQuantity': total_quantity,
                        'totalDiscrepancy': float(doc.get('totalDiscrepancy', 0)),
                        'adjustedItems': doc.get('adjustedItemsCount', 0),
                        'strategyUsed': doc.get('strategyUsed', 'N/A'),
                        'inventoryDate': doc.get('inventoryDate').isoformat() if doc.get('inventoryDate') else None 
                    }
                })
        
        return jsonify({'sessions': sessions_list})
    
    except Exception as e:
        logger.error(f"Erreur listage sessions: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour vérifier la connexion à MySQL."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        db_status = 'connected'
    except Exception:
        db_status = 'disconnected'

    conn = get_db_connection() 
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM `sessions`")
        result = cursor.fetchone()
        active_sessions_count = result['count'] if result else 0

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'activeSessionsCount': active_sessions_count,
        'mysqlStatus': db_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)Erreur de validation ou d'insertion MySQL: {str(e)}", exc_info=True)
            return False, str(e), [], None
    
    def aggregate_data(self, session_id: str) -> pd.DataFrame:
        """
        Agrège les données par CodeArticle, Statut, Emplacement, ZonePk et Unite depuis MySQL
        et stocke les résultats agrégés dans la table `aggregatedArticles`.
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Nettoyer les anciennes agrégations pour cette session avant d'insérer les nouvelles
                cursor.execute("DELETE FROM `aggregatedArticles` WHERE `sessionId` = %s", (session_id,))

                # Agrégation SQL
                aggregate_query = """
                    INSERT INTO `aggregatedArticles` (
                        `sessionId`, `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`,
                        `quantiteTheoriqueTotale`, `numeroSession`, `numeroInventaire`, `site`, `dateMin`
                    )
                    SELECT
                        `sessionId`,
                        `codeArticle`,
                        `statut`,
                        `emplacement`,
                        `zonePk`,
                        `unite`,
                        SUM(`quantiteStockOriginal`) AS quantiteTheoriqueTotale,
                        SUBSTRING_INDEX(GROUP_CONCAT(`numeroSession` ORDER BY `originalLineIndex`), ',', 1) AS numeroSession,
                        SUBSTRING_INDEX(GROUP_CONCAT(`numeroInventaire` ORDER BY `originalLineIndex`), ',', 1) AS numeroInventaire,
                        SUBSTRING_INDEX(GROUP_CONCAT(`site` ORDER BY `originalLineIndex`), ',', 1) AS site,
                        MIN(`dateLot`) AS dateMin
                    FROM
                        `inventoryLines`
                    WHERE
                        `sessionId` = %s
                    GROUP BY
                        `sessionId`, `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`
                    ORDER BY
                        `dateMin` ASC;
                """
                cursor.execute(aggregate_query, (session_id,))
                
                # Récupérer les données agrégées pour l'aperçu frontend
                select_aggregated_query = """
                    SELECT `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`, `quantiteTheoriqueTotale`,
                           `numeroSession`, `numeroInventaire`, `site`, `dateMin`
                    FROM `aggregatedArticles`
                    WHERE `sessionId` = %s
                    ORDER BY `dateMin` ASC;
                """
                cursor.execute(select_aggregated_query, (session_id,))
                aggregated_docs = cursor.fetchall()

                if not aggregated_docs:
                    raise ValueError(f"Aucune donnée agrégée trouvée pour la session {session_id}.")

                # Mettre à jour le statut de la session
                update_session_query = """
                    UPDATE `sessions` SET `status` = 'aggregated', `adjustedItemsCount` = %s WHERE `sessionId` = %s;
                """
                cursor.execute(update_session_query, (len(aggregated_docs), session_id)) # adjustedItemsCount est utilisé ici pour le count d'articles agrégés
            conn.commit()
            logger.info(f"{len(aggregated_docs)} articles agrégés stockés pour la session {session_id}.")

            return pd.DataFrame(aggregated_docs)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur d'agrégation MySQL: {str(e)}", exc_info=True)
            raise
    
    def generate_template(self, session_id: str) -> str:
        """Génère un template Excel pour la saisie à partir des données agrégées de MySQL."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                select_aggregated_query = """
                    SELECT `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`, `quantiteTheoriqueTotale`,
                           `numeroSession`, `numeroInventaire`, `site`
                    FROM `aggregatedArticles`
                    WHERE `sessionId` = %s
                    ORDER BY `dateMin` ASC;
                """
                cursor.execute(select_aggregated_query, (session_id,))
                aggregated_docs = cursor.fetchall()

                select_session_query = "SELECT `inventoryDate` FROM `sessions` WHERE `sessionId` = %s;"
                cursor.execute(select_session_query, (session_id,))
                session_info = cursor.fetchone()
                inventory_date = session_info['inventoryDate'] if session_info else None


            if not aggregated_docs:
                raise ValueError(f"Aucune donnée agrégée trouvée pour la session {session_id}.")

            aggregated_df = pd.DataFrame(aggregated_docs)

            # Récupérer NumeroSession, NumeroInventaire et Site de la première ligne agrégée
            # Ces noms de colonnes sont déjà en camelCase grâce à DictCursor
            session_num = aggregated_df['numeroSession'].iloc[0]
            inventory_num = aggregated_df['numeroInventaire'].iloc[0]
            site_code = aggregated_df['site'].iloc[0] 

            template_data = {
                'Numéro Session': [session_num] * len(aggregated_df),
                'Numéro Inventaire': [inventory_num] * len(aggregated_df),
                'Date Inventaire': [inventory_date.strftime('%Y-%m-%d')] * len(aggregated_df) if inventory_date else [''] * len(aggregated_df),
                'Code Article': aggregated_df['codeArticle'],
                'Statut Article': aggregated_df['statut'],
                'Quantité Théorique': aggregated_df['quantiteTheoriqueTotale'],  # CORRECTION : utiliser les vraies valeurs
                'Quantité Réelle': 0, 
                'Unites': aggregated_df['unite'],
                'Depots': aggregated_df['zonePk'],
                'Emplacements': aggregated_df['emplacement'],
            }
            
            template_df = pd.DataFrame(template_data)
            
            # Construction du nom de fichier selon le nouveau format
            filename = f"{site_code}_{inventory_num}_{session_id}.xlsx"
            filepath = os.path.join(config.PROCESSED_FOLDER, filename)
            
            # Optimisation de l'écriture Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='Inventaire')
                
                # Ajustement automatique des colonnes
                worksheet = writer.sheets['Inventaire']
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value)) for cell in column if cell.value is not None)
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
            
            # Mettre à jour le chemin du template dans la session MySQL
            with conn.cursor() as cursor:
                update_session_query = """
                    UPDATE `sessions` SET `templateFilePath` = %s, `status` = 'templateGenerated' WHERE `sessionId` = %s;
                """
                cursor.execute(update_session_query, (filepath, session_id))
            conn.commit()

            return filepath
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur génération template: {str(e)}", exc_info=True)
            raise
    
    def validate_completed_template(self, df: pd.DataFrame) -> bool:
        """Valide le fichier Excel complété"""
        required_columns = {
            'Numéro Session', 'Numéro Inventaire', 'Date Inventaire', 
            'Code Article', 'Statut Article', 'Quantité Théorique', 'Quantité Réelle',
            'Unites', 'Depots', 'Emplacements'
        }
        if not required_columns.issubset(df.columns):
            logger.error(f"Colonnes manquantes dans le fichier complété: {required_columns - set(df.columns)}")
            return False
        
        df['Quantité Réelle'] = pd.to_numeric(df['Quantité Réelle'], errors='coerce')
        if df['Quantité Réelle'].isna().any():
            logger.error("La colonne 'Quantité Réelle' contient des valeurs non numériques ou vides.")
            return False
        return True
    
    def process_completed_file(self, session_id: str, filepath: str) -> pd.DataFrame:
        """
        Traite le fichier complété, calcule les écarts et met à jour les données agrégées dans MySQL.
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                select_session_query = "SELECT `status` FROM `sessions` WHERE `sessionId` = %s;"
                cursor.execute(select_session_query, (session_id,))
                session_status = cursor.fetchone()
                if not session_status:
                    raise ValueError("Session invalide ou non trouvée.")
                
                completed_df = pd.read_excel(filepath)
                
                if not self.validate_completed_template(completed_df):
                    raise ValueError("Fichier complété invalide: vérifiez les colonnes ou les quantités réelles.")
                
                completed_df['Quantité Réelle'] = pd.to_numeric(completed_df['Quantité Réelle'], errors='coerce').fillna(0)

                # Récupérer les quantités théoriques agrégées depuis MySQL
                select_theoretical_query = """
                    SELECT `id`, `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`, `quantiteTheoriqueTotale`
                    FROM `aggregatedArticles`
                    WHERE `sessionId` = %s;
                """
                cursor.execute(select_theoretical_query, (session_id,))
                theoretical_docs = cursor.fetchall()
                theoretical_quantities_df = pd.DataFrame(theoretical_docs)
                
                if theoretical_quantities_df.empty:
                    raise ValueError(f"Aucune donnée théorique agrégée trouvée pour la session {session_id}.")

                # Renommer les colonnes de theoretical_quantities_df pour la fusion
                theoretical_quantities_df.rename(columns={
                    'codeArticle': 'Code Article',
                    'statut': 'Statut Article',
                    'emplacement': 'Emplacements',
                    'zonePk': 'Depots',
                    'unite': 'Unites'
                }, inplace=True)

                # Clés de fusion basées sur l'agrégation
                merge_keys = [
                    'Code Article', 'Statut Article', 'Emplacements', 'Depots', 'Unites'
                ]
                
                # Fusionner les données théoriques agrégées avec les quantités réelles saisies
                merged = pd.merge(
                    theoretical_quantities_df[merge_keys + ['quantiteTheoriqueTotale', 'id']], # Garder l'id pour la mise à jour
                    completed_df[merge_keys + ['Quantité Réelle']],
                    on=merge_keys,
                    how='left'
                )
                
                merged['Quantité Réelle'] = merged['Quantité Réelle'].fillna(0)
                merged['Ecart'] = merged['quantiteTheoriqueTotale'] - merged['Quantité Réelle']
                
                # Mettre à jour les documents agrégés dans MySQL avec les quantités réelles et les écarts
                update_aggregated_query = """
                    UPDATE `aggregatedArticles`
                    SET `quantiteReelleSaisie` = %s, `ecartCalcule` = %s
                    WHERE `id` = %s;
                """
                update_params = [(float(row['Quantité Réelle']), float(row['Ecart']), row['id']) for _, row in merged.iterrows()]
                cursor.executemany(update_aggregated_query, update_params)
                
                total_discrepancy_sum = float(merged['Ecart'].sum())
                
                # Mettre à jour le statut de la session et l'écart total
                update_session_query = """
                    UPDATE `sessions` SET `completedFilePath` = %s, `status` = 'completedFileProcessed', `totalDiscrepancy` = %s
                    WHERE `sessionId` = %s;
                """
                cursor.execute(update_session_query, (filepath, total_discrepancy_sum, session_id))
            conn.commit()
            
            return merged[['Code Article', 'Statut Article', 'Emplacements', 'Depots', 'Unites', 'Quantité Réelle', 'Ecart']].copy()
    
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur traitement du fichier complété: {str(e)}", exc_info=True)
            raise

   
    def distribute_discrepancies(self, session_id: str, strategy: str = 'FIFO') -> pd.DataFrame:
        """
        Répartit les écarts selon la stratégie spécifiée et met à jour les quantités corrigées dans MySQL.
        Cette fonction opère sur des sous-ensembles de données pour la scalabilité.
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                select_session_query = "SELECT `status` FROM `sessions` WHERE `sessionId` = %s;"
                cursor.execute(select_session_query, (session_id,))
                session_status = cursor.fetchone()
                if not session_status or session_status['status'] not in ['completedFileProcessed', 'discrepanciesDistributed', 'finalFileGenerated']:
                    raise ValueError("Session invalide ou fichier complété non traité.")
                
                # Récupérer tous les articles agrégés avec leurs écarts calculés
                select_aggregated_with_discrepancy = """
                    SELECT `codeArticle`, `statut`, `emplacement`, `zonePk`, `unite`, `ecartCalcule`
                    FROM `aggregatedArticles`
                    WHERE `sessionId` = %s AND `ecartCalcule` != 0;
                """
                cursor.execute(select_aggregated_with_discrepancy, (session_id,))
                articles_with_discrepancy = cursor.fetchall()
                
                if not articles_with_discrepancy:
                    logger.info(f"Aucun écart à distribuer pour la session {session_id}.")
                    update_session_query = """
                        UPDATE `sessions` SET `status` = 'discrepanciesDistributed', `strategyUsed` = %s, `adjustedItemsCount` = 0
                        WHERE `sessionId` = %s;
                    """
                    cursor.execute(update_session_query, (strategy, session_id))
                    conn.commit()
                    return pd.DataFrame() 

                adjusted_items_count = 0
                
                for aggregated_item in articles_with_discrepancy:
                    code_article = aggregated_item['codeArticle']
                    statut = aggregated_item['statut']
                    emplacement = aggregated_item['emplacement']
                    zone_pk = aggregated_item['zonePk']
                    unite = aggregated_item['unite']
                    ecart = float(aggregated_item['ecartCalcule'])

                    # Construire le filtre pour récupérer les lignes d'inventaire spécifiques à cette combinaison
                    filter_conditions = [
                        "`sessionId` = %s",
                        "`codeArticle` = %s",
                        "`statut` = %s",
                        "`emplacement` = %s",
                        "`zonePk` = %s",
                        "`unite` = %s"
                    ]
                    filter_params = [session_id, code_article, statut, emplacement, zone_pk, unite]
                    
                    # Déterminer l'ordre de tri pour les lots
                    order_by_clause = "ORDER BY `dateLot` ASC" if strategy == 'FIFO' else "ORDER BY `dateLot` DESC"
                    
                    select_relevant_lots_query = f"""
                        SELECT `id`, `quantiteStockOriginal`, `quantiteCorrigee`, `dateLot`
                        FROM `inventoryLines`
                        WHERE {' AND '.join(filter_conditions)}
                        {order_by_clause};
                    """
                    cursor.execute(select_relevant_lots_query, filter_params)
                    relevant_lots = cursor.fetchall()

                    if not relevant_lots:
                        logger.warning(f"Aucun lot trouvé pour {code_article}/{statut}/{emplacement}/{zone_pk}/{unite} malgré un écart.")
                        continue

                    update_lot_quantities = [] # Liste pour les mises à jour en masse

                    if ecart > 0:  # Écart positif: il manque des articles (Théorique > Réel)
                        remaining_discrepancy = ecart
                        for lot in relevant_lots:
                            if remaining_discrepancy <= 0:
                                break
                            current_qty = float(lot['quantiteCorrigee'] if lot['quantiteCorrigee'] is not None else lot['quantiteStockOriginal'])
                            ajust = min(current_qty, remaining_discrepancy)
                            
                            new_qty = current_qty - ajust
                            update_lot_quantities.append((new_qty, lot['id']))
                            remaining_discrepancy -= ajust
                            adjusted_items_count += 1
                    
                    elif ecart < 0:  # Écart négatif: il y a plus d'articles que prévu (Réel > Théorique)
                        amount_to_add = abs(ecart)
                        lot_to_adjust = relevant_lots[0] # Appliquer au premier lot selon le tri
                        current_qty = float(lot_to_adjust['quantiteCorrigee'] if lot_to_adjust['quantiteCorrigee'] is not None else lot_to_adjust['quantiteStockOriginal'])
                        
                        new_qty = current_qty + amount_to_add
                        update_lot_quantities.append((new_qty, lot_to_adjust['id']))
                        adjusted_items_count += 1
                    
                    if update_lot_quantities:
                        update_query = "UPDATE `inventoryLines` SET `quantiteCorrigee` = %s WHERE `id` = %s;"
                        cursor.executemany(update_query, update_lot_quantities)
            
                # Mettre à jour le statut de la session
                update_session_query = """
                    UPDATE `sessions` SET `status` = 'discrepanciesDistributed', `strategyUsed` = %s, `adjustedItemsCount` = %s
                    WHERE `sessionId` = %s;
                """
                cursor.execute(update_session_query, (strategy, adjusted_items_count, session_id))
            conn.commit()

            # Retourner un DataFrame des articles avec écarts pour l'aperçu frontend
            return pd.DataFrame(articles_with_discrepancy)
    
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur distribution des écarts: {str(e)}", exc_info=True)
            raise
    
    def generate_final_file(self, session_id: str) -> str:
        """
        Génère le fichier final pour l'export Sage X3 à partir des données corrigées dans MySQL.
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                select_session_query = "SELECT `headerLines`, `status` FROM `sessions` WHERE `sessionId` = %s;"
                cursor.execute(select_session_query, (session_id,))
                session_doc = cursor.fetchone()
                if not session_doc or session_doc['status'] not in ['discrepanciesDistributed', 'finalFileGenerated']:
                    raise ValueError("Session invalide ou écarts non distribués.")
                
                header_lines = json.loads(session_doc['headerLines'])
                
                # Récupérer toutes les lignes d'inventaire corrigées depuis MySQL, triées par originalLineIndex
                select_lines_query = """
                    SELECT `originalPartsJson`, `quantiteStockOriginal`, `quantiteCorrigee`, `originalLineIndex`
                    FROM `inventoryLines`
                    WHERE `sessionId` = %s
                    ORDER BY `originalLineIndex` ASC;
                """
                cursor.execute(select_lines_query, (session_id,))
                inventory_lines_docs = cursor.fetchall()
                
                reconstructed_lines = []
                for doc in inventory_lines_docs:
                    original_parts = list(json.loads(doc['originalPartsJson'])) # Crée une copie modifiable
                    
                    # Utilise quantiteCorrigee si elle existe (non NULL), sinon quantiteStockOriginal
                    corrected_qty = int(doc['quantiteCorrigee']) if doc['quantiteCorrigee'] is not None else int(doc['quantiteStockOriginal']) 

                    # Assurez-vous d'avoir suffisamment de parties avant de modifier l'index 5
                    if len(original_parts) > self.SAGE_COLUMNS['QUANTITE']:
                        original_parts[self.SAGE_COLUMNS['QUANTITE']] = str(corrected_qty)
                        reconstructed_lines.append(';'.join(original_parts))
                    else:
                        logger.warning(f"Ligne originale trop courte pour l'index quantité: {original_parts}. Ligne non modifiée.")
                        reconstructed_lines.append(';'.join(json.loads(doc['originalPartsJson']))) # Ajouter la ligne originale non modifiée

                # Concaténer en-têtes et lignes de données
                final_content = header_lines + reconstructed_lines
                
                # Génération du nom de fichier final
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"sage_x3_inventaire_corrige_{session_id}_{timestamp}.csv"
                filepath = os.path.join(config.FINAL_FOLDER, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for line in final_content:
                        f.write(line + '\n')
                
                # Mettre à jour le chemin du fichier final dans la session MySQL
                update_session_query = """
                    UPDATE `sessions` SET `finalFilePath` = %s, `status` = 'finalFileGenerated' WHERE `sessionId` = %s;
                """
                cursor.execute(update_session_query, (filepath, session_id))
            conn.commit()
            
            return filepath
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur génération fichier final: {str(e)}", exc_info=True)
            raise

# Initialisation du processeur
processor = SageX3Processor()

# Endpoints API
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Endpoint pour l'upload initial"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > config.MAX_FILE_SIZE:
        return jsonify({'error': 'Fichier trop volumineux'}), 413
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Seuls les fichiers CSV sont acceptés'}), 400
    
    session_id = str(uuid.uuid4())[:8]
    filepath = None 
    session_creation_timestamp = datetime.now() # Capture le timestamp de création de la session
    try:
        filename_on_disk = secure_filename(f"{session_id}_{file.filename}")
        filepath = os.path.join(config.UPLOAD_FOLDER, filename_on_disk)
        file.save(filepath)
        
        # Validation et importation des données dans MySQL
        is_valid, message, headers, inventory_date = processor.validate_sage_file(filepath, session_id, session_creation_timestamp)
        if not is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            # Nettoyage des données MySQL si la validation échoue
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM `sessions` WHERE `sessionId` = %s", (session_id,))
                cursor.execute("DELETE FROM `inventoryLines` WHERE `sessionId` = %s", (session_id,))
            conn.commit()
            return jsonify({'error': message}), 400
        
        # Agrégation des données depuis MySQL
        aggregated_preview_df = processor.aggregate_data(session_id)
        
        # Génération du template Excel
        template_file_path = processor.generate_template(session_id)
        
        # Récupérer les stats pour la réponse
        total_quantity = float(aggregated_preview_df['quantiteTheoriqueTotale'].sum()) if not aggregated_preview_df.empty else 0
        
        # Mettre à jour la session avec la quantité totale agrégée
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # totalDiscrepancy est utilisé ici pour le total initial de la quantité théorique agrégée
            cursor.execute("UPDATE `sessions` SET `totalDiscrepancy` = %s WHERE `sessionId` = %s", (total_quantity, session_id)) 
        conn.commit()

        return jsonify({
            'success': True,
            'sessionId': session_id,
            'templateUrl': f"/api/download/template/{session_id}",
            'stats': {
                'nbArticles': len(aggregated_preview_df), 
                'totalQuantity': total_quantity,
                'nbLots': processor.get_num_inventory_lines(session_id), # Obtenir le nombre de lots depuis la DB
                'inventoryDate': inventory_date.isoformat() if inventory_date else None 
            }
        })
    
    except Exception as e:
        logger.error(f"Erreur upload: {str(e)}", exc_info=True)
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        # Nettoyage complet en cas d'erreur
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            cursor.execute("DELETE FROM `inventoryLines` WHERE `sessionId` = %s", (session_id,))
            cursor.execute("DELETE FROM `aggregatedArticles` WHERE `sessionId` = %s", (session_id,))
        conn.commit()
        return jsonify({'error': 'Erreur interne du serveur lors de l\'upload initial'}), 500

    
    # Helper pour obtenir le nombre de lignes S; pour une session
    def get_num_inventory_lines(self, session_id: str) -> int:
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM `inventoryLines` WHERE `sessionId` = %s", (session_id,))
                count = cursor.fetchone()['COUNT(*)'] # Accès par clé pour DictCursor
                return count
        except Exception as e:
            logger.error(f"Erreur lors du comptage des lignes d'inventaire: {e}", exc_info=True)
            return 0


@app.route('/api/process', methods=['POST'])
def process_completed_file_route():
    """Endpoint pour traiter le fichier complété, calculer les écarts et générer le fichier final."""
    if 'file' not in request.files or 'sessionId' not in request.form: # sessionId dans le form
        return jsonify({'error': 'Paramètres manquants'}), 400
    
    try:
        session_id = request.form['sessionId'] # Récupération en camelCase
        file = request.files['file']
        strategy = request.form.get('strategy', 'FIFO')
        
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({'error': 'Seuls les fichiers Excel sont acceptés'}), 400
        
        filename_on_disk = secure_filename(f"completed_{session_id}_{file.filename}")
        filepath = os.path.join(config.PROCESSED_FOLDER, filename_on_disk)
        file.save(filepath)
        
        # Traitement du fichier complété et mise à jour des écarts
        processed_summary_df = processor.process_completed_file(session_id, filepath)
        
        # Distribution des écarts et mise à jour des quantités corrigées
        distributed_summary_df = processor.distribute_discrepancies(session_id, strategy)
        
        # Génération du fichier final
        final_file_path = processor.generate_final_file(session_id)
        
        # Récupérer le document de session mis à jour pour les stats
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT `totalDiscrepancy`, `adjustedItemsCount`, `strategyUsed` FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            session_stats = cursor.fetchone()

        return jsonify({
            'success': True,
            'finalUrl': f"/api/download/final/{session_id}",
            'stats': {
                'totalDiscrepancy': session_stats.get('totalDiscrepancy', 0) if session_stats else 0,
                'adjustedItems': session_stats.get('adjustedItemsCount', 0) if session_stats else 0, 
                'strategyUsed': session_stats.get('strategyUsed', 'N/A') if session_stats else 'N/A'
            }
        })
    
    except ValueError as e:
        logger.error(f"Erreur de validation/logique: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur traitement du fichier complété: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/distribute/<strategy>', methods=['POST'])
def redistribute(strategy: str):
    """Endpoint pour re-répartir avec une autre stratégie (agit sur les données en MySQL)."""
    if 'sessionId' not in request.form:
        return jsonify({'error': 'Session ID manquant'}), 400
    
    if strategy not in ['FIFO', 'LIFO']:
        return jsonify({'error': 'Stratégie non supportée'}), 400
    
    try:
        session_id = request.form['sessionId']
        
        # Répartition avec nouvelle stratégie
        distributed_summary_df = processor.distribute_discrepancies(session_id, strategy)
        final_file_path = processor.generate_final_file(session_id)
        
        # Récupérer la session mise à jour pour les stats
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT `totalDiscrepancy`, `adjustedItemsCount`, `strategyUsed` FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            session_stats = cursor.fetchone()

        return jsonify({
            'success': True,
            'finalUrl': f"/api/download/final/{session_id}",
            'strategyUsed': session_stats.get('strategyUsed', 'N/A') if session_stats else 'N/A',
            'adjustedItems': session_stats.get('adjustedItemsCount', 0) if session_stats else 0
        })
    
    except ValueError as e:
        logger.error(f"Erreur de validation/logique redistribution: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur redistribution: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/download/<file_type>/<session_id>', methods=['GET'])
def download_file(file_type: str, session_id: str):
    """Endpoint de téléchargement unifié."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT `templateFilePath`, `finalFilePath` FROM `sessions` WHERE `sessionId` = %s", (session_id,))
            session_data = cursor.fetchone()
        
        if not session_data:
            return jsonify({'error': 'Session invalide ou non trouvée'}), 404
        
        filepath = None
        download_name = None
        mimetype = None

        if file_type == 'template':
            filepath = session_data.get('templateFilePath')
            if not filepath:
                return jsonify({'error': 'Chemin du template non trouvé pour cette session.'}), 404
            download_name = os.path.basename(filepath)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif file_type == 'final':
            filepath = session_data.get('finalFilePath') 
            if not filepath:
                return jsonify({'error': 'Fichier final non généré'}), 404
            download_name = os.path.basename(filepath)
            mimetype = 'text/csv'
        else:
            return jsonify({'error': 'Type de fichier invalide'}), 400
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Fichier non trouvé sur le serveur.'}), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=download_name,
            mimetype=mimetype
        )
    
    except Exception as e:
        logger.error(f"Erreur téléchargement: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """Liste les sessions existantes avec leurs statuts et statistiques."""
    try:
        sessions_list = []
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT `sessionId`, `status`, `timestamp`, `originalFilePath`, 
                       `totalDiscrepancy`, `adjustedItemsCount`, `strategyUsed`, `inventoryDate`
                FROM `sessions`
                ORDER BY `timestamp` DESC;
            """)
            session_docs = cursor.fetchall()

            for doc in session_docs:
                nb_articles = 0
                total_quantity = 0
                
                cursor.execute("""
                    SELECT COUNT(*) AS nbArticlesAgg, SUM(`quantiteTheoriqueTotale`) AS totalQtyAgg
                    FROM `aggregatedArticles`
                    WHERE `sessionId` = %s;
                """, (doc['sessionId'],))
                agg_stats = cursor.fetchone()
                if agg_stats:
                    nb_articles = agg_stats['nbArticlesAgg'] if agg_stats['nbArticlesAgg'] is not None else 0
                    total_quantity = float(agg_stats['totalQtyAgg']) if agg_stats['totalQtyAgg'] is not None else 0

                sessions_list.append({
                    'id': doc['sessionId'],
                    'status': doc.get('status', 'unknown'),
                    'created': doc.get('timestamp').isoformat() if doc.get('timestamp') else None,
                    'originalFile': os.path.basename(doc.get('originalFilePath', '')),
                    'stats': {
                        'nbArticles': nb_articles,
                        'totalQuantity': total_quantity,
                        'totalDiscrepancy': float(doc.get('totalDiscrepancy', 0)),
                        'adjustedItems': doc.get('adjustedItemsCount', 0),
                        'strategyUsed': doc.get('strategyUsed', 'N/A'),
                        'inventoryDate': doc.get('inventoryDate').isoformat() if doc.get('inventoryDate') else None 
                    }
                })
        
        return jsonify({'sessions': sessions_list})
    
    except Exception as e:
        logger.error(f"Erreur listage sessions: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour vérifier la connexion à MySQL."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        db_status = 'connected'
    except Exception:
        db_status = 'disconnected'

    conn = get_db_connection() 
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM `sessions`")
        active_sessions_count = cursor.fetchone()['COUNT(*)']

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'activeSessionsCount': active_sessions_count,
        'mysqlStatus': db_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
