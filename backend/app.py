import os
import uuid
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/inventory_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Imports des services
from config import config
from services.file_processor import FileProcessorService
from services.session_service import SessionService
from services.lotecart_processor import LotecartProcessor
from utils.validators import FileValidator
from utils.error_handler import handle_api_errors, APIErrorHandler
from utils.rate_limiter import apply_rate_limit
from database import db_manager

# Initialisation Flask
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE
app.config['SECRET_KEY'] = config.SECRET_KEY

# Services
file_processor = FileProcessorService()
session_service = SessionService()
lotecart_processor = LotecartProcessor()

class InventoryProcessor:
    """Processeur principal pour les inventaires Sage X3"""
    
    def __init__(self):
        self.sessions = {}  # Stockage temporaire en mémoire (sera migré vers DB)
        logger.info("InventoryProcessor initialisé")
    
    def process_completed_file(self, session_id: str, completed_file_path: str) -> pd.DataFrame:
        """Traite le fichier template complété et calcule les écarts"""
        try:
            # Vérifier que le fichier existe et est accessible
            if not os.path.exists(completed_file_path):
                raise FileNotFoundError(f"Fichier complété non trouvé: {completed_file_path}")
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(completed_file_path)
            if file_size == 0:
                raise ValueError("Le fichier complété est vide")
            
            logger.info(f"Lecture du fichier complété: {completed_file_path} ({file_size} bytes)")
            
            # Tentative de lecture avec gestion d'erreur améliorée
            try:
                completed_df = pd.read_excel(completed_file_path, engine='openpyxl')
            except Exception as excel_error:
                logger.error(f"Erreur lecture Excel avec openpyxl: {excel_error}")
                # Tentative avec un autre moteur
                try:
                    completed_df = pd.read_excel(completed_file_path, engine='xlrd')
                    logger.info("Lecture réussie avec xlrd")
                except Exception as xlrd_error:
                    logger.error(f"Erreur lecture Excel avec xlrd: {xlrd_error}")
                    
                    # Dernière tentative : lire le fichier comme binaire et utiliser BytesIO
                    try:
                        import io
                        with open(completed_file_path, 'rb') as f:
                            file_content = f.read()
                        completed_df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
                        logger.info("Lecture réussie avec BytesIO")
                    except Exception as bytesio_error:
                        logger.error(f"Erreur lecture avec BytesIO: {bytesio_error}")
                        raise ValueError(f"Impossible de lire le fichier Excel: {excel_error}")        
            
            logger.info(f"Template complété chargé: {len(completed_df)} lignes")
            
            # Sauvegarder le DataFrame complété
            session_service.save_dataframe(session_id, "completed_df", completed_df)
            
            # Charger les données originales
            original_df = session_service.load_dataframe(session_id, "original_df")
            if original_df is None:
                raise ValueError("Données originales non trouvées pour cette session")
            
            # Détecter les candidats LOTECART
            lotecart_candidates = lotecart_processor.detect_lotecart_candidates(completed_df)
            if not lotecart_candidates.empty:
                session_service.save_dataframe(session_id, "lotecart_candidates", lotecart_candidates)
                logger.info(f"🎯 {len(lotecart_candidates)} candidats LOTECART détectés")
            
            # Calculer les écarts
            discrepancies = self._calculate_discrepancies(completed_df, original_df)
            session_service.save_dataframe(session_id, "discrepancies_df", discrepancies)
            
            logger.info(f"Écarts calculés: {len(discrepancies)} lignes avec écarts")
            return discrepancies
            
        except Exception as e:
            logger.error(f"Erreur traitement fichier complété: {e}")
            raise
    
    def _calculate_discrepancies(self, completed_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les écarts entre quantités théoriques et réelles"""
        discrepancies = []
        
        # Créer un dictionnaire des quantités réelles saisies (sans numéro de lot)
        real_quantities_dict = {}
        for _, row in completed_df.iterrows():
            key = (
                row["Code Article"],
                row["Numéro Inventaire"]
            )
            real_quantities_dict[key] = row["Quantité Réelle"]
        
        # Parcourir les données originales et calculer les écarts
        for _, original_row in original_df.iterrows():
            code_article = original_row["CODE_ARTICLE"]
            numero_inventaire = original_row["NUMERO_INVENTAIRE"]
            numero_lot = str(original_row["NUMERO_LOT"]).strip() if pd.notna(original_row["NUMERO_LOT"]) else ""
            quantite_originale = original_row["QUANTITE"]
            
            key = (code_article, numero_inventaire)
            quantite_reelle_saisie = real_quantities_dict.get(key, 0)
            
            # IMPORTANT: Ne pas calculer la quantité corrigée ici
            # Elle sera calculée dans distribute_discrepancies selon FIFO/LIFO
            discrepancy_row = {
                'CODE_ARTICLE': code_article,
                'NUMERO_INVENTAIRE': numero_inventaire,
                'NUMERO_LOT': numero_lot,
                'TYPE_LOT': original_row.get('Type_Lot', 'unknown'),
                'QUANTITE_ORIGINALE': quantite_originale,
                'QUANTITE_REELLE_SAISIE_TOTALE': quantite_reelle_saisie,  # Quantité totale saisie pour l'article
                'AJUSTEMENT': 0,  # Sera calculé dans distribute_discrepancies
                'QUANTITE_CORRIGEE': quantite_originale,  # Initialement = quantité originale
                'Date_Lot': original_row.get('Date_Lot'),
                'original_s_line_raw': original_row.get('original_s_line_raw')
            }
            
            discrepancies.append(discrepancy_row)
        
        return pd.DataFrame(discrepancies)
    
    def distribute_discrepancies(self, session_id: str, strategy: str = 'FIFO') -> pd.DataFrame:
        """Distribue les écarts selon la stratégie choisie (FIFO/LIFO)"""
        try:
            # Charger les écarts calculés
            discrepancies_df = session_service.load_dataframe(session_id, "discrepancies_df")
            if discrepancies_df is None:
                raise ValueError("Écarts non calculés pour cette session")
            
            logger.info(f"🔄 Distribution des écarts selon stratégie {strategy}")
            
            # Grouper par article et inventaire pour calculer les écarts totaux
            distributed_rows = []
            
            for (code_article, numero_inventaire), group in discrepancies_df.groupby(['CODE_ARTICLE', 'NUMERO_INVENTAIRE']):
                # Calculer l'écart total pour cet article
                quantite_originale_totale = group['QUANTITE_ORIGINALE'].sum()
                quantite_reelle_saisie = group['QUANTITE_REELLE_SAISIE_TOTALE'].iloc[0]  # Même valeur pour tous les lots
                ecart_total = quantite_reelle_saisie - quantite_originale_totale
                
                logger.info(f"📊 Article {code_article}: Orig={quantite_originale_totale}, Saisie={quantite_reelle_saisie}, Écart={ecart_total}")
                
                # Trier les lots selon la stratégie
                if strategy == 'FIFO':
                    # FIFO: lots les plus anciens en premier (par date puis par numéro de lot)
                    sorted_group = group.sort_values(['Date_Lot', 'NUMERO_LOT'], na_position='last')
                elif strategy == 'LIFO':
                    # LIFO: lots les plus récents en premier
                    sorted_group = group.sort_values(['Date_Lot', 'NUMERO_LOT'], ascending=[False, False], na_position='first')
                else:
                    # Par défaut, garder l'ordre original
                    sorted_group = group
                
                # Distribuer l'écart sur les lots
                ecart_restant = ecart_total
                
                for _, lot_row in sorted_group.iterrows():
                    quantite_originale_lot = lot_row['QUANTITE_ORIGINALE']
                    
                    if ecart_restant == 0:
                        # Plus d'écart à distribuer, garder la quantité originale
                        quantite_corrigee = quantite_originale_lot
                        ajustement_lot = 0
                    elif ecart_restant > 0:
                        # Écart positif: ajouter du stock
                        if ecart_restant >= quantite_originale_lot:
                            # On peut absorber tout l'écart sur ce lot
                            quantite_corrigee = quantite_originale_lot + min(ecart_restant, quantite_originale_lot)
                            ajustement_lot = min(ecart_restant, quantite_originale_lot)
                            ecart_restant -= ajustement_lot
                        else:
                            # Écart restant plus petit que la quantité du lot
                            quantite_corrigee = quantite_originale_lot + ecart_restant
                            ajustement_lot = ecart_restant
                            ecart_restant = 0
                    else:
                        # Écart négatif: retirer du stock
                        if abs(ecart_restant) >= quantite_originale_lot:
                            # On retire tout le stock de ce lot
                            quantite_corrigee = 0
                            ajustement_lot = -quantite_originale_lot
                            ecart_restant += quantite_originale_lot
                        else:
                            # On retire partiellement
                            quantite_corrigee = quantite_originale_lot + ecart_restant  # ecart_restant est négatif
                            ajustement_lot = ecart_restant
                            ecart_restant = 0
                    
                    # Créer la ligne distribuée
                    distributed_row = lot_row.copy()
                    distributed_row['AJUSTEMENT'] = ajustement_lot
                    distributed_row['QUANTITE_CORRIGEE'] = quantite_corrigee
                    distributed_row['QUANTITE_REELLE_SAISIE'] = quantite_reelle_saisie  # Garder la saisie totale pour info
                    
                    distributed_rows.append(distributed_row)
                    
                    logger.debug(f"  📦 Lot {lot_row['NUMERO_LOT']}: Orig={quantite_originale_lot} → Corrigée={quantite_corrigee} (Ajust={ajustement_lot})")
                
                # Vérifier qu'on a bien distribué tout l'écart
                if abs(ecart_restant) > 0.01:  # Tolérance pour les erreurs d'arrondi
                    logger.warning(f"⚠️ Écart non complètement distribué pour {code_article}: {ecart_restant}")
            
            distributed_df = pd.DataFrame(distributed_rows)
            
            # Charger les candidats LOTECART s'ils existent
            lotecart_candidates = session_service.load_dataframe(session_id, "lotecart_candidates")
            
            # Créer les ajustements LOTECART si nécessaire
            if lotecart_candidates is not None and not lotecart_candidates.empty:
                original_df = session_service.load_dataframe(session_id, "original_df")
                lotecart_adjustments = lotecart_processor.create_lotecart_adjustments(
                    lotecart_candidates, original_df
                )
                logger.info(f"🎯 {len(lotecart_adjustments)} ajustements LOTECART créés")
                
                # Ajouter les ajustements LOTECART
                for lotecart_adj in lotecart_adjustments:
                    lotecart_adj['QUANTITE_REELLE_SAISIE'] = lotecart_adj['QUANTITE_CORRIGEE']
                    distributed_df = pd.concat([distributed_df, pd.DataFrame([lotecart_adj])], ignore_index=True)
            
            # Sauvegarder les données distribuées
            session_service.save_dataframe(session_id, "distributed_df", distributed_df)
            
            # Mettre à jour les statistiques de session
            stats = self._calculate_session_stats(distributed_df)
            session_service.update_session(session_id, 
                                         strategy_used=strategy,
                                         **stats)
            
            logger.info(f"✅ Distribution terminée: {len(distributed_df)} ajustements selon {strategy}")
            return distributed_df
            
        except Exception as e:
            logger.error(f"❌ Erreur distribution écarts: {e}")
            raise
    
    def _calculate_session_stats(self, distributed_df: pd.DataFrame) -> dict:
        """Calcule les statistiques de session"""
        try:
            total_discrepancy = distributed_df['AJUSTEMENT'].sum()
            adjusted_items = len(distributed_df[distributed_df['AJUSTEMENT'] != 0])
            
            return {
                'total_discrepancy': float(total_discrepancy),
                'adjusted_items_count': adjusted_items,
                'status': 'completed'
            }
        except Exception as e:
            logger.error(f"Erreur calcul statistiques: {e}")
            return {'total_discrepancy': 0, 'adjusted_items_count': 0}
    
    def generate_final_file(self, session_id: str) -> str:
        """Génère le fichier final CSV avec les quantités réelles dans la colonne G"""
        try:
            # Charger les données nécessaires
            distributed_df = session_service.load_dataframe(session_id, "distributed_df")
            if distributed_df is None:
                raise ValueError("Données distribuées non trouvées")
            
            # Récupérer les métadonnées de session
            session_data = session_service.get_session_data(session_id)
            if not session_data:
                raise ValueError("Session non trouvée")
            
            header_lines = json.loads(session_data['header_lines']) if session_data['header_lines'] else []
            
            # Créer le dictionnaire des ajustements avec quantités réelles (AVEC numéro de lot)
            adjustments_dict = {}
            for _, row in distributed_df.iterrows():
                key = (
                    row["CODE_ARTICLE"],
                    row["NUMERO_INVENTAIRE"],
                    str(row["NUMERO_LOT"]).strip()
                )
                adjustments_dict[key] = {
                    "qte_theo_ajustee": row["QUANTITE_CORRIGEE"],
                    "qte_reelle_saisie": row.get("QUANTITE_REELLE_SAISIE", row["QUANTITE_CORRIGEE"]),  # Nouvelle donnée
                    "type_lot": row["TYPE_LOT"]
                }
            
            # Générer le nom du fichier final
            original_filename = session_data['original_filename']
            base_name = os.path.splitext(original_filename)[0]
            final_filename = f"{base_name}_corrige_{session_id}.csv"
            final_file_path = os.path.join(config.FINAL_FOLDER, final_filename)
            
            # Générer le fichier final
            with open(final_file_path, 'w', encoding='utf-8') as f:
                # Écrire les en-têtes
                for header in header_lines:
                    f.write(header + "\n")
                
                # Traiter les lignes existantes et ajouter les nouvelles lignes LOTECART
                original_df = session_service.load_dataframe(session_id, "original_df")
                max_line_number = 0
                
                # Traiter chaque ligne originale
                for _, original_row in original_df.iterrows():
                    parts = str(original_row["original_s_line_raw"]).split(";")
                    
                    if len(parts) >= 15:
                        code_article = original_row["CODE_ARTICLE"]
                        numero_inventaire = original_row["NUMERO_INVENTAIRE"]
                        numero_lot = str(original_row["NUMERO_LOT"]).strip()
                        
                        key = (code_article, numero_inventaire, numero_lot)
                        
                        # Mettre à jour le numéro de ligne max
                        try:
                            line_number = int(parts[3])
                            max_line_number = max(max_line_number, line_number)
                        except (ValueError, IndexError):
                            pass
                        
                        # Sauvegarder la quantité originale (elle était dans parts[5])
                        quantite_originale = parts[5]
                        
                        # Vérifier s'il y a un ajustement pour cette ligne
                        if key in adjustments_dict:
                            adjustment_data = adjustments_dict[key]
                            
                            # NOUVELLE LOGIQUE : Inverser les colonnes 5 et 6
                            parts[5] = quantite_originale  # Colonne 5 (F) = Quantité originale du fichier initial
                            qte_theo_ajustee = int(adjustment_data["qte_theo_ajustee"])
                            parts[6] = str(qte_theo_ajustee)  # Colonne 6 (G) = Quantité théorique ajustée
                            
                            # L'indicateur passe à "2" SEULEMENT si la quantité théorique ajustée est 0
                            if qte_theo_ajustee == 0:
                                parts[7] = "2"  # Indicateur de compte ajusté (quantité ajustée = 0)
                            else:
                                parts[7] = "1"  # Indicateur normal (quantité ajustée > 0)
                        else:
                            # Pas d'ajustement, garder les valeurs originales
                            parts[5] = quantite_originale  # Colonne 5 (F) = Quantité originale
                            parts[6] = quantite_originale  # Colonne 6 (G) = Quantité originale (pas d'ajustement)
                            parts[7] = "2"  # Indicateur à 2 car pas d'ajustement
                        
                        # Écrire la ligne
                        f.write(";".join(parts) + "\n")
                
                # Ajouter les nouvelles lignes LOTECART
                lotecart_adjustments = [
                    adj for adj in distributed_df.to_dict('records') 
                    if adj.get('is_new_lotecart', False) and not adj.get('is_existing_update', False)
                ]
                
                if lotecart_adjustments:
                    new_lotecart_lines = lotecart_processor.generate_lotecart_lines(
                        lotecart_adjustments, max_line_number
                    )
                    
                    for line in new_lotecart_lines:
                        # Adapter les lignes LOTECART à la nouvelle logique des colonnes
                        parts = line.split(";")
                        if len(parts) >= 15:
                            # Pour LOTECART : 
                            # Colonne 5 (F) = 0 (quantité originale était 0)
                            # Colonne 6 (G) = quantité trouvée (quantité théorique ajustée)
                            qte_lotecart = parts[5]  # Quantité trouvée
                            parts[5] = "0"  # Colonne 5 (F) = Quantité originale (était 0 pour LOTECART)
                            parts[6] = qte_lotecart  # Colonne 6 (G) = Quantité théorique ajustée
                            line = ";".join(parts)
                        
                        f.write(line + "\n")
                
                logger.info(f"✅ Fichier final généré avec {len(distributed_df)} ajustements dont {len(lotecart_adjustments)} nouvelles lignes LOTECART")
            
            # Mettre à jour la session
            session_service.update_session(session_id, 
                                         final_file_path=final_file_path,
                                         status='completed')
            
            logger.info(f"Fichier final généré: {final_file_path}")
            return final_file_path
            
        except Exception as e:
            logger.error(f"Erreur génération fichier final: {e}")
            raise

# Instance globale du processeur
processor = InventoryProcessor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de santé de l'API"""
    try:
        db_health = db_manager.health_check()
        return jsonify({
            'status': 'healthy' if db_health else 'degraded',
            'database': 'connected' if db_health else 'disconnected',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/upload', methods=['POST'])
@apply_rate_limit('upload')
@handle_api_errors('upload')
def upload_file():
    """Upload et traitement initial du fichier Sage X3"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    # Validation sécurisée du fichier
    is_valid, validation_message = FileValidator.validate_file_security(file, config.MAX_FILE_SIZE)
    if not is_valid:
        return jsonify({'error': validation_message}), 400
    
    # Sauvegarde sécurisée
    filename = secure_filename(file.filename)
    session_id = str(uuid.uuid4())[:8]
    timestamped_filename = f"{session_id}_{filename}"
    file_path = os.path.join(config.UPLOAD_FOLDER, timestamped_filename)
    
    file.save(file_path)
    logger.info(f"Fichier sauvegardé: {file_path}")
    
    # Créer la session en base
    session_creation_timestamp = datetime.now()
    session_service.create_session(
        id=session_id,
        original_filename=filename,
        original_file_path=file_path,
        status='uploaded'
    )
    
    # Traitement du fichier
    file_extension = os.path.splitext(filename)[1].lower()
    success, result, headers, inventory_date = file_processor.validate_and_process_sage_file(
        file_path, file_extension, session_creation_timestamp
    )
    
    if not success:
        session_service.update_session(session_id, status='error')
        return jsonify({'error': result}), 400
    
    # Sauvegarder les données originales
    session_service.save_dataframe(session_id, "original_df", result)
    
    # Agrégation des données
    aggregated_df = file_processor.aggregate_data(result)
    session_service.save_dataframe(session_id, "aggregated_df", aggregated_df)
    
    # Génération du template
    template_path = file_processor.generate_template(aggregated_df, session_id, config.PROCESSED_FOLDER)
    
    # Mise à jour de la session
    session_service.update_session(
        session_id,
        template_file_path=template_path,
        inventory_date=inventory_date,
        nb_articles=len(aggregated_df),
        nb_lots=len(result),
        total_quantity=float(result['QUANTITE'].sum()),
        status='template_generated',
        header_lines=json.dumps(headers)
    )
    
    return jsonify({
        'message': 'Fichier traité avec succès',
        'session_id': session_id,
        'template_url': f'/api/download/template/{session_id}',
        'stats': {
            'nb_articles': len(aggregated_df),
            'total_quantity': float(result['QUANTITE'].sum()),
            'nb_lots': len(result),
            'inventory_date': inventory_date.isoformat() if inventory_date else None
        }
    })

@app.route('/api/process', methods=['POST'])
@apply_rate_limit('upload')
@handle_api_errors('process')
def process_completed_file():
    """Traite le fichier template complété"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'error': 'ID de session manquant'}), 400
    
    strategy = request.form.get('strategy', 'FIFO')
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    # Validation du fichier complété
    logger.info(f"Début validation fichier complété: {file.filename}")
    is_valid, validation_message, validation_errors = file_processor.validate_completed_template(file)
    logger.info(f"Résultat validation: valid={is_valid}, message={validation_message}")
    
    if not is_valid:
        logger.error(f"Validation échouée: {validation_message}, erreurs: {validation_errors}")
        return jsonify({
            'error': validation_message,
            'details': validation_errors
        }), 400
    
    # Sauvegarde du fichier complété avec validation
    completed_filename = f"completed_{session_id}_{secure_filename(file.filename)}"
    completed_file_path = os.path.join(config.PROCESSED_FOLDER, completed_filename)
    
    try:
        # Diagnostic du fichier avant sauvegarde
        file.seek(0)
        file_content = file.read()
        file.seek(0)
        
        logger.info(f"Fichier reçu: {file.filename}, taille: {len(file_content)} bytes")
        
        # Sauvegarde avec gestion d'erreur améliorée
        try:
            file.save(completed_file_path)
        except Exception as save_error:
            logger.error(f"Erreur lors de la sauvegarde: {save_error}")
            # Tentative de sauvegarde alternative
            with open(completed_file_path, 'wb') as f:
                f.write(file_content)
            logger.info("Sauvegarde alternative réussie")
        
        # Vérifier que le fichier existe
        if not os.path.exists(completed_file_path):
            raise FileNotFoundError("Fichier non sauvegardé correctement")
        
        file_size = os.path.getsize(completed_file_path)
        if file_size == 0:
            raise ValueError("Fichier sauvegardé vide")
        
        logger.info(f"Fichier sauvegardé: {completed_file_path} ({file_size} bytes)")
        
        # Attendre un peu pour que le système de fichiers se synchronise
        import time
        time.sleep(0.1)
        
        # Test de lecture avec diagnostic détaillé
        try:
            # Essayer d'abord avec openpyxl
            test_df = pd.read_excel(completed_file_path, engine='openpyxl', nrows=1)
            logger.info(f"Fichier validé avec openpyxl: {len(test_df.columns)} colonnes")
        except Exception as openpyxl_error:
            logger.warning(f"Échec lecture avec openpyxl: {openpyxl_error}")
            try:
                # Essayer avec xlrd
                test_df = pd.read_excel(completed_file_path, engine='xlrd', nrows=1)
                logger.info(f"Fichier validé avec xlrd: {len(test_df.columns)} colonnes")
            except Exception as xlrd_error:
                logger.error(f"Échec lecture avec xlrd: {xlrd_error}")
                
                # Diagnostic avancé du fichier
                try:
                    import zipfile
                    with zipfile.ZipFile(completed_file_path, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        logger.info(f"Contenu ZIP du fichier Excel: {file_list}")
                except Exception as zip_error:
                    logger.error(f"Fichier n'est pas un ZIP valide: {zip_error}")
                
                # Supprimer le fichier corrompu
                if os.path.exists(completed_file_path):
                    os.remove(completed_file_path)
                
                raise ValueError(f"Fichier Excel corrompu. Erreurs: openpyxl={openpyxl_error}, xlrd={xlrd_error}")
            
    except Exception as save_error:
        logger.error(f"Erreur sauvegarde fichier complété: {save_error}")
        return jsonify({'error': f'Erreur sauvegarde fichier: {save_error}'}), 500
    
    # Traitement
    discrepancies_df = processor.process_completed_file(session_id, completed_file_path)
    distributed_df = processor.distribute_discrepancies(session_id, strategy)
    final_file_path = processor.generate_final_file(session_id)
    
    # Mise à jour de la session
    session_service.update_session(
        session_id,
        completed_file_path=completed_file_path,
        final_file_path=final_file_path
    )
    
    # Calcul des statistiques finales
    total_discrepancy = distributed_df['AJUSTEMENT'].sum()
    adjusted_items = len(distributed_df[distributed_df['AJUSTEMENT'] != 0])
    
    return jsonify({
        'message': 'Traitement terminé avec succès',
        'session_id': session_id,
        'final_url': f'/api/download/final/{session_id}',
        'stats': {
            'total_discrepancy': float(total_discrepancy),
            'adjusted_items': adjusted_items,
            'strategy_used': strategy
        }
    })

@app.route('/api/download/<file_type>/<session_id>', methods=['GET'])
@handle_api_errors('download')
def download_file(file_type, session_id):
    """Télécharge un fichier selon son type"""
    session_data = session_service.get_session_data(session_id)
    if not session_data:
        return jsonify({'error': 'Session non trouvée'}), 404
    
    if file_type == 'template':
        file_path = session_data['template_file_path']
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Template non trouvé'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    elif file_type == 'final':
        file_path = session_data['final_file_path']
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Fichier final non trouvé'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path),
            mimetype='text/csv'
        )
    
    else:
        return jsonify({'error': 'Type de fichier non supporté'}), 400

@app.route('/api/sessions', methods=['GET'])
@handle_api_errors('sessions')
def list_sessions():
    """Liste les sessions actives"""
    sessions = session_service.list_sessions()
    return jsonify({'sessions': sessions})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@handle_api_errors('delete_session')
def delete_session_endpoint(session_id):
    """Supprime une session"""
    success = session_service.delete_session(session_id)
    if success:
        # Nettoyer aussi les fichiers de données
        session_service.cleanup_session_data(session_id)
        return jsonify({'message': 'Session supprimée avec succès'})
    else:
        return jsonify({'error': 'Session non trouvée'}), 404

if __name__ == '__main__':
    # Ce bloc n'est exécuté que lors d'un lancement direct (python app.py)
    # En production, Gunicorn est le point d'entrée et n'exécute pas ce bloc.
    is_debug_mode = os.environ.get('FLASK_ENV') != 'production'
    logger.info(f"Démarrage de l'application en mode {'debug' if is_debug_mode else 'production'}")
    app.run(debug=is_debug_mode, host='0.0.0.0', port=5000)