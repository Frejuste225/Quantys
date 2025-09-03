import os
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from datetime import datetime, date
import uuid
from werkzeug.utils import secure_filename
import logging
from dotenv import load_dotenv
import re

# Charger les variables d'environnement
load_dotenv()

# Imports des services
from services.session_service import SessionService
from services.file_processor import FileProcessorService
from services.file_manager import FileManager
from services.lotecart_processor import LotecartProcessor
from utils.validators import FileValidator
from utils.error_handler import APIErrorHandler, handle_api_errors
from utils.rate_limiter import apply_rate_limit
from database import db_manager

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])


# Configuration améliorée
class Config:
    def __init__(self):
        self.UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
        self.PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER", "processed")
        self.FINAL_FOLDER = os.getenv("FINAL_FOLDER", "final")
        self.ARCHIVE_FOLDER = os.getenv("ARCHIVE_FOLDER", "archive")
        self.LOG_FOLDER = os.getenv("LOG_FOLDER", "logs")
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 16 * 1024 * 1024))
        self.SECRET_KEY = os.getenv("SECRET_KEY", "dev-key-change-in-production")

        # Créer les répertoires
        for folder in [
            self.UPLOAD_FOLDER,
            self.PROCESSED_FOLDER,
            self.FINAL_FOLDER,
            self.ARCHIVE_FOLDER,
            self.LOG_FOLDER,
        ]:
            os.makedirs(folder, exist_ok=True)


config = Config()
app.config.from_object(config)
app.secret_key = config.SECRET_KEY

# Configuration du logging améliorée
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_FOLDER, "inventory_processor.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Initialisation des services
session_service = SessionService()
file_processor = FileProcessorService()
file_manager = FileManager(
    {
        "UPLOAD_FOLDER": config.UPLOAD_FOLDER,
        "PROCESSED_FOLDER": config.PROCESSED_FOLDER,
        "FINAL_FOLDER": config.FINAL_FOLDER,
        "ARCHIVE_FOLDER": config.ARCHIVE_FOLDER,
    }
)


# Classe de compatibilité (pour migration progressive)
class SageX3Processor:
    """
    Classe de compatibilité - utilise maintenant les services
    """

    def __init__(self):
        self.session_service = session_service
        self.file_processor = file_processor
        # Initialiser le processeur LOTECART
        from services.lotecart_processor import LotecartProcessor
        self.lotecart_processor = LotecartProcessor()
        self.lotecart_processor = LotecartProcessor()

    def process_completed_file(self, session_id: str, completed_file_path: str):
        """Traite le fichier Excel complété et calcule les écarts"""
        try:
            # Lire le fichier Excel complété
            completed_df = pd.read_excel(completed_file_path)

            # Validation des colonnes requises
            required_columns = [
                "Code Article",
                "Quantité Théorique",
                "Quantité Réelle",
                "Numéro Lot",
                "Numéro Inventaire",
            ]
            missing_columns = [
                col for col in required_columns if col not in completed_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Colonnes manquantes dans le fichier: {', '.join(missing_columns)}"
                )

            # Nettoyer les numéros de lot dans le fichier complété
            completed_df["Numéro Lot"] = (
                completed_df["Numéro Lot"].fillna("").astype(str).str.strip()
            )
            completed_df.loc[
                completed_df["Numéro Lot"].str.upper().isin(["NAN", "NULL", "NONE"]),
                "Numéro Lot",
            ] = ""

            # Conversion des types
            completed_df["Quantité Théorique"] = pd.to_numeric(
                completed_df["Quantité Théorique"], errors="coerce"
            )
            completed_df["Quantité Réelle"] = pd.to_numeric(
                completed_df["Quantité Réelle"], errors="coerce"
            )

            # Calcul des écarts
            completed_df["Écart"] = (
                completed_df["Quantité Réelle"] - completed_df["Quantité Théorique"]
            )

            # Détection et traitement des lots LOTECART avec le processeur spécialisé
            lotecart_candidates = self.lotecart_processor.detect_lotecart_candidates(completed_df)
            
            # Marquer les lignes LOTECART dans le DataFrame principal
            if not lotecart_candidates.empty:
                lotecart_mask = (completed_df["Quantité Théorique"] == 0) & (
                    completed_df["Quantité Réelle"] > 0
                )
                completed_df.loc[lotecart_mask, "Type_Lot"] = "lotecart"
                
                # Sauvegarder les candidats LOTECART pour traitement ultérieur
                self.session_service.save_dataframe(session_id, "lotecart_candidates", lotecart_candidates)

            # Filtrer les articles avec écarts
            discrepancies_df = completed_df[completed_df["Écart"] != 0].copy()

            # Statistiques
            total_discrepancy = float(discrepancies_df["Écart"].sum())
            adjusted_items_count = len(discrepancies_df)

            # Sauvegarder les résultats dans les services
            self.session_service.save_dataframe(session_id, "completed_df", completed_df)
            self.session_service.save_dataframe(session_id, "discrepancies_df", discrepancies_df)

            # Mettre à jour la session en base
            self.session_service.update_session(
                session_id,
                total_discrepancy=total_discrepancy,
                adjusted_items_count=adjusted_items_count,
            )

            logger.info(
                f"Fichier complété traité pour session {session_id}: {adjusted_items_count} lots avec écarts"
            )
            return discrepancies_df

        except Exception as e:
            logger.error(f"Erreur traitement fichier complété: {e}")
            raise

    def distribute_discrepancies(self, session_id: str, strategy: str = "FIFO"):
        """Distribue les écarts selon la stratégie choisie avec priorité sur les types de lots"""
        try:
            # Charger les données depuis les services
            discrepancies_df = self.session_service.load_dataframe(session_id, "discrepancies_df")
            original_df = self.session_service.load_dataframe(session_id, "original_df")
            
            if discrepancies_df is None or original_df is None:
                raise ValueError("Données de session manquantes pour la distribution")

            # Créer une liste pour stocker les ajustements
            adjustments = []

            for _, discrepancy_row in discrepancies_df.iterrows():
                code_article = discrepancy_row["Code Article"]
                numero_inventaire = discrepancy_row.get("Numéro Inventaire", "")
                ecart = discrepancy_row["Écart"]

                if ecart == 0:
                    continue

                # Vérifier si c'est un cas LOTECART dans les écarts
                is_lotecart = discrepancy_row.get("Type_Lot") == "lotecart"

                if is_lotecart:
                    # Traitement LOTECART avec le processeur spécialisé
                    logger.info(
                        f"🎯 Lot LOTECART détecté pour {code_article} - "
                        f"Quantité théorique: 0, Quantité réelle: {discrepancy_row.get('Quantité Réelle', 0)}"
                    )
                    
                    # Créer un DataFrame temporaire pour ce candidat LOTECART
                    lotecart_candidate = pd.DataFrame([discrepancy_row])
                    
                    # Utiliser le processeur LOTECART pour créer les ajustements
                    lotecart_adjustments = self.lotecart_processor.create_lotecart_adjustments(
                        lotecart_candidate, original_df
                    )
                    
                    # Ajouter les ajustements LOTECART à la liste principale
                    adjustments.extend(lotecart_adjustments)
                    
                    logger.info(f"✅ {len(lotecart_adjustments)} ajustements LOTECART créés pour {code_article}")
                    continue

                # Traitement normal pour les autres types de lots
                # Trouver tous les lots pour cet article et cet inventaire
                if numero_inventaire:
                    article_lots = original_df[
                        (original_df["CODE_ARTICLE"] == code_article)
                        & (original_df["NUMERO_INVENTAIRE"] == numero_inventaire)
                    ].copy()
                else:
                    article_lots = original_df[
                        original_df["CODE_ARTICLE"] == code_article
                    ].copy()

                if article_lots.empty:
                    continue

                article_lots = self._sort_lots_by_priority_and_strategy(
                    article_lots, strategy
                )

                # Distribuer l'écart
                remaining_discrepancy = ecart

                for _, lot_row in article_lots.iterrows():
                    if (
                        abs(remaining_discrepancy) < 0.001
                    ):  # Éviter les erreurs de précision
                        break

                    lot_quantity = float(lot_row["QUANTITE"])
                    lot_number = lot_row["NUMERO_LOT"] if lot_row["NUMERO_LOT"] else ""

                    if remaining_discrepancy > 0:
                        # Écart positif : ajouter du stock
                        adjustment = min(
                            remaining_discrepancy, lot_quantity * 2
                        )  # Limite arbitraire
                    else:
                        # Écart négatif : retirer du stock
                        adjustment = max(remaining_discrepancy, -lot_quantity)

                    if abs(adjustment) > 0.001:
                        adjustments.append(
                            {
                                "CODE_ARTICLE": code_article,
                                "NUMERO_INVENTAIRE": numero_inventaire,
                                "NUMERO_LOT": lot_number,
                                "TYPE_LOT": lot_row.get("Type_Lot", "unknown"),
                                "QUANTITE_ORIGINALE": lot_quantity,
                                "AJUSTEMENT": adjustment,
                                "QUANTITE_CORRIGEE": lot_quantity + adjustment,
                                "Date_Lot": lot_row["Date_Lot"],
                                "original_s_line_raw": lot_row["original_s_line_raw"],
                            }
                        )

                        remaining_discrepancy -= adjustment

            # Convertir en DataFrame
            distributed_df = pd.DataFrame(adjustments)

            # Sauvegarder dans les services
            self.session_service.save_dataframe(session_id, "distributed_df", distributed_df)

            logger.info(
                f"Écarts distribués pour session {session_id} avec stratégie {strategy}: {len(adjustments)} ajustements"
            )
            return distributed_df

        except Exception as e:
            logger.error(f"Erreur distribution écarts: {e}")
            raise

    def _sort_lots_by_priority_and_strategy(
        self, lots_df: pd.DataFrame, strategy: str
    ) -> pd.DataFrame:
        """Trie les lots selon la priorité des types et la stratégie FIFO/LIFO"""
        # Définir l'ordre de priorité des types de lots (simplifié)
        type_priority = {"type1": 1, "type2": 2, "lotecart": 3, "unknown": 4}

        # Ajouter une colonne de priorité
        lots_df["priority"] = (
            lots_df.get("Type_Lot", "unknown").map(type_priority).fillna(4)
        )

        # Trier d'abord par priorité de type, puis par date selon la stratégie
        if strategy == "FIFO":
            # Type prioritaire d'abord, puis plus anciens d'abord
            sorted_lots = lots_df.sort_values(
                ["priority", "Date_Lot"], na_position="last"
            )
        else:  # LIFO
            # Type prioritaire d'abord, puis plus récents d'abord
            sorted_lots = lots_df.sort_values(
                ["priority", "Date_Lot"], ascending=[True, False], na_position="last"
            )

        # Pour les lots LOTECART, on ignore la date et on prend le premier disponible
        lotecart_lots = sorted_lots[sorted_lots.get("Type_Lot", "") == "lotecart"]
        other_lots = sorted_lots[sorted_lots.get("Type_Lot", "") != "lotecart"]

        # Recombiner : autres lots triés + lots LOTECART en premier disponible
        if not lotecart_lots.empty:
            result = pd.concat([other_lots, lotecart_lots], ignore_index=True)
        else:
            result = other_lots

        return result.drop("priority", axis=1, errors="ignore")

    def generate_final_file(self, session_id: str):
        """Génère le fichier CSV final au format Sage X3 avec TOUTES les lignes originales"""
        try:
            # Charger les données depuis les services
            distributed_df = self.session_service.load_dataframe(session_id, "distributed_df")
            original_df = self.session_service.load_dataframe(session_id, "original_df")
            completed_df = self.session_service.load_dataframe(session_id, "completed_df")
            
            if distributed_df is None or original_df is None or completed_df is None:
                raise ValueError("Données manquantes pour générer le fichier final")

            # Récupérer les données de session depuis la base
            db_session_data = self.session_service.get_session_data(session_id)
            if not db_session_data:
                raise ValueError("Session non trouvée en base")
            
            # Récupérer les header_lines depuis la base
            import json
            header_lines = json.loads(db_session_data["header_lines"]) if db_session_data["header_lines"] else []

            # Construire le nom du fichier
            original_filename = db_session_data["original_filename"]
            base_name = os.path.splitext(original_filename)[0]
            final_filename = f"{base_name}_corrige_{session_id}.csv"
            final_file_path = os.path.join(config.FINAL_FOLDER, final_filename)

            # Créer un dictionnaire des quantités réelles depuis le template complété
            # Clé: (CODE_ARTICLE, NUMERO_INVENTAIRE, NUMERO_LOT)
            real_quantities_dict = {}
            for _, row in completed_df.iterrows():
                code_article = row["Code Article"]
                numero_inventaire = row["Numéro Inventaire"]
                numero_lot = str(row["Numéro Lot"]).strip() if pd.notna(row["Numéro Lot"]) else ""
                quantite_reelle = row["Quantité Réelle"]
                
                key = (code_article, numero_inventaire, numero_lot)
                real_quantities_dict[key] = quantite_reelle
            
            # Créer un dictionnaire des ajustements pour un accès rapide
            adjustments_dict = {}
            for _, row in distributed_df.iterrows():
                code_article = row["CODE_ARTICLE"]
                numero_inventaire = row["NUMERO_INVENTAIRE"]
                numero_lot = (
                    str(row["NUMERO_LOT"]).strip()
                    if pd.notna(row["NUMERO_LOT"])
                    else ""
                )

                key = (code_article, numero_inventaire, numero_lot)
                adjustments_dict[key] = {
                    "QUANTITE_CORRIGEE": row["QUANTITE_CORRIGEE"],
                    "TYPE_LOT": row["TYPE_LOT"],
                    "AJUSTEMENT": row["AJUSTEMENT"],
                    "IS_NEW_LOTECART": pd.isna(row.get("original_s_line_raw")) or row.get("original_s_line_raw") is None
                }

            # Générer le contenu du fichier
            lines = []

            # Ajouter les en-têtes E et L
            lines.extend(header_lines)

            # Traiter TOUTES les lignes originales
            lines_processed = 0
            lines_adjusted = 0

            for _, original_row in original_df.iterrows():
                if pd.notna(original_row["original_s_line_raw"]):
                    original_line = str(original_row["original_s_line_raw"])
                    parts = original_line.split(";")

                    if len(parts) >= 6:  # S'assurer qu'on a assez de colonnes
                        # Créer la clé pour chercher un ajustement
                        code_article = original_row["CODE_ARTICLE"]
                        numero_inventaire = original_row["NUMERO_INVENTAIRE"]
                        numero_lot = (
                            str(original_row["NUMERO_LOT"]).strip()
                            if pd.notna(original_row["NUMERO_LOT"])
                            else ""
                        )

                        key = (code_article, numero_inventaire, numero_lot)
                        
                        # Récupérer la quantité réelle depuis le template complété
                        quantite_reelle = real_quantities_dict.get(key, 0)

                        # 🎯 LOGIQUE AMÉLIORÉE : Quantités théoriques ajustées + quantités réelles saisies
                        
                        # 1. TOUJOURS mettre à jour la quantité réelle depuis le template complété
                        if quantite_reelle is not None:
                            parts[6] = str(int(quantite_reelle))  # Colonne 6 = Quantité réelle saisie
                        
                        # 2. Vérifier s'il y a un ajustement pour cette ligne
                        if key in adjustments_dict:
                            adjustment = adjustments_dict[key]
                            
                            # 3. Appliquer l'ajustement sur la quantité théorique (colonne 5)
                            if adjustment["TYPE_LOT"] == "lotecart":
                                # Pour LOTECART : quantité théorique = quantité réelle (pas d'écart)
                                parts[5] = str(int(quantite_reelle))
                                logger.debug(f"🏷️ LOTECART {code_article}: Qté théo = Qté réelle = {quantite_reelle}")
                            else:
                                # Pour ajustements normaux : quantité théorique ajustée
                                parts[5] = str(int(adjustment["QUANTITE_CORRIGEE"]))
                                logger.debug(f"⚖️ Ajustement {code_article}: Qté théo ajustée = {adjustment['QUANTITE_CORRIGEE']}, Qté réelle = {quantite_reelle}")

                            # 4. S'assurer que le numéro de lot est correct
                            if len(parts) > 14:
                                if adjustment["TYPE_LOT"] == "lotecart" or numero_lot == "LOTECART":
                                    parts[14] = "LOTECART"
                                else:
                                    parts[14] = numero_lot

                            lines_adjusted += 1
                            
                        else:
                            # 5. Pour les lignes NON ajustées : garder quantité théorique originale
                            # La quantité réelle a déjà été mise à jour ci-dessus
                            quantite_theo_originale = parts[5]
                            logger.debug(f"📋 Ligne standard {code_article}: Qté théo originale = {quantite_theo_originale}, Qté réelle saisie = {quantite_reelle}")
                        
                        # 6. Log de vérification pour traçabilité
                        final_qte_theo = parts[5]
                        final_qte_reelle = parts[6]
                        ecart_final = float(final_qte_reelle) - float(final_qte_theo) if final_qte_theo and final_qte_reelle else 0
                        
                        if abs(ecart_final) > 0.001:  # Il y a encore un écart
                            logger.debug(f"📊 {code_article} - Écart final: {ecart_final} (Théo: {final_qte_theo}, Réel: {final_qte_reelle})")
                        else:
                            logger.debug(f"✅ {code_article} - Pas d'écart (Théo: {final_qte_theo}, Réel: {final_qte_reelle})")

                        # Vérifier si la quantité finale est nulle et mettre INDICATEUR_COMPTE à 2
                        quantite_finale = float(parts[5]) if parts[5] else 0
                        quantite_theorique_originale = float(
                            original_row.get("QUANTITE", 0)
                        )
                        quantite_reelle_finale = float(parts[6]) if parts[6] else 0

                        # Mettre INDICATEUR_COMPTE à 2 dans les cas suivants :
                        # 1. La quantité théorique finale est 0 ET quantité réelle > 0 (LOTECART)
                        # 2. La quantité théorique originale était 0 (cas LOTECART détecté)
                        # 3. Les quantités théorique et réelle sont égales (pas d'écart)
                        if (
                            (quantite_theorique_originale == 0 and quantite_reelle_finale > 0) or
                            (quantite_finale == quantite_reelle_finale and quantite_reelle_finale > 0) or
                            numero_lot == "LOTECART"
                        ) and len(parts) > 7:
                            parts[7] = "2"  # INDICATEUR_COMPTE à l'index 7
                            logger.debug(
                                f"INDICATEUR_COMPTE mis à 2 pour {code_article} - {numero_lot} (qté théo finale: {quantite_finale}, qté réelle: {quantite_reelle_finale})"
                            )

                        # Ajouter la ligne (ajustée ou originale)
                        corrected_line = ";".join(parts)
                        lines.append(corrected_line)
                        lines_processed += 1

            # Générer les nouvelles lignes LOTECART avec le processeur spécialisé
            max_line_number = 0
            if original_df is not None and not original_df.empty:
                # Extraire les numéros de ligne existants pour éviter les doublons
                line_numbers = []
                for _, row in original_df.iterrows():
                    line_raw = str(row.get("original_s_line_raw", ""))
                    parts = line_raw.split(";")
                    if len(parts) > 3:
                        try:
                            line_num = int(parts[3])
                            line_numbers.append(line_num)
                        except (ValueError, IndexError):
                            pass
                max_line_number = max(line_numbers) if line_numbers else 0

            # Filtrer les ajustements LOTECART qui nécessitent de nouvelles lignes
            lotecart_adjustments = [
                adj for _, adj in distributed_df.iterrows()
                if (adj.get("TYPE_LOT") == "lotecart" and 
                    (pd.isna(adj.get("original_s_line_raw")) or adj.get("original_s_line_raw") is None))
            ]
            
            # Convertir en format attendu par le processeur LOTECART
            lotecart_adjustments_dict = []
            for adj in lotecart_adjustments:
                # Récupérer la quantité réelle depuis le template complété
                key = (adj["CODE_ARTICLE"], adj["NUMERO_INVENTAIRE"], "LOTECART")
                quantite_reelle = real_quantities_dict.get(key, adj["QUANTITE_CORRIGEE"])
                
                lotecart_adjustments_dict.append({
                    "CODE_ARTICLE": adj["CODE_ARTICLE"],
                    "NUMERO_INVENTAIRE": adj["NUMERO_INVENTAIRE"],
                    "NUMERO_LOT": "LOTECART",
                    "TYPE_LOT": "lotecart",
                    "QUANTITE_CORRIGEE": adj["QUANTITE_CORRIGEE"],
                    "QUANTITE_REELLE": quantite_reelle,  # Ajouter la quantité réelle
                    "reference_line": adj.get("reference_line"),
                    "is_new_lotecart": True
                })

            # Générer les nouvelles lignes LOTECART
            new_lotecart_lines = self.lotecart_processor.generate_lotecart_lines(
                lotecart_adjustments_dict, max_line_number
            )
            
            # Ajouter les nouvelles lignes au fichier
            lines.extend(new_lotecart_lines)
            lotecart_lines_created = len(new_lotecart_lines)
            
            logger.info(f"🎯 {lotecart_lines_created} nouvelles lignes LOTECART ajoutées au fichier final")

            # Écrire le fichier
            with open(final_file_path, "w", encoding="utf-8", newline="") as f:
                for line in lines:
                    f.write(line + "\n")

            # Mettre à jour la session
            self.session_service.update_session(
                session_id, final_file_path=final_file_path
            )

            logger.info(f"Fichier final généré: {final_file_path}")
            logger.info(
                f"Total lignes traitées: {lines_processed}, Lignes ajustées: {lines_adjusted}, Nouvelles lignes LOTECART: {lotecart_lines_created}"
            )
            
            # Vérification finale avec le processeur LOTECART
            expected_lotecart_count = lotecart_lines_created
            validation_result = self.lotecart_processor.validate_lotecart_processing(
                final_file_path, expected_lotecart_count
            )
            
            if validation_result["success"]:
                logger.info("✅ Validation LOTECART réussie")
            else:
                logger.warning(f"⚠️ Problèmes détectés lors de la validation LOTECART: {validation_result['issues']}")
            
            # Vérification finale générale avec résumé détaillé
            summary = self._verify_final_file_with_summary(final_file_path)
            logger.info(f"📊 Résumé du fichier final: {summary}")
            
            # Vérification spécifique des quantités théoriques ajustées vs quantités réelles
            quantities_verification = self._verify_quantities_consistency(
                final_file_path, completed_df, distributed_df
            )
            logger.info(f"🔍 Vérification quantités: {quantities_verification}")
            
            return final_file_path

        except Exception as e:
            logger.error(f"Erreur génération fichier final: {e}")
            raise
    
    def _verify_final_file_with_summary(self, final_file_path: str) -> dict:
        """Vérifie le contenu du fichier final généré et retourne un résumé détaillé"""
        summary = {
            "success": True,
            "file_path": final_file_path,
            "verification_timestamp": datetime.now().isoformat(),
            "structure": {
                "total_lines": 0,
                "header_lines_e": 0,
                "header_lines_l": 0,
                "data_lines_s": 0,
                "invalid_lines": 0
            },
            "quantities": {
                "total_theoretical": 0.0,
                "total_real": 0.0,
                "total_discrepancy": 0.0,
                "zero_theoretical_count": 0,
                "zero_real_count": 0,
                "negative_quantities": 0
            },
            "lotecart": {
                "total_lines": 0,
                "correct_indicators": 0,
                "incorrect_indicators": 0,
                "sample_articles": []
            },
            "adjustments": {
                "lines_with_adjustments": 0,
                "lines_with_indicator_2": 0,
                "articles_adjusted": set(),
                "adjustment_summary": {}
            },
            "quality": {
                "lines_with_missing_data": 0,
                "lines_with_invalid_quantities": 0,
                "duplicate_line_numbers": [],
                "inconsistent_inventories": []
            },
            "sample_data": {
                "first_10_s_lines": [],
                "lotecart_samples": [],
                "adjustment_samples": []
            },
            "issues": [],
            "warnings": []
        }
        
        try:
            if not os.path.exists(final_file_path):
                summary["success"] = False
                summary["issues"].append("Fichier final non trouvé")
                return summary
            
            # Dictionnaires pour tracking
            line_numbers_seen = set()
            inventories_seen = set()
            articles_by_inventory = {}
            
            with open(final_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    summary["structure"]["total_lines"] += 1
                    
                    if not line:
                        continue
                    
                    # Analyser selon le type de ligne
                    if line.startswith('E;'):
                        summary["structure"]["header_lines_e"] += 1
                        
                    elif line.startswith('L;'):
                        summary["structure"]["header_lines_l"] += 1
                        
                    elif line.startswith('S;'):
                        self._analyze_s_line(line, line_num, summary, line_numbers_seen, 
                                           inventories_seen, articles_by_inventory)
                        
                    else:
                        summary["structure"]["invalid_lines"] += 1
                        summary["warnings"].append(f"Ligne {line_num}: Format non reconnu")
            
            # Post-traitement et calculs finaux
            self._finalize_verification_summary(summary, articles_by_inventory)
            
            # Validation finale
            self._validate_file_integrity(summary)
            
            logger.info(f"✅ Vérification fichier final terminée: {summary['structure']['data_lines_s']} lignes S, "
                       f"{summary['lotecart']['total_lines']} LOTECART, "
                       f"{summary['adjustments']['lines_with_adjustments']} ajustements")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification fichier final: {e}", exc_info=True)
            summary["success"] = False
            summary["issues"].append(f"Erreur de vérification: {str(e)}")
            return summary
    
    def _analyze_s_line(self, line: str, line_num: int, summary: dict, 
                       line_numbers_seen: set, inventories_seen: set, 
                       articles_by_inventory: dict):
        """Analyse une ligne S; en détail"""
        try:
            parts = line.split(';')
            summary["structure"]["data_lines_s"] += 1
            
            # Vérifier la structure minimale
            if len(parts) < 15:
                summary["quality"]["lines_with_missing_data"] += 1
                summary["warnings"].append(f"Ligne {line_num}: Structure incomplète ({len(parts)} colonnes)")
                return
            
            # Extraire les données principales
            numero_session = parts[1]
            numero_inventaire = parts[2]
            rang = parts[3]
            site = parts[4]
            qte_theo_str = parts[5]
            qte_reelle_str = parts[6]
            indicateur = parts[7]
            code_article = parts[8]
            emplacement = parts[9]
            statut = parts[10]
            unite = parts[11]
            valeur_str = parts[12]
            zone_pk = parts[13]
            numero_lot = parts[14]
            
            # Validation et conversion des quantités
            try:
                qte_theo = float(qte_theo_str) if qte_theo_str else 0.0
                qte_reelle = float(qte_reelle_str) if qte_reelle_str else 0.0
            except ValueError:
                summary["quality"]["lines_with_invalid_quantities"] += 1
                summary["warnings"].append(f"Ligne {line_num}: Quantités invalides")
                return
            
            # Vérifier les quantités négatives
            if qte_theo < 0 or qte_reelle < 0:
                summary["quantities"]["negative_quantities"] += 1
                summary["warnings"].append(f"Ligne {line_num}: Quantités négatives détectées")
            
            # Accumuler les totaux
            summary["quantities"]["total_theoretical"] += qte_theo
            summary["quantities"]["total_real"] += qte_reelle
            summary["quantities"]["total_discrepancy"] += (qte_reelle - qte_theo)
            
            # Compter les zéros
            if qte_theo == 0:
                summary["quantities"]["zero_theoretical_count"] += 1
            if qte_reelle == 0:
                summary["quantities"]["zero_real_count"] += 1
            
            # Analyser les LOTECART
            if numero_lot == "LOTECART":
                summary["lotecart"]["total_lines"] += 1
                
                if indicateur == "2":
                    summary["lotecart"]["correct_indicators"] += 1
                else:
                    summary["lotecart"]["incorrect_indicators"] += 1
                    summary["warnings"].append(f"Ligne {line_num}: LOTECART avec indicateur incorrect ({indicateur})")
                
                # Échantillon LOTECART
                if len(summary["lotecart"]["sample_articles"]) < 5:
                    summary["lotecart"]["sample_articles"].append({
                        "line_number": line_num,
                        "article": code_article,
                        "inventory": numero_inventaire,
                        "quantity_theo": qte_theo,
                        "quantity_real": qte_reelle,
                        "indicator": indicateur
                    })
            
            # Analyser les ajustements (indicateur = 2)
            if indicateur == "2":
                summary["adjustments"]["lines_with_indicator_2"] += 1
                
                # Détecter si c'est un ajustement (écart non nul)
                if abs(qte_reelle - qte_theo) > 0.001:
                    summary["adjustments"]["lines_with_adjustments"] += 1
                    summary["adjustments"]["articles_adjusted"].add(code_article)
                    
                    # Résumé par article
                    if code_article not in summary["adjustments"]["adjustment_summary"]:
                        summary["adjustments"]["adjustment_summary"][code_article] = {
                            "total_discrepancy": 0,
                            "lines_count": 0,
                            "inventories": set()
                        }
                    
                    summary["adjustments"]["adjustment_summary"][code_article]["total_discrepancy"] += (qte_reelle - qte_theo)
                    summary["adjustments"]["adjustment_summary"][code_article]["lines_count"] += 1
                    summary["adjustments"]["adjustment_summary"][code_article]["inventories"].add(numero_inventaire)
            
            # Vérifier les doublons de numéros de ligne
            try:
                rang_int = int(rang)
                if rang_int in line_numbers_seen:
                    summary["quality"]["duplicate_line_numbers"].append(rang_int)
                else:
                    line_numbers_seen.add(rang_int)
            except ValueError:
                summary["warnings"].append(f"Ligne {line_num}: Numéro de rang invalide ({rang})")
            
            # Tracking des inventaires
            inventories_seen.add(numero_inventaire)
            if numero_inventaire not in articles_by_inventory:
                articles_by_inventory[numero_inventaire] = set()
            articles_by_inventory[numero_inventaire].add(code_article)
            
            # Échantillons pour les 10 premières lignes S
            if len(summary["sample_data"]["first_10_s_lines"]) < 10:
                summary["sample_data"]["first_10_s_lines"].append({
                    "line_number": line_num,
                    "article": code_article,
                    "inventory": numero_inventaire,
                    "lot": numero_lot,
                    "qty_theo": qte_theo,
                    "qty_real": qte_reelle,
                    "indicator": indicateur,
                    "site": site
                })
            
            # Échantillons d'ajustements
            if (abs(qte_reelle - qte_theo) > 0.001 and 
                len(summary["sample_data"]["adjustment_samples"]) < 5):
                summary["sample_data"]["adjustment_samples"].append({
                    "line_number": line_num,
                    "article": code_article,
                    "inventory": numero_inventaire,
                    "lot": numero_lot,
                    "qty_theo": qte_theo,
                    "qty_real": qte_reelle,
                    "discrepancy": qte_reelle - qte_theo,
                    "indicator": indicateur
                })
                
        except Exception as e:
            summary["warnings"].append(f"Ligne {line_num}: Erreur d'analyse - {str(e)}")
    
    def _finalize_verification_summary(self, summary: dict, articles_by_inventory: dict):
        """Finalise le résumé de vérification avec des calculs additionnels"""
        try:
            # Convertir les sets en listes pour la sérialisation JSON
            summary["adjustments"]["articles_adjusted"] = list(summary["adjustments"]["articles_adjusted"])
            
            # Finaliser les résumés d'ajustements
            for article, adj_data in summary["adjustments"]["adjustment_summary"].items():
                adj_data["inventories"] = list(adj_data["inventories"])
            
            # Statistiques par inventaire
            summary["inventories"] = {}
            for inventory, articles in articles_by_inventory.items():
                summary["inventories"][inventory] = {
                    "articles_count": len(articles),
                    "articles_list": list(articles)[:10]  # Limiter à 10 pour éviter la surcharge
                }
            
            # Calculs de pourcentages
            total_s_lines = summary["structure"]["data_lines_s"]
            if total_s_lines > 0:
                summary["statistics"] = {
                    "lotecart_percentage": round((summary["lotecart"]["total_lines"] / total_s_lines) * 100, 2),
                    "adjustment_percentage": round((summary["adjustments"]["lines_with_adjustments"] / total_s_lines) * 100, 2),
                    "zero_theo_percentage": round((summary["quantities"]["zero_theoretical_count"] / total_s_lines) * 100, 2),
                    "indicator_2_percentage": round((summary["adjustments"]["lines_with_indicator_2"] / total_s_lines) * 100, 2)
                }
            
            # Résumé global
            summary["global_summary"] = {
                "total_inventories": len(articles_by_inventory),
                "total_articles_unique": len(set().union(*[articles for articles in articles_by_inventory.values()])),
                "total_discrepancy_value": round(summary["quantities"]["total_discrepancy"], 2),
                "has_lotecart": summary["lotecart"]["total_lines"] > 0,
                "has_adjustments": summary["adjustments"]["lines_with_adjustments"] > 0
            }
            
        except Exception as e:
            summary["warnings"].append(f"Erreur finalisation résumé: {str(e)}")
    
    def _validate_file_integrity(self, summary: dict):
        """Valide l'intégrité globale du fichier"""
        try:
            # Vérifications critiques
            if summary["structure"]["data_lines_s"] == 0:
                summary["issues"].append("Aucune ligne de données S; trouvée")
                summary["success"] = False
            
            if summary["structure"]["header_lines_e"] == 0:
                summary["warnings"].append("Aucune ligne d'en-tête E; trouvée")
            
            if summary["structure"]["header_lines_l"] == 0:
                summary["warnings"].append("Aucune ligne d'inventaire L; trouvée")
            
            # Vérifications LOTECART
            if summary["lotecart"]["total_lines"] > 0:
                if summary["lotecart"]["incorrect_indicators"] > 0:
                    summary["issues"].append(
                        f"{summary['lotecart']['incorrect_indicators']} lignes LOTECART avec indicateurs incorrects"
                    )
            
            # Vérifications de cohérence
            if len(summary["quality"]["duplicate_line_numbers"]) > 0:
                summary["issues"].append(
                    f"Numéros de ligne dupliqués: {summary['quality']['duplicate_line_numbers'][:10]}"
                )
            
            # Vérifications des quantités
            if summary["quantities"]["negative_quantities"] > 0:
                summary["warnings"].append(
                    f"{summary['quantities']['negative_quantities']} lignes avec quantités négatives"
                )
            
            # Score de qualité (0-100)
            quality_score = 100
            quality_score -= min(len(summary["issues"]) * 20, 80)  # -20 par issue critique
            quality_score -= min(len(summary["warnings"]) * 5, 20)  # -5 par warning
            quality_score -= min(summary["quality"]["lines_with_invalid_quantities"] * 2, 10)
            
            summary["quality_score"] = max(quality_score, 0)
            
            # Déterminer le succès global
            if len(summary["issues"]) > 0:
                summary["success"] = False
            
        except Exception as e:
            summary["issues"].append(f"Erreur validation intégrité: {str(e)}")
            summary["success"] = False
    
    def _verify_quantities_consistency(self, final_file_path: str, completed_df: pd.DataFrame, distributed_df: pd.DataFrame) -> dict:
        """Vérifie la cohérence entre quantités théoriques ajustées et quantités réelles"""
        verification = {
            "success": True,
            "total_lines_checked": 0,
            "consistent_lines": 0,
            "inconsistent_lines": 0,
            "lotecart_lines": 0,
            "adjusted_lines": 0,
            "standard_lines": 0,
            "issues": [],
            "samples": {
                "consistent": [],
                "inconsistent": [],
                "lotecart": [],
                "adjusted": []
            }
        }
        
        try:
            # Créer les dictionnaires de référence
            completed_dict = {}
            for _, row in completed_df.iterrows():
                key = (row["Code Article"], row["Numéro Inventaire"], str(row["Numéro Lot"]).strip())
                completed_dict[key] = {
                    "qte_theo_originale": row["Quantité Théorique"],
                    "qte_reelle_saisie": row["Quantité Réelle"]
                }
            
            adjustments_dict = {}
            for _, row in distributed_df.iterrows():
                key = (row["CODE_ARTICLE"], row["NUMERO_INVENTAIRE"], str(row["NUMERO_LOT"]).strip())
                adjustments_dict[key] = {
                    "qte_theo_ajustee": row["QUANTITE_CORRIGEE"],
                    "type_lot": row["TYPE_LOT"],
                    "ajustement": row["AJUSTEMENT"]
                }
            
            # Analyser le fichier final
            with open(final_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('S;'):
                        parts = line.strip().split(';')
                        if len(parts) >= 15:
                            verification["total_lines_checked"] += 1
                            
                            # Extraire les données
                            code_article = parts[8]
                            numero_inventaire = parts[2]
                            numero_lot = parts[14].strip()
                            qte_theo_finale = float(parts[5]) if parts[5] else 0
                            qte_reelle_finale = float(parts[6]) if parts[6] else 0
                            
                            key = (code_article, numero_inventaire, numero_lot)
                            
                            # Récupérer les données de référence
                            completed_data = completed_dict.get(key, {})
                            adjustment_data = adjustments_dict.get(key, {})
                            
                            # Déterminer le type de ligne
                            line_type = "standard"
                            expected_qte_theo = completed_data.get("qte_theo_originale", 0)
                            expected_qte_reelle = completed_data.get("qte_reelle_saisie", 0)
                            
                            if numero_lot == "LOTECART":
                                line_type = "lotecart"
                                verification["lotecart_lines"] += 1
                                # Pour LOTECART : qté théo = qté réelle
                                expected_qte_theo = expected_qte_reelle
                            elif key in adjustments_dict:
                                line_type = "adjusted"
                                verification["adjusted_lines"] += 1
                                # Pour ajustements : qté théo ajustée
                                expected_qte_theo = adjustment_data["qte_theo_ajustee"]
                            else:
                                verification["standard_lines"] += 1
                            
                            # Vérifier la cohérence
                            theo_ok = abs(qte_theo_finale - expected_qte_theo) < 0.001
                            reelle_ok = abs(qte_reelle_finale - expected_qte_reelle) < 0.001
                            
                            if theo_ok and reelle_ok:
                                verification["consistent_lines"] += 1
                                
                                # Échantillon de lignes cohérentes
                                if len(verification["samples"]["consistent"]) < 3:
                                    verification["samples"]["consistent"].append({
                                        "line": line_num,
                                        "article": code_article,
                                        "lot": numero_lot,
                                        "type": line_type,
                                        "qte_theo": qte_theo_finale,
                                        "qte_reelle": qte_reelle_finale
                                    })
                            else:
                                verification["inconsistent_lines"] += 1
                                
                                # Détail de l'incohérence
                                issue = {
                                    "line": line_num,
                                    "article": code_article,
                                    "lot": numero_lot,
                                    "type": line_type,
                                    "qte_theo_attendue": expected_qte_theo,
                                    "qte_theo_trouvee": qte_theo_finale,
                                    "qte_reelle_attendue": expected_qte_reelle,
                                    "qte_reelle_trouvee": qte_reelle_finale,
                                    "theo_ok": theo_ok,
                                    "reelle_ok": reelle_ok
                                }
                                
                                verification["issues"].append(
                                    f"Ligne {line_num} ({code_article}): "
                                    f"Théo attendue={expected_qte_theo}, trouvée={qte_theo_finale}, "
                                    f"Réelle attendue={expected_qte_reelle}, trouvée={qte_reelle_finale}"
                                )
                                
                                # Échantillon d'incohérences
                                if len(verification["samples"]["inconsistent"]) < 5:
                                    verification["samples"]["inconsistent"].append(issue)
                            
                            # Échantillons par type
                            if line_type == "lotecart" and len(verification["samples"]["lotecart"]) < 3:
                                verification["samples"]["lotecart"].append({
                                    "line": line_num,
                                    "article": code_article,
                                    "qte_theo": qte_theo_finale,
                                    "qte_reelle": qte_reelle_finale,
                                    "consistent": theo_ok and reelle_ok
                                })
                            elif line_type == "adjusted" and len(verification["samples"]["adjusted"]) < 3:
                                verification["samples"]["adjusted"].append({
                                    "line": line_num,
                                    "article": code_article,
                                    "lot": numero_lot,
                                    "qte_theo_originale": completed_data.get("qte_theo_originale", 0),
                                    "qte_theo_ajustee": qte_theo_finale,
                                    "qte_reelle": qte_reelle_finale,
                                    "ajustement": adjustment_data.get("ajustement", 0),
                                    "consistent": theo_ok and reelle_ok
                                })
            
            # Calculs finaux
            if verification["total_lines_checked"] > 0:
                consistency_rate = (verification["consistent_lines"] / verification["total_lines_checked"]) * 100
                verification["consistency_rate"] = round(consistency_rate, 2)
                
                if consistency_rate < 95:
                    verification["success"] = False
                    verification["issues"].insert(0, f"Taux de cohérence trop faible: {consistency_rate}%")
            
            # Résumé
            verification["summary"] = {
                "total_checked": verification["total_lines_checked"],
                "consistent": verification["consistent_lines"],
                "inconsistent": verification["inconsistent_lines"],
                "consistency_rate": verification.get("consistency_rate", 0),
                "lotecart_count": verification["lotecart_lines"],
                "adjusted_count": verification["adjusted_lines"],
                "standard_count": verification["standard_lines"]
            }
            
            logger.info(
                f"✅ Vérification quantités terminée: {verification['consistent_lines']}/{verification['total_lines_checked']} "
                f"lignes cohérentes ({verification.get('consistency_rate', 0)}%)"
            )
            
            return verification
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification quantités: {e}", exc_info=True)
            verification["success"] = False
            verification["issues"].append(f"Erreur de vérification: {str(e)}")
            return verification


# Initialisation du processeur
processor = SageX3Processor()


# Endpoints API
@app.route("/api/upload", methods=["POST"])
@apply_rate_limit("upload")
@handle_api_errors("file_upload")
def upload_file():
    """Endpoint amélioré pour l'upload initial d'un fichier Sage X3"""
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Nom de fichier vide"}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".csv", ".xlsx", ".xls"]:
        return (
            jsonify({"error": "Format non supporté. Seuls CSV et XLSX sont acceptés"}),
            400,
        )

    # Validation sécurisée du fichier
    is_valid, error_msg = FileValidator.validate_file_security(
        file, config.MAX_FILE_SIZE
    )
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    session_creation_timestamp = datetime.now()
    filepath = None

    try:
        # Créer la session en base de données
        session_id = session_service.create_session(
            original_filename=file.filename,
            original_file_path="",  # Sera mis à jour après sauvegarde
            status="uploading",
        )

        filename_on_disk = secure_filename(f"{session_id}_{file.filename}")
        filepath = os.path.join(config.UPLOAD_FOLDER, filename_on_disk)
        file.save(filepath)

        # Mettre à jour le chemin du fichier
        session_service.update_session(session_id, original_file_path=filepath)

        # Traitement du fichier
        is_valid, result_data, headers, inventory_date = (
            file_processor.validate_and_process_sage_file(
                filepath, file_extension, session_creation_timestamp
            )
        )

        if not is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            session_service.delete_session(session_id)
            return jsonify({"error": str(result_data)}), 400

        original_df = result_data

        # Agrégation
        aggregated_df = file_processor.aggregate_data(original_df)

        # Génération du template
        template_file_path = file_processor.generate_template(
            aggregated_df, session_id, config.PROCESSED_FOLDER
        )

        # Mise à jour de la session
        session_service.update_session(
            session_id,
            template_file_path=template_file_path,
            status="template_generated",
            inventory_date=inventory_date,
            nb_articles=len(aggregated_df),
            nb_lots=len(original_df),
            total_quantity=float(aggregated_df["Quantite_Theorique_Totale"].sum()),
            header_lines=json.dumps(headers),
        )

        # Sauvegarder les DataFrames de manière persistante
        session_service.save_dataframe(session_id, "original_df", original_df)
        session_service.save_dataframe(session_id, "aggregated_df", aggregated_df)

        # Sauvegarder aussi dans le stockage persistant
        session_service.save_dataframe(session_id, "original_df", original_df)
        session_service.save_dataframe(session_id, "aggregated_df", aggregated_df)

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "template_url": f"/api/download/template/{session_id}",
                "stats": {
                    "nb_articles": len(aggregated_df),
                    "total_quantity": float(
                        aggregated_df["Quantite_Theorique_Totale"].sum()
                    ),
                    "nb_lots": len(original_df),
                    "inventory_date": (
                        inventory_date.isoformat() if inventory_date else None
                    ),
                },
            }
        )

    except Exception as e:
        logger.error(f"Erreur upload: {e}", exc_info=True)
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/process", methods=["POST"])
@apply_rate_limit("upload")
@handle_api_errors("file_processing")
def process_completed_file_route():
    """Endpoint amélioré pour traiter le fichier complété"""
    if "file" not in request.files or "session_id" not in request.form:
        return jsonify({"error": "Paramètres manquants"}), 400

    try:
        session_id = request.form["session_id"]
        file = request.files["file"]
        strategy = request.form.get("strategy", "FIFO")

        # Vérifier que la session existe
        session_data = session_service.get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Session non trouvée"}), 404

        if not file.filename.lower().endswith((".xlsx", ".xls")):
            return jsonify({"error": "Seuls les fichiers Excel sont acceptés"}), 400

        # Validation du fichier complété
        temp_filepath = os.path.join(
            config.PROCESSED_FOLDER, f"temp_{session_id}_{file.filename}"
        )
        file.save(temp_filepath)

        is_valid, validation_msg, errors = file_processor.validate_completed_template(
            temp_filepath
        )
        if not is_valid:
            os.remove(temp_filepath)
            return jsonify({"error": validation_msg, "details": errors}), 400

        filename_on_disk = secure_filename(f"completed_{session_id}_{file.filename}")
        filepath = os.path.join(config.PROCESSED_FOLDER, filename_on_disk)
        os.rename(temp_filepath, filepath)

        # Traitement (utilise encore l'ancienne méthode pour compatibilité)
        processed_summary_df = processor.process_completed_file(session_id, filepath)
        distributed_summary_df = processor.distribute_discrepancies(
            session_id, strategy
        )
        final_file_path = processor.generate_final_file(session_id)

        # Mise à jour de la session en base
        session_service.update_session(
            session_id,
            completed_file_path=filepath,
            final_file_path=final_file_path,
            status="completed",
            strategy_used=strategy,
        )

        # Récupérer les données depuis les services
        session_data = session_service.get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Session non trouvée"}), 404

        return jsonify(
            {
                "success": True,
                "final_url": f"/api/download/final/{session_id}",
                "stats": {
                    "total_discrepancy": session_data.get("total_discrepancy", 0),
                    "adjusted_items": session_data.get("adjusted_items_count", 0),
                    "strategy_used": session_data.get("strategy_used", "N/A"),
                },
            }
        )

    except ValueError as e:
        logger.error(f"Erreur validation: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur traitement: {e}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/download/<file_type>/<session_id>", methods=["GET"])
@handle_api_errors("file_download")
def download_file(file_type: str, session_id: str):
    """Endpoint de téléchargement amélioré"""
    try:
        session_data = session_service.get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Session non trouvée"}), 404

        filepath = None
        download_name = None
        mimetype = None

        if file_type == "template":
            filepath = session_data["template_file_path"]
            if not filepath:
                return jsonify({"error": "Template non généré"}), 404
            download_name = os.path.basename(filepath)
            mimetype = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif file_type == "final":
            filepath = session_data["final_file_path"]
            if not filepath:
                return jsonify({"error": "Fichier final non généré"}), 404
            download_name = os.path.basename(filepath)
            mimetype = "text/csv"
        else:
            return jsonify({"error": "Type de fichier invalide"}), 400

        if not os.path.exists(filepath):
            return jsonify({"error": "Fichier non trouvé sur le serveur"}), 404

        return send_file(
            filepath, as_attachment=True, download_name=download_name, mimetype=mimetype
        )

    except Exception as e:
        logger.error(f"Erreur téléchargement: {e}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """Liste les sessions avec pagination"""
    try:
        limit = int(request.args.get("limit", 50))
        include_expired = request.args.get("include_expired", "false").lower() == "true"

        sessions_list = session_service.list_sessions(
            limit=limit, include_expired=include_expired
        )

        return jsonify({"sessions": sessions_list})

    except Exception as e:
        logger.error(f"Erreur listage sessions: {e}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Supprime une session"""
    try:
        success = session_service.delete_session(session_id)
        if success:
            # Nettoyer les données de session
            session_service.cleanup_session_data(session_id)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Session non trouvée"}), 404
    except Exception as e:
        logger.error(f"Erreur suppression session: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/analyze/<session_id>", methods=["GET"])
def analyze_file_format(session_id: str):
    """Endpoint pour analyser le format d'un fichier uploadé"""
    try:
        session = session_service.get_session(session_id)
        if not session:
            return jsonify({"error": "Session non trouvée"}), 404

        filepath = session.original_file_path
        if not os.path.exists(filepath):
            return jsonify({"error": "Fichier non trouvé"}), 404

        format_detected, format_msg, format_info = file_processor.detect_file_format(
            filepath
        )

        return jsonify(
            {
                "success": format_detected,
                "message": format_msg,
                "format_info": format_info,
                "expected_format": {
                    "columns_required": len(file_processor.SAGE_COLUMN_NAMES_ORDERED),
                    "column_names": file_processor.SAGE_COLUMN_NAMES_ORDERED,
                    "expected_line_types": ["E", "L", "S"],
                },
            }
        )

    except Exception as e:
        logger.error(f"Erreur analyse format: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint de santé amélioré"""
    try:
        db_healthy = db_manager.health_check()
        sessions_count = len(session_service.list_sessions(limit=1000))

        status = "healthy" if db_healthy else "degraded"

        return jsonify(
            {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "database": "healthy" if db_healthy else "error",
                "active_sessions_count": sessions_count,
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
            ),
            500,
        )


# Tâche de nettoyage (à exécuter périodiquement)
@app.route("/api/cleanup", methods=["POST"])
def cleanup_sessions():
    """Nettoie les sessions expirées"""
    try:
        hours = int(request.json.get("hours", 24))

        # Nettoyage des sessions en base
        cleaned_sessions = session_service.cleanup_expired_sessions(hours)

        # Nettoyage des fichiers anciens
        days_old = int(request.json.get("days_old", 7))
        file_stats = file_manager.cleanup_old_files(days_old)

        return jsonify(
            {
                "cleaned_sessions": cleaned_sessions,
                "cleaned_files": file_stats,
                "total_files_cleaned": sum(file_stats.values()),
            }
        )
    except Exception as e:
        logger.error(f"Erreur nettoyage: {e}")
        return jsonify({"error": "Erreur nettoyage"}), 500


@app.route("/api/archive/<session_id>", methods=["POST"])
def archive_session(session_id: str):
    """Archive une session et ses fichiers"""
    try:
        # Vérifier que la session existe
        session_data = session_service.get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Session non trouvée"}), 404

        # Archiver les fichiers
        success = file_manager.archive_session_files(
            session_id, session_data.get("created_at")
        )

        if success:
            # Marquer la session comme archivée
            session_service.update_session(session_id, status="archived")
            return jsonify({"success": True, "message": "Session archivée avec succès"})
        else:
            return jsonify({"error": "Erreur lors de l'archivage"}), 500

    except Exception as e:
        logger.error(f"Erreur archivage session {session_id}: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/verify/<session_id>", methods=["GET"])
@handle_api_errors("file_verification")
def verify_final_file(session_id: str):
    """Endpoint pour vérifier en détail le fichier final d'une session"""
    try:
        # Récupérer les données de session
        session_data = session_service.get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Session non trouvée"}), 404
        
        final_file_path = session_data.get("final_file_path")
        if not final_file_path:
            return jsonify({"error": "Aucun fichier final généré pour cette session"}), 404
        
        # Effectuer la vérification détaillée
        verification_summary = processor._verify_final_file_with_summary(final_file_path)
        
        # Ajouter des métadonnées de session
        verification_summary["session_info"] = {
            "session_id": session_id,
            "original_filename": session_data.get("original_filename"),
            "status": session_data.get("status"),
            "created_at": session_data.get("created_at"),
            "strategy_used": session_data.get("strategy_used", "FIFO")
        }
        
        return jsonify({
            "success": True,
            "verification": verification_summary
        })
        
    except Exception as e:
        logger.error(f"Erreur vérification session {session_id}: {e}")
        return jsonify({"error": "Erreur lors de la vérification"}), 500

@app.route("/api/verify-quantities/<session_id>", methods=["GET"])
@handle_api_errors("quantities_verification")
def verify_quantities_consistency(session_id: str):
    """Endpoint pour vérifier spécifiquement la cohérence des quantités"""
    try:
        # Récupérer les données de session
        session_data = session_service.get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Session non trouvée"}), 404
        
        final_file_path = session_data.get("final_file_path")
        if not final_file_path:
            return jsonify({"error": "Aucun fichier final généré pour cette session"}), 404
        
        # Charger les DataFrames nécessaires
        completed_df = session_service.load_dataframe(session_id, "completed_df")
        distributed_df = session_service.load_dataframe(session_id, "distributed_df")
        
        if completed_df is None or distributed_df is None:
            return jsonify({"error": "Données de traitement manquantes"}), 404
        
        # Effectuer la vérification des quantités
        quantities_verification = processor._verify_quantities_consistency(
            final_file_path, completed_df, distributed_df
        )
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "quantities_verification": quantities_verification,
            "message": "Vérification des quantités terminée"
        })
        
    except Exception as e:
        logger.error(f"Erreur vérification quantités session {session_id}: {e}")
        return jsonify({"error": "Erreur lors de la vérification des quantités"}), 500

@app.route("/api/stats/sessions", methods=["GET"])
@handle_api_errors("session_stats")
def get_session_stats():
    """Retourne les statistiques détaillées de toutes les sessions"""
    try:
        # Récupérer toutes les sessions
        sessions = session_service.list_sessions(limit=100, include_expired=True)
        
        # Calculer les statistiques globales
        stats = {
            "total_sessions": len(sessions),
            "sessions_by_status": {},
            "total_files_processed": 0,
            "total_articles_processed": 0,
            "total_quantity_processed": 0,
            "total_discrepancies": 0,
            "sessions_with_lotecart": 0,
            "strategies_used": {},
            "recent_sessions": [],
            "processing_times": {
                "last_24h": 0,
                "last_7d": 0,
                "last_30d": 0
            }
        }
        
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        
        for session in sessions:
            # Compter par statut
            status = session.get("status", "unknown")
            stats["sessions_by_status"][status] = stats["sessions_by_status"].get(status, 0) + 1
            
            # Accumuler les totaux
            session_stats = session.get("stats", {})
            stats["total_articles_processed"] += session_stats.get("nb_articles", 0)
            stats["total_quantity_processed"] += session_stats.get("total_quantity", 0)
            stats["total_discrepancies"] += session_stats.get("total_discrepancy", 0)
            
            # Compter les stratégies
            strategy = session_stats.get("strategy_used", "unknown")
            stats["strategies_used"][strategy] = stats["strategies_used"].get(strategy, 0) + 1
            
            # Sessions récentes (dernières 10)
            if len(stats["recent_sessions"]) < 10:
                stats["recent_sessions"].append({
                    "id": session["id"],
                    "filename": session.get("original_filename", "N/A"),
                    "status": status,
                    "created_at": session.get("created_at"),
                    "articles": session_stats.get("nb_articles", 0),
                    "discrepancy": session_stats.get("total_discrepancy", 0)
                })
            
            # Compter par période
            if session.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(session["created_at"].replace('Z', '+00:00'))
                    if created_at > now - timedelta(hours=24):
                        stats["processing_times"]["last_24h"] += 1
                    if created_at > now - timedelta(days=7):
                        stats["processing_times"]["last_7d"] += 1
                    if created_at > now - timedelta(days=30):
                        stats["processing_times"]["last_30d"] += 1
                except:
                    pass
        
        return jsonify({
            "success": True,
            "stats": stats,
            "generated_at": now.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur récupération stats sessions: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500

@app.route("/api/stats/files", methods=["GET"])
def get_file_stats():
    """Retourne les statistiques des fichiers"""
    try:
        stats = file_manager.get_folder_stats()
        return jsonify({"folder_stats": stats})
    except Exception as e:
        logger.error(f"Erreur stats fichiers: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500


if __name__ == "__main__":
    logger.info("Démarrage de l'application Moulinette Sage X3")
    app.run(host="0.0.0.0", port=5000, debug=True)
