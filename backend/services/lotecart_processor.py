import pandas as pd
import logging
from typing import Tuple, List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class LotecartProcessor:
    """
    Service spécialisé pour le traitement des lots LOTECART
    
    LOTECART = Lot d'écart automatique créé quand:
    - Quantité Théorique = 0 (pas de stock prévu)
    - Quantité Réelle > 0 (stock trouvé lors de l'inventaire)
    
    Ces lots nécessitent la création de nouvelles lignes dans le fichier Sage X3
    """
    
    def __init__(self):
        self.lotecart_counter = 0
    
    def detect_lotecart_candidates(self, completed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte les candidats LOTECART dans le fichier complété
        
        Args:
            completed_df: DataFrame du template complété avec quantités réelles
            
        Returns:
            DataFrame contenant uniquement les candidats LOTECART
        """
        try:
            if completed_df.empty:
                logger.warning("DataFrame complété vide pour détection LOTECART")
                return pd.DataFrame()
            
            # Nettoyer et convertir les colonnes
            df_clean = completed_df.copy()
            
            # Conversion sécurisée des quantités
            df_clean["Quantité Théorique"] = pd.to_numeric(
                df_clean["Quantité Théorique"], errors="coerce"
            ).fillna(0)
            
            df_clean["Quantité Réelle"] = pd.to_numeric(
                df_clean["Quantité Réelle"], errors="coerce"
            ).fillna(0)
            
            # Critère LOTECART: Qté Théorique = 0 ET Qté Réelle > 0
            lotecart_mask = (
                (df_clean["Quantité Théorique"] == 0) & 
                (df_clean["Quantité Réelle"] > 0)
            )
            
            lotecart_candidates = df_clean[lotecart_mask].copy()
            
            if not lotecart_candidates.empty:
                # Marquer comme LOTECART et calculer l'écart
                lotecart_candidates["Type_Lot"] = "lotecart"
                lotecart_candidates["Écart"] = lotecart_candidates["Quantité Réelle"]
                lotecart_candidates["Is_Lotecart"] = True
                
                logger.info(f"🎯 {len(lotecart_candidates)} candidats LOTECART détectés")
                
                # Log détaillé pour traçabilité
                for _, row in lotecart_candidates.iterrows():
                    logger.info(
                        f"   📦 LOTECART: {row['Code Article']} "
                        f"(Inv: {row.get('Numéro Inventaire', 'N/A')}) - "
                        f"Qté Théo=0 → Qté Réelle={row['Quantité Réelle']}"
                    )
            else:
                logger.info("ℹ️ Aucun candidat LOTECART détecté")
            
            return lotecart_candidates
            
        except Exception as e:
            logger.error(f"❌ Erreur détection candidats LOTECART: {e}", exc_info=True)
            return pd.DataFrame()
    
    def create_lotecart_adjustments(
        self, 
        lotecart_candidates: pd.DataFrame, 
        original_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Crée les ajustements pour les lots LOTECART en vérifiant d'abord si des lignes existent déjà
        
        Args:
            lotecart_candidates: DataFrame des candidats LOTECART
            original_df: DataFrame des données originales Sage X3
            
        Returns:
            Liste des ajustements à appliquer
        """
        adjustments = []
        
        try:
            if lotecart_candidates.empty:
                logger.info("ℹ️ Aucun candidat LOTECART à traiter")
                return adjustments
            
            for _, candidate in lotecart_candidates.iterrows():
                code_article = candidate["Code Article"]
                numero_inventaire = candidate.get("Numéro Inventaire", "")
                quantite_reelle = float(candidate["Quantité Réelle"])
                
                # Vérifier d'abord s'il existe déjà une ligne avec quantité théorique = 0 pour cet article
                existing_zero_qty_line = original_df[
                    (original_df["CODE_ARTICLE"] == code_article) &
                    (original_df["NUMERO_INVENTAIRE"] == numero_inventaire) &
                    (original_df["QUANTITE"] == 0)
                ]
                
                if not existing_zero_qty_line.empty:
                    # Ligne existante trouvée avec quantité = 0, la mettre à jour
                    existing_line = existing_zero_qty_line.iloc[0]
                    
                    adjustment = {
                        "CODE_ARTICLE": code_article,
                        "NUMERO_INVENTAIRE": numero_inventaire,
                        "NUMERO_LOT": existing_line.get("NUMERO_LOT", ""),  # Garder le lot original
                        "TYPE_LOT": "lotecart",
                        "QUANTITE_ORIGINALE": 0,  # Était 0 dans le fichier original
                        "AJUSTEMENT": quantite_reelle,
                        "QUANTITE_CORRIGEE": quantite_reelle,
                        "Date_Lot": existing_line.get("Date_Lot"),
                        "original_s_line_raw": existing_line.get("original_s_line_raw"),
                        "is_new_lotecart": False,  # Pas une nouvelle ligne, mise à jour d'une existante
                        "is_existing_update": True,  # Flag pour indiquer que c'est une mise à jour
                        # Métadonnées pour traçabilité
                        "metadata": {
                            "detection_reason": "qty_theo_0_qty_real_positive",
                            "existing_lot": existing_line.get("NUMERO_LOT", ""),
                            "existing_site": existing_line.get("SITE", ""),
                            "existing_emplacement": existing_line.get("EMPLACEMENT", ""),
                            "update_type": "existing_line_update"
                        }
                    }
                    
                    adjustments.append(adjustment)
                    
                    logger.info(
                        f"✅ Mise à jour ligne existante LOTECART: {code_article} "
                        f"(Lot={existing_line.get('NUMERO_LOT', 'N/A')}, Qté=0→{quantite_reelle})"
                    )
                    continue
                
                # Si aucune ligne existante avec quantité = 0, chercher une ligne de référence pour créer une nouvelle ligne
                reference_query = original_df["CODE_ARTICLE"] == code_article
                
                if numero_inventaire:
                    reference_query &= original_df["NUMERO_INVENTAIRE"] == numero_inventaire
                
                reference_lots = original_df[reference_query]
                
                if not reference_lots.empty:
                    # Prendre la première ligne comme référence
                    ref_lot = reference_lots.iloc[0]
                    
                    # Créer un nouvel ajustement LOTECART (nouvelle ligne)
                    adjustment = {
                        "CODE_ARTICLE": code_article,
                        "NUMERO_INVENTAIRE": numero_inventaire,
                        "NUMERO_LOT": "LOTECART",
                        "TYPE_LOT": "lotecart",
                        "QUANTITE_ORIGINALE": 0,  # Toujours 0 pour LOTECART
                        "AJUSTEMENT": quantite_reelle,
                        "QUANTITE_CORRIGEE": quantite_reelle,
                        "Date_Lot": None,  # Pas de date pour LOTECART
                        "original_s_line_raw": None,  # Nouvelle ligne à créer
                        "reference_line": ref_lot.get("original_s_line_raw"),
                        "is_new_lotecart": True,  # Flag spécial LOTECART
                        "is_existing_update": False,  # Pas une mise à jour, nouvelle ligne
                        # Métadonnées pour traçabilité
                        "metadata": {
                            "detection_reason": "qty_theo_0_qty_real_positive",
                            "reference_lot": ref_lot.get("NUMERO_LOT", ""),
                            "reference_site": ref_lot.get("SITE", ""),
                            "reference_emplacement": ref_lot.get("EMPLACEMENT", ""),
                            "update_type": "new_line_creation"
                        }
                    }
                    
                    adjustments.append(adjustment)
                    
                    logger.info(
                        f"✅ Nouvelle ligne LOTECART créée: {code_article} "
                        f"(Qté={quantite_reelle}, Ref={ref_lot.get('NUMERO_LOT', 'N/A')})"
                    )
                else:
                    logger.warning(
                        f"⚠️ Aucune ligne de référence trouvée pour LOTECART: "
                        f"{code_article} dans inventaire {numero_inventaire}"
                    )
            
            logger.info(f"🎯 {len(adjustments)} ajustements LOTECART créés au total")
            return adjustments
            
        except Exception as e:
            logger.error(f"❌ Erreur création ajustements LOTECART: {e}", exc_info=True)
            return []
    
    def generate_lotecart_lines(
        self, 
        lotecart_adjustments: List[Dict[str, Any]], 
        max_line_number: int = 0
    ) -> List[str]:
        """
        Génère les nouvelles lignes LOTECART pour le fichier final Sage X3
        
        Args:
            lotecart_adjustments: Liste des ajustements LOTECART
            max_line_number: Numéro de ligne maximum existant
            
        Returns:
            Liste des nouvelles lignes au format Sage X3
        """
        new_lines = []
        
        try:
            if not lotecart_adjustments:
                logger.info("ℹ️ Aucun ajustement LOTECART à générer")
                return new_lines
            
            current_line_number = max_line_number
            
            for adjustment in lotecart_adjustments:
                if not adjustment.get("is_new_lotecart", False):
                    continue
                
                reference_line = adjustment.get("reference_line")
                if not reference_line:
                    logger.warning(
                        f"⚠️ Pas de ligne de référence pour LOTECART {adjustment['CODE_ARTICLE']}"
                    )
                    continue
                
                # Validation et conversion sécurisée de la ligne de référence
                try:
                    # Vérifier si c'est NaN (pour les floats)
                    import pandas as pd
                    if pd.isna(reference_line):
                        logger.warning(
                            f"⚠️ Ligne de référence NaN pour LOTECART {adjustment['CODE_ARTICLE']}"
                        )
                        continue
                    
                    # Convertir en string de manière sécurisée
                    reference_line_str = str(reference_line).strip()
                    if not reference_line_str or reference_line_str.lower() in ['nan', 'none', '']:
                        logger.warning(
                            f"⚠️ Ligne de référence vide ou invalide pour LOTECART {adjustment['CODE_ARTICLE']}"
                        )
                        continue
                    
                    # Parser la ligne de référence
                    parts = reference_line_str.split(";")
                    
                except Exception as parse_error:
                    logger.warning(
                        f"⚠️ Erreur parsing ligne de référence pour LOTECART {adjustment['CODE_ARTICLE']}: {parse_error}"
                    )
                    continue
                
                if len(parts) < 15:
                    logger.warning(
                        f"⚠️ Ligne de référence trop courte ({len(parts)} colonnes) "
                        f"pour {adjustment['CODE_ARTICLE']}"
                    )
                    continue
                
                # Générer un nouveau numéro de ligne unique
                current_line_number += 1000
                self.lotecart_counter += 1
                
                # Construire la nouvelle ligne LOTECART
                new_parts = parts.copy()
                
                # Récupérer les quantités
                quantite_reelle = adjustment.get("QUANTITE_REELLE", adjustment["QUANTITE_CORRIGEE"])
                quantite_reelle_saisie = adjustment.get("QUANTITE_REELLE_SAISIE", quantite_reelle)
                
                # Modifications spécifiques LOTECART
                new_parts[3] = str(current_line_number)  # RANG - nouveau numéro
                new_parts[5] = str(int(quantite_reelle))  # QUANTITE théorique = quantité réelle pour LOTECART
                new_parts[6] = str(int(quantite_reelle_saisie))  # QUANTITE_REELLE_IN_INPUT = quantité réelle saisie
                new_parts[7] = "2"  # INDICATEUR_COMPTE - toujours 2 pour LOTECART
                new_parts[14] = "LOTECART"  # NUMERO_LOT - identifiant spécial
                
                # Assurer la cohérence des autres champs
                # Garder les mêmes valeurs que la ligne de référence pour:
                # - SITE, EMPLACEMENT, STATUT, UNITE, ZONE_PK
                
                new_line = ";".join(new_parts)
                new_lines.append(new_line)
                
                logger.info(
                    f"✅ Ligne LOTECART générée: {adjustment['CODE_ARTICLE']} "
                    f"(Ligne={current_line_number}, Qté={adjustment['QUANTITE_CORRIGEE']})"
                )
            
            logger.info(f"🎯 {len(new_lines)} nouvelles lignes LOTECART générées")
            return new_lines
            
        except Exception as e:
            logger.error(f"❌ Erreur génération lignes LOTECART: {e}", exc_info=True)
            return []
    
    def validate_lotecart_processing(
        self, 
        final_file_path: str, 
        expected_lotecart_count: int
    ) -> Dict[str, Any]:
        """
        Valide que le traitement LOTECART s'est bien déroulé
        
        Args:
            final_file_path: Chemin vers le fichier final généré
            expected_lotecart_count: Nombre de LOTECART attendus
            
        Returns:
            Dictionnaire avec les résultats de validation
        """
        validation_result = {
            "success": False,
            "lotecart_lines_found": 0,
            "correct_indicators": 0,
            "issues": []
        }
        
        try:
            if not final_file_path or not expected_lotecart_count:
                validation_result["issues"].append("Paramètres de validation manquants")
                return validation_result
            
            # Lire et analyser le fichier final
            lotecart_lines = []
            
            with open(final_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line.startswith('S;') and 'LOTECART' in line:
                        parts = line.split(';')
                        lotecart_lines.append({
                            'line_number': line_num,
                            'article': parts[8] if len(parts) > 8 else 'N/A',
                            'quantite': parts[5] if len(parts) > 5 else 'N/A',
                            'indicateur': parts[7] if len(parts) > 7 else 'N/A'
                        })
            
            validation_result["lotecart_lines_found"] = len(lotecart_lines)
            
            # Vérifications
            if len(lotecart_lines) < expected_lotecart_count:
                validation_result["issues"].append(
                    f"Nombre de lignes LOTECART insuffisant: {len(lotecart_lines)} < {expected_lotecart_count}"
                )
            
            # Vérifier les indicateurs de compte
            incorrect_indicators = [
                line for line in lotecart_lines 
                if line['indicateur'] != '2'
            ]
            
            validation_result["correct_indicators"] = len(lotecart_lines) - len(incorrect_indicators)
            
            if incorrect_indicators:
                validation_result["issues"].append(
                    f"Indicateurs incorrects sur {len(incorrect_indicators)} lignes LOTECART"
                )
            
            # Succès si pas de problèmes majeurs
            validation_result["success"] = len(validation_result["issues"]) == 0
            
            if validation_result["success"]:
                logger.info(
                    f"✅ Validation LOTECART réussie: {len(lotecart_lines)} lignes correctes"
                )
            else:
                logger.warning(
                    f"⚠️ Validation LOTECART avec problèmes: {validation_result['issues']}"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Erreur validation LOTECART: {e}", exc_info=True)
            validation_result["issues"].append(f"Erreur de validation: {str(e)}")
            return validation_result
    
    def get_lotecart_summary(
        self, 
        lotecart_candidates: pd.DataFrame,
        lotecart_adjustments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Génère un résumé du traitement LOTECART
        
        Args:
            lotecart_candidates: DataFrame des candidats détectés
            lotecart_adjustments: Liste des ajustements créés
            
        Returns:
            Dictionnaire avec le résumé
        """
        try:
            total_quantity = 0
            articles_by_inventory = {}
            existing_updates = 0
            new_lines = 0
            
            if not lotecart_candidates.empty:
                total_quantity = lotecart_candidates["Quantité Réelle"].sum()
                
                # Grouper par inventaire
                for _, row in lotecart_candidates.iterrows():
                    inv = row.get("Numéro Inventaire", "N/A")
                    if inv not in articles_by_inventory:
                        articles_by_inventory[inv] = []
                    
                    articles_by_inventory[inv].append({
                        "article": row["Code Article"],
                        "quantity": row["Quantité Réelle"]
                    })
            
            # Compter les types d'ajustements
            for adjustment in lotecart_adjustments:
                if adjustment.get("is_existing_update", False):
                    existing_updates += 1
                elif adjustment.get("is_new_lotecart", False):
                    new_lines += 1
            
            summary = {
                "candidates_detected": len(lotecart_candidates),
                "adjustments_created": len(lotecart_adjustments),
                "existing_lines_updated": existing_updates,
                "new_lines_created": new_lines,
                "total_quantity": float(total_quantity),
                "inventories_affected": len(articles_by_inventory),
                "articles_by_inventory": articles_by_inventory,
                "processing_timestamp": pd.Timestamp.now().isoformat()
            }
            
            logger.info(
                f"📊 Résumé LOTECART: {summary['candidates_detected']} candidats, "
                f"{existing_updates} lignes mises à jour, {new_lines} nouvelles lignes, "
                f"{summary['total_quantity']} unités"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Erreur génération résumé LOTECART: {e}", exc_info=True)
            return {
                "candidates_detected": 0,
                "adjustments_created": 0,
                "existing_lines_updated": 0,
                "new_lines_created": 0,
                "total_quantity": 0,
                "inventories_affected": 0,
                "articles_by_inventory": {},
                "error": str(e)
            }
    
    def reset_counter(self):
        """Remet à zéro le compteur LOTECART"""
        self.lotecart_counter = 0
        logger.debug("🔄 Compteur LOTECART remis à zéro")