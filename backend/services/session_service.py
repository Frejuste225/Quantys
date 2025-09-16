import json
from datetime import datetime, timedelta
import os
from sqlalchemy.orm import Session as DBSession
from models.session import Session
from models.inventory_item import InventoryItem
from database import db_manager
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SessionService:
    def __init__(self):
        self.db = db_manager
        # Dossiers pour la persistance des DataFrames
        self.data_folder = "data/session_data"
        os.makedirs(self.data_folder, exist_ok=True)
        # Cache en mémoire pour éviter les rechargements répétés
        self._dataframe_cache = {}

    def create_session(
        self, original_filename: str, original_file_path: str, **kwargs
    ) -> str:
        """Crée une nouvelle session en base de données"""
        db_session = self.db.get_session()
        try:
            session = Session(
                original_filename=original_filename,
                original_file_path=original_file_path,
                **kwargs,
            )
            db_session.add(session)
            db_session.commit()

            logger.info(f"Session créée: {session.id}")
            return session.id
        except Exception as e:
            db_session.rollback()
            logger.error(f"Erreur création session: {e}")
            raise
        finally:
            db_session.close()

    def save_dataframe(self, session_id: str, df_name: str, dataframe: pd.DataFrame):
        """Sauvegarde un DataFrame en format Parquet pour une session"""
        try:
            file_path = os.path.join(
                self.data_folder, f"{session_id}_{df_name}.parquet"
            )
            dataframe.to_parquet(file_path, index=False)

            # Mettre à jour le cache
            cache_key = f"{session_id}_{df_name}"
            self._dataframe_cache[cache_key] = dataframe.copy()

            logger.info(f"DataFrame {df_name} sauvegardé pour session {session_id}")
        except Exception as e:
            logger.error(
                f"Erreur sauvegarde DataFrame {df_name} pour session {session_id}: {e}"
            )
            raise

    def load_dataframe(self, session_id: str, df_name: str) -> pd.DataFrame:
        """Charge un DataFrame depuis le stockage pour une session avec cache"""
        cache_key = f"{session_id}_{df_name}"

        # Vérifier le cache d'abord
        if cache_key in self._dataframe_cache:
            logger.debug(
                f"DataFrame {df_name} récupéré du cache pour session {session_id}"
            )
            return self._dataframe_cache[cache_key]

        try:
            file_path = os.path.join(
                self.data_folder, f"{session_id}_{df_name}.parquet"
            )
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                # Mettre en cache
                self._dataframe_cache[cache_key] = df
                logger.info(f"DataFrame {df_name} chargé pour session {session_id}")
                return df
            else:
                logger.warning(
                    f"DataFrame {df_name} non trouvé pour session {session_id}"
                )
                return None
        except Exception as e:
            logger.error(
                f"Erreur chargement DataFrame {df_name} pour session {session_id}: {e}"
            )
            return None

    def cleanup_session_data(self, session_id: str):
        """Nettoie les fichiers de données d'une session"""
        try:
            import glob

            pattern = os.path.join(self.data_folder, f"{session_id}_*.parquet")
            files = glob.glob(pattern)
            for file_path in files:
                os.remove(file_path)
                logger.info(f"Fichier de données supprimé: {file_path}")
        except Exception as e:
            logger.error(f"Erreur nettoyage données session {session_id}: {e}")

    def get_session(self, session_id: str) -> Session:
        """Récupère une session par ID"""
        db_session = self.db.get_session()
        try:
            session = db_session.query(Session).filter(Session.id == session_id).first()
            if session:
                # Mettre à jour last_accessed
                session.last_accessed = datetime.utcnow()
                db_session.commit()
                # Rafraîchir l'objet pour s'assurer qu'il reste attaché
                db_session.refresh(session)
            return session
        except Exception as e:
            logger.error(f"Erreur récupération session {session_id}: {e}")
            return None
        finally:
            db_session.close()

    def get_session_data(self, session_id: str) -> dict:
        """Récupère les données d'une session sous forme de dictionnaire"""
        db_session = self.db.get_session()
        try:
            session = db_session.query(Session).filter(Session.id == session_id).first()
            if session:
                # Mettre à jour last_accessed
                session.last_accessed = datetime.utcnow()
                db_session.commit()

                # Retourner un dictionnaire avec toutes les données nécessaires
                return {
                    "id": session.id,
                    "original_filename": session.original_filename,
                    "original_file_path": session.original_file_path,
                    "template_file_path": session.template_file_path,
                    "completed_file_path": session.completed_file_path,
                    "final_file_path": session.final_file_path,
                    "status": session.status,
                    "inventory_date": session.inventory_date,
                    "nb_articles": session.nb_articles,
                    "nb_lots": session.nb_lots,
                    "total_quantity": session.total_quantity,
                    "total_discrepancy": session.total_discrepancy,
                    "adjusted_items_count": session.adjusted_items_count,
                    "strategy_used": session.strategy_used,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "last_accessed": session.last_accessed,
                    "header_lines": session.header_lines,
                }
            return None
        except Exception as e:
            logger.error(f"Erreur récupération données session {session_id}: {e}")
            return None
        finally:
            db_session.close()

    def update_session(self, session_id: str, **updates) -> bool:
        """Met à jour une session"""
        db_session = self.db.get_session()
        try:
            session = db_session.query(Session).filter(Session.id == session_id).first()
            if not session:
                return False

            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)

            session.updated_at = datetime.utcnow()
            session.last_accessed = datetime.utcnow()
            db_session.commit()

            logger.info(f"Session {session_id} mise à jour")
            return True
        except Exception as e:
            db_session.rollback()
            logger.error(f"Erreur mise à jour session {session_id}: {e}")
            return False
        finally:
            db_session.close()

    def list_sessions(self, limit: int = 50, include_expired: bool = False) -> list:
        """Liste les sessions"""
        db_session = self.db.get_session()
        try:
            query = db_session.query(Session)

            if not include_expired:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                query = query.filter(Session.last_accessed > cutoff_time)

            sessions = query.order_by(Session.created_at.desc()).limit(limit).all()
            return [session.to_dict() for session in sessions]
        except Exception as e:
            logger.error(f"Erreur listage sessions: {e}")
            return []
        finally:
            db_session.close()

    def delete_session(self, session_id: str) -> bool:
        """Supprime une session et ses données associées"""
        db_session = self.db.get_session()
        try:
            # Supprimer les items d'inventaire
            db_session.query(InventoryItem).filter(
                InventoryItem.session_id == session_id
            ).delete()

            # Supprimer la session
            session = db_session.query(Session).filter(Session.id == session_id).first()
            if session:
                db_session.delete(session)
                db_session.commit()
                logger.info(f"Session {session_id} supprimée")
                return True
            return False
        except Exception as e:
            db_session.rollback()
            logger.error(f"Erreur suppression session {session_id}: {e}")
            return False
        finally:
            db_session.close()

    def cleanup_expired_sessions(self, hours: int = 24):
        """Nettoie les sessions expirées"""
        db_session = self.db.get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # Récupérer les sessions expirées
            expired_sessions = (
                db_session.query(Session)
                .filter(Session.last_accessed < cutoff_time)
                .all()
            )

            count = 0
            for session in expired_sessions:
                # Supprimer les items associés
                db_session.query(InventoryItem).filter(
                    InventoryItem.session_id == session.id
                ).delete()

                # Supprimer la session
                db_session.delete(session)
                count += 1

            db_session.commit()
            logger.info(f"{count} sessions expirées supprimées")
            return count
        except Exception as e:
            db_session.rollback()
            logger.error(f"Erreur nettoyage sessions: {e}")
            return 0
        finally:
            db_session.close()

    def save_inventory_items(self, session_id: str, items_data: list):
        """Sauvegarde les items d'inventaire en base"""
        db_session = self.db.get_session()
        try:
            # Supprimer les anciens items de cette session
            db_session.query(InventoryItem).filter(
                InventoryItem.session_id == session_id
            ).delete()

            # Ajouter les nouveaux items
            for item_data in items_data:
                item = InventoryItem(session_id=session_id, **item_data)
                db_session.add(item)

            db_session.commit()
            logger.info(
                f"{len(items_data)} items sauvegardés pour session {session_id}"
            )
        except Exception as e:
            db_session.rollback()
            logger.error(f"Erreur sauvegarde items session {session_id}: {e}")
            raise
        finally:
            db_session.close()

    def get_inventory_items(self, session_id: str) -> list:
        """Récupère les items d'inventaire d'une session"""
        db_session = self.db.get_session()
        try:
            items = (
                db_session.query(InventoryItem)
                .filter(InventoryItem.session_id == session_id)
                .all()
            )
            return [item.to_dict() for item in items]
        except Exception as e:
            logger.error(f"Erreur récupération items session {session_id}: {e}")
            return []
        finally:
            db_session.close()
