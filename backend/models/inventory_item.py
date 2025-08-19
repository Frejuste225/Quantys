from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from .session import Base

class InventoryItem(Base):
    __tablename__ = 'inventory_items'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(8), ForeignKey('sessions.id'), nullable=False)
    
    # Données Sage X3
    type_ligne = Column(String(1), default='S')
    numero_session = Column(String(50))
    numero_inventaire = Column(String(50))
    rang = Column(Integer)
    site = Column(String(10))
    quantite = Column(Float, nullable=False)
    quantite_reelle_input = Column(Float, default=0)
    indicateur_compte = Column(String(10))
    code_article = Column(String(50), nullable=False)
    emplacement = Column(String(50))
    statut = Column(String(10))
    unite = Column(String(10))
    valeur = Column(Float)
    zone_pk = Column(String(50))
    numero_lot = Column(String(100))
    
    # Données calculées
    date_lot = Column(DateTime)
    quantite_corrigee = Column(Float)
    original_s_line_raw = Column(Text)
    
    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relation
    session = relationship("Session", backref="inventory_items")
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'code_article': self.code_article,
            'quantite': self.quantite,
            'quantite_corrigee': self.quantite_corrigee,
            'numero_lot': self.numero_lot,
            'date_lot': self.date_lot.isoformat() if self.date_lot else None,
            'emplacement': self.emplacement,
            'statut': self.statut,
            'zone_pk': self.zone_pk,
            'unite': self.unite
        }