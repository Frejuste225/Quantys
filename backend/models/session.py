from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class Session(Base):
    __tablename__ = 'sessions'
    
    id = Column(String(8), primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    original_filename = Column(String(255), nullable=False)
    original_file_path = Column(String(500), nullable=False)
    template_file_path = Column(String(500))
    completed_file_path = Column(String(500))
    final_file_path = Column(String(500))
    
    status = Column(String(50), default='created')
    inventory_date = Column(DateTime)
    
    # Statistiques
    nb_articles = Column(Integer, default=0)
    nb_lots = Column(Integer, default=0)
    total_quantity = Column(Float, default=0.0)
    total_discrepancy = Column(Float, default=0.0)
    adjusted_items_count = Column(Integer, default=0)
    strategy_used = Column(String(10), default='FIFO')
    
    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # Données sérialisées (JSON)
    header_lines = Column(Text)  # JSON string
    
    def to_dict(self):
        return {
            'id': self.id,
            'original_filename': self.original_filename,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'inventory_date': self.inventory_date.isoformat() if self.inventory_date else None,
            'stats': {
                'nb_articles': self.nb_articles,
                'nb_lots': self.nb_lots,
                'total_quantity': self.total_quantity,
                'total_discrepancy': self.total_discrepancy,
                'adjusted_items_count': self.adjusted_items_count,
                'strategy_used': self.strategy_used
            }
        }