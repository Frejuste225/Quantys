import pytest
import os
import tempfile
import shutil
from flask import Flask
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

# Import de l'application
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, config
from services.session_service import SessionService
from services.file_processor import FileProcessorService
from database import db_manager

@pytest.fixture
def client():
    """Client de test Flask"""
    app.config['TESTING'] = True
    app.config['DEBUG'] = True
    
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture
def temp_dir():
    """Répertoire temporaire pour les tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_config(temp_dir):
    """Configuration mockée pour les tests"""
    test_config = Mock()
    test_config.UPLOAD_FOLDER = os.path.join(temp_dir, 'uploads')
    test_config.PROCESSED_FOLDER = os.path.join(temp_dir, 'processed')
    test_config.FINAL_FOLDER = os.path.join(temp_dir, 'final')
    test_config.ARCHIVE_FOLDER = os.path.join(temp_dir, 'archive')
    test_config.LOG_FOLDER = os.path.join(temp_dir, 'logs')
    test_config.MAX_FILE_SIZE = 16 * 1024 * 1024
    
    # Créer les dossiers
    for folder in [test_config.UPLOAD_FOLDER, test_config.PROCESSED_FOLDER,
                   test_config.FINAL_FOLDER, test_config.ARCHIVE_FOLDER,
                   test_config.LOG_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    return test_config

@pytest.fixture
def sample_csv_content():
    """Contenu CSV Sage X3 de test"""
    return """E;BKE022508SES00000003;test depot conf;1;BKE02;;;;;;;;;;
L;BKE022508SES00000003;BKE022508INV00000006;1;BKE02;;;;;;;;;;
S;BKE022508SES00000003;BKE022508INV00000006;1000;BKE02;100;0;1;ART001;EMP001;A;UN;0;ZONE1;LOT123456;
S;BKE022508SES00000003;BKE022508INV00000006;1001;BKE02;50;0;1;ART002;EMP001;A;UN;0;ZONE1;CPKU070725001;
S;BKE022508SES00000003;BKE022508INV00000006;1002;BKE02;0;0;1;ART003;EMP001;A;UN;0;ZONE1;;"""

@pytest.fixture
def sample_csv_file(temp_dir, sample_csv_content):
    """Fichier CSV de test"""
    file_path = os.path.join(temp_dir, 'test_sage.csv')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_csv_content)
    return file_path

@pytest.fixture
def sample_dataframe():
    """DataFrame de test"""
    data = {
        'TYPE_LIGNE': ['S', 'S', 'S'],
        'NUMERO_SESSION': ['BKE022508SES00000003'] * 3,
        'NUMERO_INVENTAIRE': ['BKE022508INV00000006'] * 3,
        'RANG': [1000, 1001, 1002],
        'SITE': ['BKE02'] * 3,
        'QUANTITE': [100.0, 50.0, 0.0],
        'QUANTITE_REELLE_IN_INPUT': [0.0] * 3,
        'INDICATEUR_COMPTE': [1] * 3,
        'CODE_ARTICLE': ['ART001', 'ART002', 'ART003'],
        'EMPLACEMENT': ['EMP001'] * 3,
        'STATUT': ['A'] * 3,
        'UNITE': ['UN'] * 3,
        'VALEUR': [0.0] * 3,
        'ZONE_PK': ['ZONE1'] * 3,
        'NUMERO_LOT': ['LOT123456', 'CPKU070725001', '']
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_session_service():
    """Service de session mocké"""
    with patch('services.session_service.SessionService') as mock:
        service = Mock()
        service.create_session.return_value = 'test123'
        service.get_session_data.return_value = {
            'id': 'test123',
            'status': 'uploaded',
            'original_filename': 'test.csv'
        }
        mock.return_value = service
        yield service

@pytest.fixture
def mock_file_processor():
    """Service de traitement de fichiers mocké"""
    with patch('services.file_processor.FileProcessorService') as mock:
        service = Mock()
        service.validate_and_process_sage_file.return_value = (
            True, pd.DataFrame(), [], datetime.now().date()
        )
        service.aggregate_data.return_value = pd.DataFrame()
        service.generate_template.return_value = '/path/to/template.xlsx'
        mock.return_value = service
        yield service

@pytest.fixture
def mock_db():
    """Base de données mockée"""
    with patch('database.db_manager') as mock:
        yield mock