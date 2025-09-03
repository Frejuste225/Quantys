import pytest
import io
from unittest.mock import Mock, patch
from utils.validators import FileValidator, DataValidator
import pandas as pd

class TestFileValidator:
    """Tests pour FileValidator"""
    
    def test_validate_file_security_valid_csv(self):
        """Test validation d'un fichier CSV valide"""
        # Créer un fichier mock
        file_content = "E;test;data\nS;test;data;1;site;100;0;1;ART001;emp;A;UN;0;zone;lot"
        file_mock = Mock()
        file_mock.filename = "test.csv"
        file_mock.read.return_value = file_content.encode('utf-8')
        file_mock.seek = Mock()
        file_mock.tell.return_value = len(file_content)
        
        # Test
        is_valid, message = FileValidator.validate_file_security(file_mock, 1024*1024)
        
        assert is_valid == True
        assert "valide" in message.lower()
    
    def test_validate_file_security_too_large(self):
        """Test validation d'un fichier trop volumineux"""
        file_mock = Mock()
        file_mock.filename = "large.csv"
        file_mock.seek = Mock()
        file_mock.tell.return_value = 20 * 1024 * 1024  # 20MB
        
        is_valid, message = FileValidator.validate_file_security(file_mock, 16*1024*1024)
        
        assert is_valid == False
        assert "trop volumineux" in message.lower()
    
    def test_validate_file_security_empty_file(self):
        """Test validation d'un fichier vide"""
        file_mock = Mock()
        file_mock.filename = "empty.csv"
        file_mock.seek = Mock()
        file_mock.tell.return_value = 0
        
        is_valid, message = FileValidator.validate_file_security(file_mock, 1024*1024)
        
        assert is_valid == False
        assert "vide" in message.lower()
    
    def test_validate_file_security_too_small(self):
        """Test validation d'un fichier trop petit"""
        file_mock = Mock()
        file_mock.filename = "tiny.csv"
        file_mock.seek = Mock()
        file_mock.tell.return_value = 5  # 5 bytes
        
        is_valid, message = FileValidator.validate_file_security(file_mock, 1024*1024)
        
        assert is_valid == False
        assert "trop petit" in message.lower()
    
    def test_validate_file_security_invalid_extension(self):
        """Test validation d'une extension non autorisée"""
        file_mock = Mock()
        file_mock.filename = "test.txt"
        file_mock.seek = Mock()
        file_mock.tell.return_value = 100
        file_mock.read.return_value = b"some content"
        
        with patch('utils.validators.MAGIC_AVAILABLE', False):
            is_valid, message = FileValidator.validate_file_security(file_mock, 1024*1024)
        
        assert is_valid == False
        assert "non autorisée" in message.lower()
    
    def test_validate_csv_content_valid_sage_format(self):
        """Test validation du contenu CSV avec format Sage X3"""
        file_content = "E;header\nL;line\nS;data;line;with;sage;format"
        file_mock = Mock()
        file_mock.seek = Mock()
        file_mock.__iter__ = Mock(return_value=iter(file_content.split('\n')))
        file_mock.read.return_value = file_content.encode('utf-8')
        
        is_valid, message = FileValidator._validate_csv_content(file_mock)
        
        assert is_valid == True
        assert "valide" in message.lower()
    
    def test_validate_csv_content_no_sage_format(self):
        """Test validation du contenu CSV sans format Sage X3"""
        file_content = "header1,header2,header3\ndata1,data2,data3"
        file_mock = Mock()
        file_mock.seek = Mock()
        file_mock.__iter__ = Mock(return_value=iter(file_content.split('\n')))
        file_mock.read.return_value = file_content.encode('utf-8')
        
        is_valid, message = FileValidator._validate_csv_content(file_mock)
        
        assert is_valid == False
        assert "format sage x3 non détecté" in message.lower()
    
    def test_validate_csv_content_suspicious_content(self):
        """Test validation du contenu CSV avec contenu suspect"""
        file_content = "E;header\nS;data\n<script>alert('xss')</script>"
        file_mock = Mock()
        file_mock.seek = Mock()
        file_mock.__iter__ = Mock(return_value=iter(file_content.split('\n')))
        file_mock.read.return_value = file_content.encode('utf-8')
        
        is_valid, message = FileValidator._validate_csv_content(file_mock)
        
        assert is_valid == False
        assert "suspect" in message.lower()

class TestDataValidator:
    """Tests pour DataValidator"""
    
    def test_validate_sage_structure_valid(self):
        """Test validation d'une structure Sage X3 valide"""
        # Créer un DataFrame de test
        data = {
            0: ['S'] * 3,  # TYPE_LIGNE
            1: ['SESSION'] * 3,  # NUMERO_SESSION
            2: ['INV001'] * 3,  # NUMERO_INVENTAIRE
            3: [1, 2, 3],  # RANG
            4: ['SITE'] * 3,  # SITE
            5: [100.0, 50.0, 25.0],  # QUANTITE
            6: [0.0] * 3,  # QUANTITE_REELLE_IN_INPUT
            7: [1] * 3,  # INDICATEUR_COMPTE
            8: ['ART001', 'ART002', 'ART003'],  # CODE_ARTICLE
            9: ['EMP001'] * 3,  # EMPLACEMENT
            10: ['A'] * 3,  # STATUT
            11: ['UN'] * 3,  # UNITE
            12: [0.0] * 3,  # VALEUR
            13: ['ZONE1'] * 3,  # ZONE_PK
            14: ['LOT001', 'LOT002', 'LOT003']  # NUMERO_LOT
        }
        df = pd.DataFrame(data)
        
        required_columns = {
            'QUANTITE': 5,
            'CODE_ARTICLE': 8
        }
        
        is_valid, message = DataValidator.validate_sage_structure(df, required_columns)
        
        assert is_valid == True
        assert "valide" in message.lower()
    
    def test_validate_sage_structure_insufficient_columns(self):
        """Test validation avec nombre de colonnes insuffisant"""
        # DataFrame avec seulement 5 colonnes
        data = {
            0: ['S'] * 3,
            1: ['SESSION'] * 3,
            2: ['INV001'] * 3,
            3: [1, 2, 3],
            4: ['SITE'] * 3
        }
        df = pd.DataFrame(data)
        
        required_columns = {
            'QUANTITE': 5,
            'CODE_ARTICLE': 8
        }
        
        is_valid, message = DataValidator.validate_sage_structure(df, required_columns)
        
        assert is_valid == False
        assert "colonnes insuffisant" in message.lower()
    
    def test_validate_sage_structure_invalid_quantities(self):
        """Test validation avec quantités invalides"""
        data = {
            0: ['S'] * 3,
            1: ['SESSION'] * 3,
            2: ['INV001'] * 3,
            3: [1, 2, 3],
            4: ['SITE'] * 3,
            5: [100.0, 'invalid', 25.0],  # Quantité invalide
            6: [0.0] * 3,
            7: [1] * 3,
            8: ['ART001', 'ART002', 'ART003'],
            9: ['EMP001'] * 3,
            10: ['A'] * 3,
            11: ['UN'] * 3,
            12: [0.0] * 3,
            13: ['ZONE1'] * 3,
            14: ['LOT001', 'LOT002', 'LOT003']
        }
        df = pd.DataFrame(data)
        
        required_columns = {
            'QUANTITE': 5,
            'CODE_ARTICLE': 8
        }
        
        is_valid, message = DataValidator.validate_sage_structure(df, required_columns)
        
        assert is_valid == False
        assert "invalides" in message.lower()
    
    def test_validate_sage_structure_negative_quantities(self):
        """Test validation avec quantités négatives"""
        data = {
            0: ['S'] * 3,
            1: ['SESSION'] * 3,
            2: ['INV001'] * 3,
            3: [1, 2, 3],
            4: ['SITE'] * 3,
            5: [100.0, -50.0, 25.0],  # Quantité négative
            6: [0.0] * 3,
            7: [1] * 3,
            8: ['ART001', 'ART002', 'ART003'],
            9: ['EMP001'] * 3,
            10: ['A'] * 3,
            11: ['UN'] * 3,
            12: [0.0] * 3,
            13: ['ZONE1'] * 3,
            14: ['LOT001', 'LOT002', 'LOT003']
        }
        df = pd.DataFrame(data)
        
        required_columns = {
            'QUANTITE': 5,
            'CODE_ARTICLE': 8
        }
        
        is_valid, message = DataValidator.validate_sage_structure(df, required_columns)
        
        assert is_valid == False
        assert "négatives" in message.lower()
    
    def test_validate_template_completion_valid(self):
        """Test validation d'un template complété valide"""
        data = {
            'Numéro Session': ['SESSION001'] * 3,
            'Numéro Inventaire': ['INV001'] * 3,
            'Code Article': ['ART001', 'ART002', 'ART003'],
            'Quantité Théorique': [100, 50, 25],
            'Quantité Réelle': [95, 52, 25],
            'Numéro Lot': ['LOT001', 'LOT002', 'LOT003']
        }
        df = pd.DataFrame(data)
        
        is_valid, message, errors = DataValidator.validate_template_completion(df)
        
        assert is_valid == True
        assert "valide" in message.lower()
        assert len(errors) == 0
    
    def test_validate_template_completion_missing_columns(self):
        """Test validation d'un template avec colonnes manquantes"""
        data = {
            'Code Article': ['ART001', 'ART002', 'ART003'],
            'Quantité Théorique': [100, 50, 25],
            # Colonnes manquantes: Numéro Session, Numéro Inventaire, Quantité Réelle, Numéro Lot
        }
        df = pd.DataFrame(data)
        
        is_valid, message, errors = DataValidator.validate_template_completion(df)
        
        assert is_valid == False
        assert len(errors) > 0
        assert any("manquantes" in error.lower() for error in errors)
    
    def test_validate_template_completion_missing_quantities(self):
        """Test validation d'un template avec quantités manquantes"""
        data = {
            'Numéro Session': ['SESSION001'] * 3,
            'Numéro Inventaire': ['INV001'] * 3,
            'Code Article': ['ART001', 'ART002', 'ART003'],
            'Quantité Théorique': [100, 50, 25],
            'Quantité Réelle': [95, None, 25],  # Quantité manquante
            'Numéro Lot': ['LOT001', 'LOT002', 'LOT003']
        }
        df = pd.DataFrame(data)
        
        is_valid, message, errors = DataValidator.validate_template_completion(df)
        
        assert is_valid == False
        assert len(errors) > 0
        assert any("manquantes" in error.lower() for error in errors)