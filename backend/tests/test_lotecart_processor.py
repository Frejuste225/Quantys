import pytest
import pandas as pd
from unittest.mock import Mock, patch, mock_open
from services.lotecart_processor import LotecartProcessor

class TestLotecartProcessor:
    """Tests pour LotecartProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Instance du processeur LOTECART"""
        return LotecartProcessor()
    
    @pytest.fixture
    def sample_completed_df(self):
        """DataFrame de template complété avec candidats LOTECART"""
        data = {
            'Code Article': ['ART001', 'ART002', 'ART003', 'ART004'],
            'Numéro Inventaire': ['INV001'] * 4,
            'Quantité Théorique': [100, 0, 50, 0],  # ART002 et ART004 sont candidats
            'Quantité Réelle': [95, 25, 48, 10],    # ART002 et ART004 ont qté réelle > 0
            'Numéro Lot': ['LOT001', '', 'LOT003', '']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_original_df(self):
        """DataFrame des données originales Sage X3"""
        data = {
            'CODE_ARTICLE': ['ART001', 'ART002', 'ART003', 'ART004'],
            'NUMERO_INVENTAIRE': ['INV001'] * 4,
            'QUANTITE': [100.0, 0.0, 50.0, 0.0],
            'SITE': ['SITE01'] * 4,
            'EMPLACEMENT': ['EMP001'] * 4,
            'NUMERO_LOT': ['LOT001', '', 'LOT003', ''],
            'original_s_line_raw': [
                'S;SESSION;INV001;1000;SITE01;100;0;1;ART001;EMP001;A;UN;0;ZONE1;LOT001',
                'S;SESSION;INV001;1001;SITE01;0;0;1;ART002;EMP001;A;UN;0;ZONE1;',
                'S;SESSION;INV001;1002;SITE01;50;0;1;ART003;EMP001;A;UN;0;ZONE1;LOT003',
                'S;SESSION;INV001;1003;SITE01;0;0;1;ART004;EMP001;A;UN;0;ZONE1;'
            ]
        }
        return pd.DataFrame(data)
    
    def test_detect_lotecart_candidates_valid(self, processor, sample_completed_df):
        """Test détection des candidats LOTECART valides"""
        candidates = processor.detect_lotecart_candidates(sample_completed_df)
        
        # Doit détecter ART002 et ART004 (qté théo = 0, qté réelle > 0)
        assert len(candidates) == 2
        assert 'ART002' in candidates['Code Article'].values
        assert 'ART004' in candidates['Code Article'].values
        
        # Vérifier les propriétés des candidats
        for _, candidate in candidates.iterrows():
            assert candidate['Quantité Théorique'] == 0
            assert candidate['Quantité Réelle'] > 0
            assert candidate['Type_Lot'] == 'lotecart'
            assert candidate['Is_Lotecart'] == True
            assert candidate['Écart'] == candidate['Quantité Réelle']
    
    def test_detect_lotecart_candidates_empty_df(self, processor):
        """Test détection avec DataFrame vide"""
        empty_df = pd.DataFrame()
        candidates = processor.detect_lotecart_candidates(empty_df)
        
        assert candidates.empty
    
    def test_detect_lotecart_candidates_no_candidates(self, processor):
        """Test détection sans candidats LOTECART"""
        data = {
            'Code Article': ['ART001', 'ART002'],
            'Quantité Théorique': [100, 50],  # Pas de 0
            'Quantité Réelle': [95, 48]
        }
        df = pd.DataFrame(data)
        
        candidates = processor.detect_lotecart_candidates(df)
        assert candidates.empty
    
    def test_detect_lotecart_candidates_invalid_data(self, processor):
        """Test détection avec données invalides"""
        data = {
            'Code Article': ['ART001', 'ART002'],
            'Quantité Théorique': ['invalid', 0],
            'Quantité Réelle': [95, 'invalid']
        }
        df = pd.DataFrame(data)
        
        candidates = processor.detect_lotecart_candidates(df)
        # Doit gérer les erreurs gracieusement
        assert isinstance(candidates, pd.DataFrame)
    
    def test_create_lotecart_adjustments_valid(self, processor, sample_completed_df, sample_original_df):
        """Test création d'ajustements LOTECART valides"""
        candidates = processor.detect_lotecart_candidates(sample_completed_df)
        adjustments = processor.create_lotecart_adjustments(candidates, sample_original_df)
        
        assert len(adjustments) == 2  # ART002 et ART004
        
        for adjustment in adjustments:
            assert adjustment['NUMERO_LOT'] == 'LOTECART'
            assert adjustment['TYPE_LOT'] == 'lotecart'
            assert adjustment['QUANTITE_ORIGINALE'] == 0
            assert adjustment['AJUSTEMENT'] > 0
            assert adjustment['QUANTITE_CORRIGEE'] > 0
            assert adjustment['is_new_lotecart'] == True
            assert adjustment['reference_line'] is not None
            assert 'metadata' in adjustment
    
    def test_create_lotecart_adjustments_no_reference(self, processor):
        """Test création d'ajustements sans ligne de référence"""
        candidates_data = {
            'Code Article': ['ART999'],  # Article inexistant
            'Numéro Inventaire': ['INV001'],
            'Quantité Théorique': [0],
            'Quantité Réelle': [10],
            'Type_Lot': ['lotecart'],
            'Is_Lotecart': [True]
        }
        candidates = pd.DataFrame(candidates_data)
        
        original_data = {
            'CODE_ARTICLE': ['ART001'],  # Pas ART999
            'NUMERO_INVENTAIRE': ['INV001'],
            'original_s_line_raw': ['S;SESSION;INV001;1000;SITE01;100;0;1;ART001;EMP001;A;UN;0;ZONE1;LOT001']
        }
        original_df = pd.DataFrame(original_data)
        
        adjustments = processor.create_lotecart_adjustments(candidates, original_df)
        assert len(adjustments) == 0  # Aucun ajustement créé
    
    def test_generate_lotecart_lines_valid(self, processor):
        """Test génération de lignes LOTECART valides"""
        adjustments = [
            {
                'CODE_ARTICLE': 'ART002',
                'NUMERO_INVENTAIRE': 'INV001',
                'QUANTITE_CORRIGEE': 25,
                'is_new_lotecart': True,
                'reference_line': 'S;SESSION;INV001;1001;SITE01;0;0;1;ART002;EMP001;A;UN;0;ZONE1;'
            },
            {
                'CODE_ARTICLE': 'ART004',
                'NUMERO_INVENTAIRE': 'INV001',
                'QUANTITE_CORRIGEE': 10,
                'is_new_lotecart': True,
                'reference_line': 'S;SESSION;INV001;1003;SITE01;0;0;1;ART004;EMP001;A;UN;0;ZONE1;'
            }
        ]
        
        lines = processor.generate_lotecart_lines(adjustments, max_line_number=2000)
        
        assert len(lines) == 2
        
        for line in lines:
            parts = line.split(';')
            assert len(parts) >= 15
            assert parts[0] == 'S'  # Type ligne
            assert parts[14] == 'LOTECART'  # Numéro lot
            assert parts[7] == '2'  # Indicateur compte
            assert int(parts[5]) > 0  # Quantité > 0
            assert int(parts[6]) > 0  # Quantité réelle > 0
    
    def test_generate_lotecart_lines_invalid_reference(self, processor):
        """Test génération avec ligne de référence invalide"""
        adjustments = [
            {
                'CODE_ARTICLE': 'ART002',
                'is_new_lotecart': True,
                'reference_line': 'S;SHORT;LINE'  # Ligne trop courte
            }
        ]
        
        lines = processor.generate_lotecart_lines(adjustments)
        assert len(lines) == 0  # Aucune ligne générée
    
    def test_generate_lotecart_lines_no_adjustments(self, processor):
        """Test génération sans ajustements"""
        lines = processor.generate_lotecart_lines([])
        assert len(lines) == 0
    
    def test_validate_lotecart_processing_success(self, processor, tmp_path):
        """Test validation réussie du traitement LOTECART"""
        # Créer un fichier de test avec lignes LOTECART
        test_file = tmp_path / "test_final.csv"
        content = """E;HEADER;LINE
L;INVENTORY;LINE
S;SESSION;INV001;1000;SITE01;100;100;2;ART001;EMP001;A;UN;0;ZONE1;LOTECART
S;SESSION;INV001;1001;SITE01;25;25;2;ART002;EMP001;A;UN;0;ZONE1;LOTECART
"""
        test_file.write_text(content, encoding='utf-8')
        
        result = processor.validate_lotecart_processing(str(test_file), expected_lotecart_count=2)
        
        assert result['success'] == True
        assert result['lotecart_lines_found'] == 2
        assert result['correct_indicators'] == 2
        assert len(result['issues']) == 0
    
    def test_validate_lotecart_processing_insufficient_lines(self, processor, tmp_path):
        """Test validation avec nombre insuffisant de lignes LOTECART"""
        test_file = tmp_path / "test_final.csv"
        content = """E;HEADER;LINE
S;SESSION;INV001;1000;SITE01;100;100;2;ART001;EMP001;A;UN;0;ZONE1;LOTECART
"""
        test_file.write_text(content, encoding='utf-8')
        
        result = processor.validate_lotecart_processing(str(test_file), expected_lotecart_count=3)
        
        assert result['success'] == False
        assert result['lotecart_lines_found'] == 1
        assert len(result['issues']) > 0
        assert any('insuffisant' in issue.lower() for issue in result['issues'])
    
    def test_validate_lotecart_processing_incorrect_indicators(self, processor, tmp_path):
        """Test validation avec indicateurs incorrects"""
        test_file = tmp_path / "test_final.csv"
        content = """E;HEADER;LINE
S;SESSION;INV001;1000;SITE01;100;100;1;ART001;EMP001;A;UN;0;ZONE1;LOTECART
S;SESSION;INV001;1001;SITE01;25;25;2;ART002;EMP001;A;UN;0;ZONE1;LOTECART
"""
        test_file.write_text(content, encoding='utf-8')
        
        result = processor.validate_lotecart_processing(str(test_file), expected_lotecart_count=2)
        
        assert result['success'] == False
        assert result['lotecart_lines_found'] == 2
        assert result['correct_indicators'] == 1  # Seulement une ligne avec indicateur correct
        assert any('indicateurs incorrects' in issue.lower() for issue in result['issues'])
    
    def test_get_lotecart_summary(self, processor, sample_completed_df):
        """Test génération du résumé LOTECART"""
        candidates = processor.detect_lotecart_candidates(sample_completed_df)
        
        # Simuler des ajustements
        adjustments = [
            {'CODE_ARTICLE': 'ART002', 'QUANTITE_CORRIGEE': 25},
            {'CODE_ARTICLE': 'ART004', 'QUANTITE_CORRIGEE': 10}
        ]
        
        summary = processor.get_lotecart_summary(candidates, adjustments)
        
        assert summary['candidates_detected'] == 2
        assert summary['adjustments_created'] == 2
        assert summary['total_quantity'] == 35.0  # 25 + 10
        assert summary['inventories_affected'] == 1  # Un seul inventaire
        assert 'INV001' in summary['articles_by_inventory']
        assert len(summary['articles_by_inventory']['INV001']) == 2
        assert 'processing_timestamp' in summary
    
    def test_get_lotecart_summary_empty(self, processor):
        """Test génération du résumé avec données vides"""
        empty_df = pd.DataFrame()
        summary = processor.get_lotecart_summary(empty_df, [])
        
        assert summary['candidates_detected'] == 0
        assert summary['adjustments_created'] == 0
        assert summary['total_quantity'] == 0
        assert summary['inventories_affected'] == 0
        assert summary['articles_by_inventory'] == {}
    
    def test_reset_counter(self, processor):
        """Test remise à zéro du compteur"""
        processor.lotecart_counter = 5
        processor.reset_counter()
        assert processor.lotecart_counter == 0
    
    @patch('services.lotecart_processor.logger')
    def test_logging_behavior(self, mock_logger, processor, sample_completed_df):
        """Test du comportement de logging"""
        processor.detect_lotecart_candidates(sample_completed_df)
        
        # Vérifier que les logs appropriés sont appelés
        mock_logger.info.assert_called()
        
        # Vérifier le format des messages de log
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any('candidats LOTECART détectés' in str(call) for call in log_calls)
    
    def test_error_handling_in_detection(self, processor):
        """Test gestion d'erreurs dans la détection"""
        # DataFrame avec structure incorrecte
        invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        # Ne doit pas lever d'exception
        result = processor.detect_lotecart_candidates(invalid_df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_error_handling_in_adjustment_creation(self, processor):
        """Test gestion d'erreurs dans la création d'ajustements"""
        # Candidats avec structure incorrecte
        invalid_candidates = pd.DataFrame({'wrong_column': [1, 2, 3]})
        invalid_original = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        # Ne doit pas lever d'exception
        result = processor.create_lotecart_adjustments(invalid_candidates, invalid_original)
        assert isinstance(result, list)
        assert len(result) == 0