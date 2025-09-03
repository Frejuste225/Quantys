import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from services.file_processor import FileProcessorService
from datetime import datetime, date

class TestFileProcessorService:
    """Tests pour FileProcessorService"""
    
    @pytest.fixture
    def processor(self):
        """Instance du service de traitement de fichiers"""
        with patch('services.file_processor.config_service') as mock_config:
            # Configuration mockée
            mock_config.get_sage_columns.return_value = {
                'TYPE_LIGNE': 0,
                'NUMERO_SESSION': 1,
                'NUMERO_INVENTAIRE': 2,
                'RANG': 3,
                'SITE': 4,
                'QUANTITE': 5,
                'QUANTITE_REELLE_IN_INPUT': 6,
                'INDICATEUR_COMPTE': 7,
                'CODE_ARTICLE': 8,
                'EMPLACEMENT': 9,
                'STATUT': 10,
                'UNITE': 11,
                'VALEUR': 12,
                'ZONE_PK': 13,
                'NUMERO_LOT': 14,
            }
            mock_config.get_validation_config.return_value = {
                'required_line_types': ['E', 'L', 'S'],
                'min_columns': 15,
                'max_file_size_mb': 16
            }
            mock_config.get_processing_config.return_value = {
                'aggregation_keys': ['CODE_ARTICLE', 'STATUT', 'EMPLACEMENT', 'ZONE_PK', 'UNITE']
            }
            mock_config.get_lot_patterns.return_value = {
                'type1_pattern': r'^([A-Z0-9]{3,4})(\d{6})(\d+)$',
                'type2_pattern': r'^LOT(\d{6})$'
            }
            
            return FileProcessorService()
    
    @pytest.fixture
    def sample_csv_content(self):
        """Contenu CSV Sage X3 valide"""
        return """E;BKE022508SES00000003;test depot conf;1;BKE02;;;;;;;;;;
L;BKE022508SES00000003;BKE022508INV00000006;1;BKE02;;;;;;;;;;
S;BKE022508SES00000003;BKE022508INV00000006;1000;BKE02;100;0;1;ART001;EMP001;A;UN;0;ZONE1;LOT123456;
S;BKE022508SES00000003;BKE022508INV00000006;1001;BKE02;50;0;1;ART002;EMP001;A;UN;0;ZONE1;CPKU070725001;
S;BKE022508SES00000003;BKE022508INV00000006;1002;BKE02;0;0;1;ART003;EMP001;A;UN;0;ZONE1;;"""
    
    @pytest.fixture
    def sample_dataframe(self):
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
    
    def test_detect_file_format_csv(self, processor, tmp_path):
        """Test détection de format CSV"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("E;header\nS;data;line", encoding='utf-8')
        
        is_valid, message, info = processor.detect_file_format(str(csv_file))
        
        assert is_valid == True
        assert "détecté" in message.lower()
        assert 'total_lines' in info
        assert 'e_lines' in info
        assert 's_lines' in info
    
    def test_detect_file_format_xlsx(self, processor, tmp_path):
        """Test détection de format XLSX"""
        # Créer un fichier Excel simple
        xlsx_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({'col1': ['E', 'S'], 'col2': ['header', 'data']})
        df.to_excel(xlsx_file, index=False, header=False)
        
        is_valid, message, info = processor.detect_file_format(str(xlsx_file))
        
        assert is_valid == True
        assert "détecté" in message.lower()
        assert 'total_rows' in info
        assert 'sample_data' in info
    
    def test_detect_file_format_unsupported(self, processor, tmp_path):
        """Test détection de format non supporté"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some text")
        
        is_valid, message, info = processor.detect_file_format(str(txt_file))
        
        assert is_valid == False
        assert "non supportée" in message.lower()
    
    def test_extract_date_from_lot_type1(self, processor):
        """Test extraction de date pour lot type 1"""
        lot_number = "CPKU070725001"
        date_result, lot_type = processor._extract_date_from_lot(lot_number)
        
        assert date_result is not None
        assert lot_type == "type1"
        assert date_result.day == 7
        assert date_result.month == 7
        assert date_result.year == 2025
    
    def test_extract_date_from_lot_type2(self, processor):
        """Test extraction de date pour lot type 2"""
        lot_number = "LOT311224"
        date_result, lot_type = processor._extract_date_from_lot(lot_number)
        
        assert date_result is not None
        assert lot_type == "type2"
        assert date_result.day == 31
        assert date_result.month == 12
        assert date_result.year == 2024
    
    def test_extract_date_from_lot_unknown(self, processor):
        """Test extraction de date pour lot inconnu"""
        lot_number = "UNKNOWN_LOT"
        date_result, lot_type = processor._extract_date_from_lot(lot_number)
        
        assert date_result is None
        assert lot_type == "unknown"
    
    def test_extract_date_from_lot_empty(self, processor):
        """Test extraction de date pour lot vide"""
        date_result, lot_type = processor._extract_date_from_lot("")
        
        assert date_result is None
        assert lot_type == "unknown"
    
    def test_extract_inventory_date(self, processor):
        """Test extraction de date d'inventaire"""
        numero_inventaire = "BKE022508INV00000006"
        session_timestamp = datetime(2024, 12, 15)
        
        inventory_date = processor._extract_inventory_date(numero_inventaire, session_timestamp)
        
        assert inventory_date is not None
        assert inventory_date.day == 25
        assert inventory_date.month == 8
        assert inventory_date.year == 2024
    
    def test_extract_inventory_date_invalid(self, processor):
        """Test extraction de date d'inventaire invalide"""
        numero_inventaire = "INVALID_FORMAT"
        session_timestamp = datetime(2024, 12, 15)
        
        inventory_date = processor._extract_inventory_date(numero_inventaire, session_timestamp)
        
        assert inventory_date is None
    
    def test_process_dataframe(self, processor, sample_dataframe):
        """Test traitement du DataFrame"""
        original_lines = [
            'S;BKE022508SES00000003;BKE022508INV00000006;1000;BKE02;100;0;1;ART001;EMP001;A;UN;0;ZONE1;LOT123456',
            'S;BKE022508SES00000003;BKE022508INV00000006;1001;BKE02;50;0;1;ART002;EMP001;A;UN;0;ZONE1;CPKU070725001',
            'S;BKE022508SES00000003;BKE022508INV00000006;1002;BKE02;0;0;1;ART003;EMP001;A;UN;0;ZONE1;'
        ]
        
        processed_df = processor._process_dataframe(sample_dataframe, original_lines)
        
        # Vérifier que les colonnes ont été ajoutées
        assert 'Date_Lot' in processed_df.columns
        assert 'Type_Lot' in processed_df.columns
        assert 'original_s_line_raw' in processed_df.columns
        
        # Vérifier les types de lots détectés
        assert processed_df.iloc[1]['Type_Lot'] == 'type1'  # CPKU070725001
        assert processed_df.iloc[2]['Type_Lot'] == 'unknown'  # Lot vide
        
        # Vérifier les quantités
        assert processed_df['QUANTITE'].dtype == 'float64'
    
    def test_aggregate_data(self, processor, sample_dataframe):
        """Test agrégation des données"""
        aggregated_df = processor.aggregate_data(sample_dataframe)
        
        # Vérifier la structure du résultat
        assert not aggregated_df.empty
        assert 'Quantite_Theorique_Totale' in aggregated_df.columns
        assert 'Date_Min' in aggregated_df.columns
        assert 'Type_Lot_Prioritaire' in aggregated_df.columns
        
        # Vérifier l'agrégation
        total_quantity = aggregated_df['Quantite_Theorique_Totale'].sum()
        expected_total = sample_dataframe['QUANTITE'].sum()
        assert total_quantity == expected_total
    
    def test_aggregate_data_empty(self, processor):
        """Test agrégation avec DataFrame vide"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame vide"):
            processor.aggregate_data(empty_df)
    
    def test_get_priority_lot_type(self, processor):
        """Test détermination du type de lot prioritaire"""
        # Test avec différents types
        lot_types = ['unknown', 'type1', 'type2']
        priority_type = processor._get_priority_lot_type(lot_types)
        assert priority_type == 'type1'  # Plus haute priorité
        
        # Test avec seulement type2
        lot_types = ['unknown', 'type2']
        priority_type = processor._get_priority_lot_type(lot_types)
        assert priority_type == 'type2'
        
        # Test avec seulement unknown
        lot_types = ['unknown']
        priority_type = processor._get_priority_lot_type(lot_s Nonentory_date iassert inve
        ers == [] head   assertr
     sage d'erreur)  # Mesata, stinstance(dsert isas   
     alsecess == Fassert suc
             )
     
      imestampon_t sessi5,x), 1ke_xlstr(fa         s   xlsx_file(
ss_._proceprocessorate =  inventory_dta, headers,ccess, da su      
      ow()
   .netimeestamp = datim  session_t         
")
     xcel filenot an ete_text("x.wriake_xls  fx"
      / "fake.xls tmp_path sx =e_xl        fak valide
un Excelpas t i n'esn fichier quéer u       # Cr""
 X"ent XLSitemtras le daneurs n d'errestioest g """T):
       tmp_pathprocessor, (self, ngx_processig_in_xlsindlanor_hrrst_ef te  
    deone
  ate is N inventory_d     assert
   s == []ert header       assr
 reuge d'ertr)  # Messadata, sstance(inassert is       alse
 cess == Fert suc     ass    
   
    
        )pimestamon_tssi5, secsv_file), 1   str(        e(
 v_filcsrocess_ssor._pe = procery_datntoders, inveea, hccess, data su       
    )
    .now(tetimemestamp = daon_ti    sessi
           
 ire invalideContenu bina0\x00')  # xfe\x0xff\s(b'\bytefile.write_    csv_  sv"
  rrupted.cth / "coile = tmp_pa csv_fu
        CSV corromphierun ficer ré       # CSV"""
 itement Cle trareurs dans 'ern dstio""Test ge       "
 , tmp_path):sor procesng(self,essi_in_csv_prochandlingror_er   def test_)
    
 lled(rt_cas.asse_lot_patterngetig._conf       mocked()
     t_callnfig.assersing_cooceset_prck_config.g         mo)
   called(rt_g.assenfiion_coidatet_valig.gck_conf      mo)
      ed(ert_callassns.t_sage_columck_config.ge         mo   _once()
_calledonfig.assertig.reload_cconf  mock_     
               ()
  figonor.reload_crocess    p  
                  
COLUMN': 0} {'NEW_e =luns.return_vage_columig.get_sak_conf   moc      onfig:
   ock_c) as mice'_serv.config_processors.filech('service   with pat  n"""
   ioonfigurat de la centchargem""Test re"       ):
  processorself,ad_config( test_relo
    def    > 0
 errors)t len(sser   a    
 lid == Falset is_va      asser 
      
   file))cel_(explate(stremted_te_compleidatcessor.val= proors sage, err_valid, mes        is        
ex=False)
ind, xcel_file_excel(e.to  df     e.xlsx"
 "templat/ _path mpile = t  excel_f     
        
 ata)Frame(d= pd.Data       df       }
 s
  antemanqu # Colonnes         0],
   : [100, 5ique' Théor  'Quantité
          ART002'],01', '': ['ART0rticle 'Code A     
      = {     data s
   anquantenes mc colon aveelichier Excr un frée    # C    ""
nvalide"é iate complétemplation d'un t"Test valid" "
       h):p_pat, tm, processorinvalid(self_template_e_completed_validatest   def t= 0
    
 (errors) =rt len     asselower()
   in message.de" ali assert "v     rue
   == T_validsert is        as  

      excel_file))mplate(str(ted_tecompleate_alidsor.vroces errors = pd, message,     is_vali
   e)
        ndex=Falsel_file, iel(exco_exc        df.t
xlsx""template._path /  tmpe =_fil     excel       
   me(data)
 taFra df = pd.Da  }
       02']
      01', 'LOT0T0o Lot': ['LOumér        'N    ],
e': [95, 52éell'Quantité R          ,
   [100, 50]rique':ntité ThéoQua '        ],
   ', 'ART002'': ['ART001icle ArtCode          '
  * 2, ['INV001'] Inventaire':éro  'Num          
 '] * 2,001['SESSIONon': méro Sessi 'Nu         = {
   data       l de test
 er ExcehiCréer un fic     # """
   lidevacomplété late empion d'un talidatst v """Te   h):
    patr, tmp_sso, proce_valid(selfemplatecompleted_tvalidate_   def test_
   ))
  _path str(tmpt123',pty_df, 'tes(emplatee_tem.generatsorroces     p"):
       géegréne donnée aucutch="A, maueErrorses(Valh pytest.rai
        wit
        DataFrame()_df = pd.     emptys"""
   s videavec donnéee template ation dnér"Test gé     ""   path):
or, tmp_ocess(self, prpty_datamplate_em_teneratetest_geef    
    dx')
 with('.xlspath.endstemplate_  assert          ath)
 emplate_path.exists(t os.p     assert
       s not Nonee_path implatt teser   as
            )
                   mp_path)
     str(t           
   st123',te   '           _df, 
  gatedaggre                emplate(
.generate_trocessor_path = ptemplate            
           )
 gated_datae(aggreFram= pd.Dataf gated_d      aggre         }
   
      ']: ['SITE01    'Site'            001'],
INVIRE': ['NTA'NUMERO_INVE               ],
 SION001': ['SESsion' 'Numero_Ses       ,
        ': [100]_TotaleueriqheoQuantite_T        '
        '],TE': ['UN    'UNI            1'],
PK': ['ZONE 'ZONE_      ,
         ['EMP001']MENT': ACE'EMPL                '],
'A'STATUT': [           01'],
     T0 ['ARDE_ARTICLE':  'CO       = {
       d_data   aggregate        
  simulée sgéess agrénnée # Do                  
 
    gneune liner tourc[:1]  # Reataframe.ilople_d= samurn_value s.rett_lotock_ge m        t_lots:
   ) as mock_ger_article'l_lots_foina_orig, '_getcessorject(proch.obpat    with    ervice
 n sessiok de la s# Moc"
        ""te Excelde templa génération Test"       ""_path):
 ame, tmpafre_datsamplrocessor, ead_excel, pelf, mock_re_template(sneratf test_ge
    de)l'xceead_essor.pd.re_proceervices.filch('s    @patwn'
    
 'unknoype ==y_tpriorit     assert pes)
   ty