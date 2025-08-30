#!/usr/bin/env python3
"""
Test complet du processus LOTECART avec simulation
"""
import sys
import os
sys.path.append('.')

import pandas as pd
import json
from datetime import datetime
from app import processor, session_service

def create_test_template():
    """Crée un template de test avec des cas LOTECART"""
    print("=== CRÉATION DU TEMPLATE DE TEST ===")
    
    # Données de test avec des cas LOTECART
    test_data = [
        {
            'Numéro Session': 'BKE022508SES00000004',
            'Numéro Inventaire': 'BKE022508INV00000008',
            'Code Article': '37CV045045GAM',
            'Statut Article': 'AM',
            'Quantité Théorique': 0,  # LOTECART candidat
            'Quantité Réelle': 3,     # Quantité trouvée
            'Numéro Lot': '',
            'Unites': 'UN',
            'Depots': 'BKE2',
            'Emplacements': 'BKE02'
        },
        {
            'Numéro Session': 'BKE022508SES00000004',
            'Numéro Inventaire': 'BKE022508INV00000008',
            'Code Article': '37CV150150GAM',
            'Statut Article': 'AM',
            'Quantité Théorique': 0,  # LOTECART candidat
            'Quantité Réelle': 2,     # Quantité trouvée
            'Numéro Lot': '',
            'Unites': 'UN',
            'Depots': 'BKE2',
            'Emplacements': 'BKE02'
        },
        {
            'Numéro Session': 'BKE022508SES00000004',
            'Numéro Inventaire': 'BKE022508INV00000008',
            'Code Article': '32BTCS2GAM',
            'Statut Article': 'AM',
            'Quantité Théorique': 100,  # Article normal
            'Quantité Réelle': 95,      # Écart négatif
            'Numéro Lot': 'LOT311223',
            'Unites': 'UN',
            'Depots': 'BKE2',
            'Emplacements': 'BKE02'
        }
    ]
    
    df = pd.DataFrame(test_data)
    test_file = 'test_template_completed.xlsx'
    df.to_excel(test_file, index=False)
    
    print(f"✅ Template de test créé: {test_file}")
    return test_file, df

def simulate_full_process():
    """Simule le processus complet avec des données de test"""
    print("=== SIMULATION DU PROCESSUS COMPLET ===")
    
    # Créer une session de test
    session_id = "test_" + str(datetime.now().strftime("%H%M%S"))
    
    # Simuler les données originales
    original_data = [
        ['S', 'BKE022508SES00000004', 'BKE022508INV00000008', '1000', 'BKE02', '0', '0', '1', '37CV045045GAM', 'BKE02', 'AM', 'UN', '0', 'BKE2', ''],
        ['S', 'BKE022508SES00000004', 'BKE022508INV00000008', '2000', 'BKE02', '0', '0', '1', '37CV150150GAM', 'BKE02', 'AM', 'UN', '0', 'BKE2', ''],
        ['S', 'BKE022508SES00000004', 'BKE022508INV00000008', '3000', 'BKE02', '100', '0', '1', '32BTCS2GAM', 'BKE02', 'AM', 'UN', '842', 'BKE2', 'LOT311223'],
    ]
    
    columns = ['TYPE_LIGNE', 'NUMERO_SESSION', 'NUMERO_INVENTAIRE', 'RANG', 'SITE', 'QUANTITE', 'QUANTITE_REELLE_IN_INPUT', 'INDICATEUR_COMPTE', 'CODE_ARTICLE', 'EMPLACEMENT', 'STATUT', 'UNITE', 'VALEUR', 'ZONE_PK', 'NUMERO_LOT']
    
    original_df = pd.DataFrame(original_data, columns=columns)
    original_df['QUANTITE'] = pd.to_numeric(original_df['QUANTITE'])
    original_df['original_s_line_raw'] = original_df.apply(lambda row: ';'.join(row.astype(str)), axis=1)
    
    # Ajouter les informations de lot
    original_df['Date_Lot'] = None
    original_df['Type_Lot'] = 'unknown'
    
    # Headers simulés
    headers = [
        'E;BKE022508SES00000004;test depot conf;1;BKE02;;;;;;;;;;',
        'L;BKE022508SES00000004;BKE022508INV00000008;1;BKE02;;;;;;;;;;'
    ]
    
    # Initialiser la session dans le processeur
    processor.sessions[session_id] = {
        'original_df': original_df,
        'header_lines': headers
    }
    
    print(f"✅ Session {session_id} initialisée avec {len(original_df)} lignes")
    
    # Créer et traiter le template complété
    test_file, completed_df = create_test_template()
    
    try:
        # Traitement du fichier complété
        print(f"\n📊 Traitement du fichier complété...")
        processed_df = processor.process_completed_file(session_id, test_file)
        print(f"✅ {len(processed_df)} écarts détectés")
        
        # Distribution des écarts
        print(f"\n🔄 Distribution des écarts...")
        distributed_df = processor.distribute_discrepancies(session_id, 'FIFO')
        print(f"✅ {len(distributed_df)} ajustements créés")
        
        # Afficher les ajustements LOTECART
        lotecart_adjustments = distributed_df[distributed_df['TYPE_LOT'] == 'lotecart']
        print(f"\n📋 Ajustements LOTECART ({len(lotecart_adjustments)}):")
        for _, adj in lotecart_adjustments.iterrows():
            print(f"   - {adj['CODE_ARTICLE']}: Qté={adj['QUANTITE_CORRIGEE']}, Lot={adj['NUMERO_LOT']}")
        
        # Génération du fichier final
        print(f"\n📄 Génération du fichier final...")
        final_file = processor.generate_final_file(session_id)
        print(f"✅ Fichier final généré: {final_file}")
        
        # Analyse du fichier final
        print(f"\n🔍 ANALYSE DU FICHIER FINAL:")
        analyze_final_file(final_file, completed_df)
        
    finally:
        # Nettoyage
        if os.path.exists(test_file):
            os.remove(test_file)
        if session_id in processor.sessions:
            del processor.sessions[session_id]

def analyze_final_file(final_file, completed_df):
    """Analyse détaillée du fichier final généré"""
    
    # Candidats LOTECART du template
    lotecart_candidates = completed_df[(completed_df['Quantité Théorique'] == 0) & (completed_df['Quantité Réelle'] > 0)]
    
    print(f"   Candidats LOTECART attendus: {len(lotecart_candidates)}")
    
    # Analyser le fichier final
    with open(final_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    s_lines = [line for line in lines if line.startswith('S;')]
    lotecart_lines = [line for line in s_lines if 'LOTECART' in line]
    
    print(f"   Lignes S totales: {len(s_lines)}")
    print(f"   Lignes LOTECART trouvées: {len(lotecart_lines)}")
    
    # Analyser chaque ligne LOTECART
    for i, line in enumerate(lotecart_lines):
        parts = line.strip().split(';')
        if len(parts) > 14:
            article = parts[8]
            qte_theo = parts[5]
            qte_reelle = parts[6] if len(parts) > 6 else '0'
            indicateur = parts[7] if len(parts) > 7 else '1'
            
            # Trouver le candidat correspondant
            candidate = lotecart_candidates[lotecart_candidates['Code Article'] == article]
            if not candidate.empty:
                expected_qty = candidate.iloc[0]['Quantité Réelle']
                
                print(f"   Ligne LOTECART {i+1}: {article}")
                print(f"     - Quantité attendue: {expected_qty}")
                print(f"     - Quantité théorique: {qte_theo}")
                print(f"     - Quantité réelle: {qte_reelle}")
                print(f"     - Indicateur: {indicateur}")
                
                if qte_theo == str(int(expected_qty)) and qte_reelle == str(int(expected_qty)) and indicateur == '2':
                    print(f"     ✅ CORRECT")
                else:
                    print(f"     ❌ INCORRECT")
    
    # Résumé final
    correct_lotecart = 0
    for line in lotecart_lines:
        parts = line.strip().split(';')
        if len(parts) > 7 and parts[7] == '2':
            correct_lotecart += 1
    
    print(f"\n📊 RÉSUMÉ FINAL:")
    print(f"   • Candidats LOTECART: {len(lotecart_candidates)}")
    print(f"   • Lignes LOTECART créées: {len(lotecart_lines)}")
    print(f"   • Indicateurs corrects: {correct_lotecart}/{len(lotecart_lines)}")
    
    if len(lotecart_lines) >= len(lotecart_candidates) and correct_lotecart == len(lotecart_lines):
        print("\n🎉 SUCCÈS: Le processus LOTECART fonctionne correctement!")
    else:
        print("\n⚠️  PROBLÈMES DÉTECTÉS dans le processus LOTECART")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'simulate':
        simulate_full_process()
    else:
        analyze_lotecart_files()