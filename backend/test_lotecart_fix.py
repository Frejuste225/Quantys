#!/usr/bin/env python3
"""
Script de test pour vérifier les corrections LOTECART
"""
import sys
import os
sys.path.append('.')

import pandas as pd
from app import processor

def test_lotecart_processing():
    """Test du traitement des LOTECART"""
    print("=== TEST DES CORRECTIONS LOTECART ===")
    
    # Session ID de test
    session_id = "4d334531"
    
    try:
        # 1. Vérifier les données du template complété
        print("\n1. Analyse du template complété...")
        template_path = f"processed/completed_{session_id}_BKE02_BKE022508SES00000004_BKE022508INV00000008_{session_id}.xlsx"
        
        if not os.path.exists(template_path):
            print(f"❌ Template non trouvé: {template_path}")
            return
        
        df = pd.read_excel(template_path)
        
        # Identifier les candidats LOTECART
        lotecart_candidates = df[(df['Quantité Théorique'] == 0) & (df['Quantité Réelle'] > 0)]
        print(f"✅ {len(lotecart_candidates)} candidats LOTECART trouvés:")
        
        for _, row in lotecart_candidates.iterrows():
            print(f"   - {row['Code Article']}: Qté Théo={row['Quantité Théorique']}, Qté Réelle={row['Quantité Réelle']}")
        
        # 2. Simuler le traitement
        print("\n2. Simulation du traitement...")
        
        # Vérifier si la session existe dans le processeur
        if session_id not in processor.sessions:
            print(f"❌ Session {session_id} non trouvée dans le processeur")
            return
        
        session_data = processor.sessions[session_id]
        
        if 'distributed_df' not in session_data:
            print("❌ Données distribuées non trouvées")
            return
        
        distributed_df = session_data['distributed_df']
        
        # Analyser les ajustements LOTECART
        lotecart_adjustments = distributed_df[distributed_df['TYPE_LOT'] == 'lotecart']
        print(f"✅ {len(lotecart_adjustments)} ajustements LOTECART créés:")
        
        for _, row in lotecart_adjustments.iterrows():
            print(f"   - {row['CODE_ARTICLE']}: Quantité={row['QUANTITE_CORRIGEE']}, Lot={row['NUMERO_LOT']}")
        
        # 3. Vérifier le fichier final
        print("\n3. Analyse du fichier final...")
        final_path = f"final/bke new_corrige_{session_id}.csv"
        
        if not os.path.exists(final_path):
            print(f"❌ Fichier final non trouvé: {final_path}")
            return
        
        # Compter les lignes LOTECART dans le fichier final
        lotecart_lines = []
        with open(final_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if 'LOTECART' in line:
                    parts = line.strip().split(';')
                    if len(parts) > 14:
                        article = parts[8] if len(parts) > 8 else 'N/A'
                        quantite = parts[5] if len(parts) > 5 else 'N/A'
                        quantite_reelle_input = parts[6] if len(parts) > 6 else 'N/A'
                        indicateur = parts[7] if len(parts) > 7 else 'N/A'
                        lotecart_lines.append({
                            'ligne': i+1,
                            'article': article,
                            'quantite': quantite,
                            'quantite_reelle_input': quantite_reelle_input,
                            'indicateur_compte': indicateur
                        })
        
        print(f"✅ {len(lotecart_lines)} lignes LOTECART dans le fichier final:")
        for line_info in lotecart_lines:
            status = "✅" if line_info['indicateur_compte'] == '2' else "❌"
            print(f"   {status} Ligne {line_info['ligne']}: {line_info['article']} - Qté Théo={line_info['quantite']} - Qté Réelle={line_info['quantite_reelle_input']} - Indicateur={line_info['indicateur_compte']}")
        
        # 4. Vérification des lignes originales avec quantité théorique 0
        print("\n4. Vérification des lignes originales avec quantité théorique 0...")
        
        # Chercher les lignes avec quantité théorique 0 dans le fichier final
        zero_qty_lines = []
        with open(final_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('S;'):
                    parts = line.strip().split(';')
                    if len(parts) > 8:
                        article = parts[8]
                        quantite = parts[5]
                        quantite_reelle_input = parts[6] if len(parts) > 6 else 'N/A'
                        indicateur = parts[7] if len(parts) > 7 else 'N/A'
                        
                        # Vérifier si c'est un des articles candidats LOTECART
                        for _, candidate in lotecart_candidates.iterrows():
                            if candidate['Code Article'] == article:
                                zero_qty_lines.append({
                                    'ligne': i+1,
                                    'article': article,
                                    'quantite': quantite,
                                    'quantite_reelle_input': quantite_reelle_input,
                                    'indicateur_compte': indicateur
                                })
                                break
        
        print(f"✅ {len(zero_qty_lines)} lignes originales avec quantité théorique 0:")
        for line_info in zero_qty_lines:
            status = "✅" if line_info['indicateur_compte'] == '2' else "❌"
            print(f"   {status} Ligne {line_info['ligne']}: {line_info['article']} - Qté Théo={line_info['quantite']} - Qté Réelle={line_info['quantite_reelle_input']} - Indicateur={line_info['indicateur_compte']}")
        
        # 5. Résumé
        print("\n=== RÉSUMÉ ===")
        total_lotecart_expected = len(lotecart_candidates)
        total_lotecart_created = len(lotecart_lines)
        correct_indicators = sum(1 for line in lotecart_lines if line['indicateur_compte'] == '2')
        
        print(f"Candidats LOTECART attendus: {total_lotecart_expected}")
        print(f"Lignes LOTECART créées: {total_lotecart_created}")
        print(f"Indicateurs corrects (=2): {correct_indicators}/{total_lotecart_created}")
        
        if correct_indicators == total_lotecart_created and total_lotecart_created >= total_lotecart_expected:
            print("✅ TOUS LES TESTS PASSENT - Les corrections LOTECART fonctionnent correctement!")
        else:
            print("❌ PROBLÈMES DÉTECTÉS - Les corrections nécessitent des ajustements")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lotecart_processing()