#!/usr/bin/env python3
"""
Analyse simple des fichiers LOTECART
"""
import pandas as pd
import os

def analyze_lotecart_files():
    """Analyse les fichiers pour vérifier les corrections LOTECART"""
    print("=== ANALYSE DES FICHIERS LOTECART ===")
    
    session_id = "4d334531"
    
    # 1. Analyser le template complété
    print("\n1. Template complété:")
    template_path = f"processed/completed_{session_id}_BKE02_BKE022508SES00000004_BKE022508INV00000008_{session_id}.xlsx"
    
    if os.path.exists(template_path):
        df = pd.read_excel(template_path)
        lotecart_candidates = df[(df['Quantité Théorique'] == 0) & (df['Quantité Réelle'] > 0)]
        
        print(f"✅ {len(lotecart_candidates)} candidats LOTECART:")
        for _, row in lotecart_candidates.iterrows():
            print(f"   - {row['Code Article']}: Théo={row['Quantité Théorique']}, Réel={row['Quantité Réelle']}")
    else:
        print(f"❌ Template non trouvé: {template_path}")
        return
    
    # 2. Analyser le fichier final
    print("\n2. Fichier final:")
    final_path = f"final/bke new_corrige_{session_id}.csv"
    
    if os.path.exists(final_path):
        # Analyser les lignes LOTECART
        lotecart_lines = []
        original_lines = []
        
        with open(final_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('S;'):
                    parts = line.strip().split(';')
                    if len(parts) > 14:
                        article = parts[8]
                        quantite = parts[5]
                        quantite_reelle_input = parts[6] if len(parts) > 6 else '0'
                        indicateur = parts[7]
                        numero_lot = parts[14]
                        
                        if 'LOTECART' in line:
                            lotecart_lines.append({
                                'ligne': i+1,
                                'article': article,
                                'quantite': quantite,
                                'quantite_reelle_input': quantite_reelle_input,
                                'indicateur': indicateur,
                                'lot': numero_lot
                            })
                        elif article in ['37CV045045GAM', '37CV150150GAM']:
                            original_lines.append({
                                'ligne': i+1,
                                'article': article,
                                'quantite': quantite,
                                'quantite_reelle_input': quantite_reelle_input,
                                'indicateur': indicateur,
                                'lot': numero_lot
                            })
        
        print(f"✅ {len(lotecart_lines)} lignes LOTECART trouvées:")
        for line in lotecart_lines:
            status = "✅" if line['indicateur'] == '2' else "❌"
            print(f"   {status} {line['article']}: Qté Théo={line['quantite']}, Qté Réelle Input={line['quantite_reelle_input']}, Indicateur={line['indicateur']}, Lot={line['lot']}")
        
        print(f"\n✅ {len(original_lines)} lignes originales des articles LOTECART:")
        for line in original_lines:
            status = "✅" if line['indicateur'] == '2' else "❌"
            print(f"   {status} {line['article']}: Qté Théo={line['quantite']}, Qté Réelle Input={line['quantite_reelle_input']}, Indicateur={line['indicateur']}, Lot={line['lot']}")
        
        # 3. Vérification globale
        print("\n=== VÉRIFICATION ===")
        
        # Compter les indicateurs corrects
        correct_lotecart = sum(1 for line in lotecart_lines if line['indicateur'] == '2')
        correct_original = sum(1 for line in original_lines if line['indicateur'] == '2')
        
        print(f"Lignes LOTECART avec indicateur=2: {correct_lotecart}/{len(lotecart_lines)}")
        print(f"Lignes originales avec indicateur=2: {correct_original}/{len(original_lines)}")
        
        # Vérifier les quantités
        expected_quantities = {'37CV045045GAM': 3, '37CV150150GAM': 2}
        
        for line in lotecart_lines:
            expected_qty = expected_quantities.get(line['article'])
            if expected_qty and int(line['quantite']) == expected_qty:
                print(f"✅ {line['article']}: Quantité correcte ({line['quantite']})")
            else:
                print(f"❌ {line['article']}: Quantité incorrecte (attendu: {expected_qty}, trouvé: {line['quantite']})")
        
        # Résultat final
        total_correct = correct_lotecart + correct_original
        total_lines = len(lotecart_lines) + len(original_lines)
        
        if total_correct == total_lines and len(lotecart_lines) >= len(lotecart_candidates):
            print("\n🎉 SUCCÈS: Toutes les corrections LOTECART sont appliquées correctement!")
        else:
            print(f"\n⚠️  PROBLÈME: {total_correct}/{total_lines} lignes ont l'indicateur correct")
    
    else:
        print(f"❌ Fichier final non trouvé: {final_path}")

if __name__ == "__main__":
    analyze_lotecart_files()