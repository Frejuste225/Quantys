#!/usr/bin/env python3
"""
Script d'analyse amélioré pour vérifier les corrections LOTECART
"""
import pandas as pd
import os
import sys

def analyze_lotecart_files():
    """Analyse les fichiers pour vérifier les corrections LOTECART"""
    print("=== ANALYSE DÉTAILLÉE DES FICHIERS LOTECART ===")
    
    # Chercher la session la plus récente
    session_folders = []
    for folder in ['processed', 'final']:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if 'completed_' in filename or '_corrige_' in filename:
                    # Extraire l'ID de session
                    parts = filename.split('_')
                    for part in parts:
                        if len(part) == 8 and part.isalnum():
                            session_folders.append(part)
    
    if not session_folders:
        print("❌ Aucune session trouvée")
        return
    
    # Prendre la session la plus récente
    session_id = max(set(session_folders))
    print(f"📋 Analyse de la session: {session_id}")
    
    # 1. Analyser le template complété
    print("\n1. 📊 TEMPLATE COMPLÉTÉ:")
    template_files = [f for f in os.listdir('processed') if f.startswith(f'completed_{session_id}')]
    
    if not template_files:
        print(f"❌ Template complété non trouvé pour session {session_id}")
        return
    
    template_path = os.path.join('processed', template_files[0])
    print(f"   Fichier: {template_files[0]}")
    
    df = pd.read_excel(template_path)
    
    # Identifier les candidats LOTECART (quantité théorique = 0, quantité réelle > 0)
    lotecart_candidates = df[(df['Quantité Théorique'] == 0) & (df['Quantité Réelle'] > 0)]
    
    print(f"✅ {len(lotecart_candidates)} candidats LOTECART détectés:")
    for _, row in lotecart_candidates.iterrows():
        print(f"   - {row['Code Article']}: Théo=0, Réel={row['Quantité Réelle']}, Lot='{row['Numéro Lot']}'")
    
    # 2. Analyser le fichier final
    print("\n2. 📄 FICHIER FINAL:")
    final_files = [f for f in os.listdir('final') if f.endswith(f'_corrige_{session_id}.csv')]
    
    if not final_files:
        print(f"❌ Fichier final non trouvé pour session {session_id}")
        return
    
    final_path = os.path.join('final', final_files[0])
    print(f"   Fichier: {final_files[0]}")
    
    # Analyser toutes les lignes du fichier final
    lotecart_lines = []
    original_zero_lines = []
    all_s_lines = []
    
    with open(final_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.startswith('S;'):
                parts = line.strip().split(';')
                if len(parts) > 14:
                    article = parts[8]
                    qte_theo = parts[5]
                    qte_reelle = parts[6] if len(parts) > 6 else '0'
                    indicateur = parts[7] if len(parts) > 7 else '1'
                    numero_lot = parts[14]
                    
                    line_info = {
                        'ligne': i,
                        'article': article,
                        'qte_theo': qte_theo,
                        'qte_reelle': qte_reelle,
                        'indicateur': indicateur,
                        'lot': numero_lot,
                        'is_lotecart': 'LOTECART' in line
                    }
                    
                    all_s_lines.append(line_info)
                    
                    if 'LOTECART' in line:
                        lotecart_lines.append(line_info)
                    
                    # Vérifier les articles qui étaient candidats LOTECART
                    for _, candidate in lotecart_candidates.iterrows():
                        if candidate['Code Article'] == article:
                            original_zero_lines.append(line_info)
                            break
    
    print(f"✅ {len(all_s_lines)} lignes S au total dans le fichier final")
    print(f"✅ {len(lotecart_lines)} lignes avec LOTECART:")
    
    for line in lotecart_lines:
        status = "✅" if line['indicateur'] == '2' else "❌"
        print(f"   {status} Ligne {line['ligne']}: {line['article']} - Théo={line['qte_theo']}, Réel={line['qte_reelle']}, Indicateur={line['indicateur']}")
    
    print(f"\n✅ {len(original_zero_lines)} lignes pour les articles candidats LOTECART:")
    for line in original_zero_lines:
        status = "✅" if line['indicateur'] == '2' else "❌"
        lotecart_marker = " [LOTECART]" if line['is_lotecart'] else ""
        print(f"   {status} Ligne {line['ligne']}: {line['article']} - Théo={line['qte_theo']}, Réel={line['qte_reelle']}, Indicateur={line['indicateur']}, Lot={line['lot']}{lotecart_marker}")
    
    # 3. Vérifications détaillées
    print("\n3. 🔍 VÉRIFICATIONS DÉTAILLÉES:")
    
    # Vérifier que chaque candidat LOTECART a bien une ligne correspondante
    for _, candidate in lotecart_candidates.iterrows():
        article = candidate['Code Article']
        qte_reelle_attendue = candidate['Quantité Réelle']
        
        # Chercher les lignes correspondantes dans le fichier final
        matching_lines = [line for line in all_s_lines if line['article'] == article]
        
        print(f"\n   Article {article} (attendu: {qte_reelle_attendue}):")
        
        if not matching_lines:
            print(f"   ❌ Aucune ligne trouvée dans le fichier final")
            continue
        
        # Vérifier s'il y a une ligne LOTECART
        lotecart_line = next((line for line in matching_lines if line['is_lotecart']), None)
        
        if lotecart_line:
            if lotecart_line['qte_theo'] == str(int(qte_reelle_attendue)) and lotecart_line['qte_reelle'] == str(int(qte_reelle_attendue)):
                print(f"   ✅ Ligne LOTECART correcte: Théo={lotecart_line['qte_theo']}, Réel={lotecart_line['qte_reelle']}")
            else:
                print(f"   ❌ Ligne LOTECART incorrecte: Théo={lotecart_line['qte_theo']}, Réel={lotecart_line['qte_reelle']} (attendu: {qte_reelle_attendue})")
        else:
            print(f"   ❌ Aucune ligne LOTECART créée")
        
        # Vérifier les lignes originales (quantité théorique = 0)
        zero_lines = [line for line in matching_lines if line['qte_theo'] == '0']
        if zero_lines:
            for zero_line in zero_lines:
                if zero_line['indicateur'] == '2':
                    print(f"   ✅ Ligne originale correctement mise à jour: Indicateur=2")
                else:
                    print(f"   ❌ Ligne originale non mise à jour: Indicateur={zero_line['indicateur']} (devrait être 2)")
    
    # 4. Résumé final
    print("\n4. 📋 RÉSUMÉ:")
    
    expected_lotecart = len(lotecart_candidates)
    actual_lotecart = len(lotecart_lines)
    correct_indicators = sum(1 for line in lotecart_lines if line['indicateur'] == '2')
    
    print(f"   • Candidats LOTECART attendus: {expected_lotecart}")
    print(f"   • Lignes LOTECART créées: {actual_lotecart}")
    print(f"   • Indicateurs corrects: {correct_indicators}/{actual_lotecart}")
    
    if actual_lotecart >= expected_lotecart and correct_indicators == actual_lotecart:
        print("\n🎉 SUCCÈS: Toutes les corrections LOTECART sont appliquées correctement!")
    else:
        print("\n⚠️  PROBLÈMES DÉTECTÉS:")
        if actual_lotecart < expected_lotecart:
            print(f"   - Lignes LOTECART manquantes: {expected_lotecart - actual_lotecart}")
        if correct_indicators < actual_lotecart:
            print(f"   - Indicateurs incorrects: {actual_lotecart - correct_indicators}")

if __name__ == "__main__":
    analyze_lotecart_files()