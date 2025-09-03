#!/usr/bin/env python3
"""
Script de test pour vérifier que les quantités théoriques ajustées 
et les quantités réelles sont correctement appliquées dans le fichier final.
"""

import pandas as pd
import os
import tempfile
from datetime import datetime

# Simuler les données pour le test
def create_test_data():
    """Crée des données de test simulées"""
    
    # 1. DataFrame original (données Sage X3)
    original_data = {
        'TYPE_LIGNE': ['S', 'S', 'S', 'S'],
        'NUMERO_SESSION': ['SESSION001'] * 4,
        'NUMERO_INVENTAIRE': ['INV001'] * 4,
        'RANG': [1000, 1001, 1002, 1003],
        'SITE': ['SITE01'] * 4,
        'QUANTITE': [100.0, 50.0, 0.0, 75.0],  # Quantités théoriques originales
        'QUANTITE_REELLE_IN_INPUT': [0.0] * 4,
        'INDICATEUR_COMPTE': [1] * 4,
        'CODE_ARTICLE': ['ART001', 'ART002', 'ART003', 'ART004'],
        'EMPLACEMENT': ['EMP001'] * 4,
        'STATUT': ['A'] * 4,
        'UNITE': ['UN'] * 4,
        'VALEUR': [0.0] * 4,
        'ZONE_PK': ['ZONE1'] * 4,
        'NUMERO_LOT': ['LOT001', 'LOT002', '', 'LOT004'],
        'original_s_line_raw': [
            'S;SESSION001;INV001;1000;SITE01;100;0;1;ART001;EMP001;A;UN;0;ZONE1;LOT001',
            'S;SESSION001;INV001;1001;SITE01;50;0;1;ART002;EMP001;A;UN;0;ZONE1;LOT002',
            'S;SESSION001;INV001;1002;SITE01;0;0;1;ART003;EMP001;A;UN;0;ZONE1;',
            'S;SESSION001;INV001;1003;SITE01;75;0;1;ART004;EMP001;A;UN;0;ZONE1;LOT004'
        ]
    }
    original_df = pd.DataFrame(original_data)
    
    # 2. DataFrame complété (template avec quantités réelles saisies)
    completed_data = {
        'Numéro Session': ['SESSION001'] * 4,
        'Numéro Inventaire': ['INV001'] * 4,
        'Code Article': ['ART001', 'ART002', 'ART003', 'ART004'],
        'Quantité Théorique': [100, 50, 0, 75],  # Quantités théoriques originales
        'Quantité Réelle': [95, 55, 10, 70],     # Quantités réelles saisies
        'Numéro Lot': ['LOT001', 'LOT002', '', 'LOT004']
    }
    completed_df = pd.DataFrame(completed_data)
    
    # 3. DataFrame distribué (ajustements calculés)
    distributed_data = {
        'CODE_ARTICLE': ['ART001', 'ART002', 'ART003', 'ART004'],
        'NUMERO_INVENTAIRE': ['INV001'] * 4,
        'NUMERO_LOT': ['LOT001', 'LOT002', 'LOTECART', 'LOT004'],
        'TYPE_LOT': ['type1', 'type1', 'lotecart', 'type1'],
        'QUANTITE_ORIGINALE': [100, 50, 0, 75],
        'AJUSTEMENT': [-5, 5, 10, -5],  # Écarts calculés
        'QUANTITE_CORRIGEE': [95, 55, 10, 70],  # Quantités théoriques ajustées
        'QUANTITE_REELLE_SAISIE': [95, 55, 10, 70],  # Quantités réelles saisies
        'original_s_line_raw': [
            'S;SESSION001;INV001;1000;SITE01;100;0;1;ART001;EMP001;A;UN;0;ZONE1;LOT001',
            'S;SESSION001;INV001;1001;SITE01;50;0;1;ART002;EMP001;A;UN;0;ZONE1;LOT002',
            None,  # LOTECART - nouvelle ligne
            'S;SESSION001;INV001;1003;SITE01;75;0;1;ART004;EMP001;A;UN;0;ZONE1;LOT004'
        ]
    }
    distributed_df = pd.DataFrame(distributed_data)
    
    return original_df, completed_df, distributed_df

def simulate_final_file_generation(original_df, completed_df, distributed_df):
    """Simule la génération du fichier final selon la logique améliorée"""
    
    # Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        final_file_path = f.name
        
        # En-têtes
        f.write("E;SESSION001;test;1;SITE01;;;;;;;;;;\n")
        f.write("L;SESSION001;INV001;1;SITE01;;;;;;;;;;\n")
        
        # Dictionnaires pour la logique
        real_quantities_dict = {}
        for _, row in completed_df.iterrows():
            key = (row["Code Article"], row["Numéro Inventaire"], str(row["Numéro Lot"]).strip())
            real_quantities_dict[key] = row["Quantité Réelle"]
        
        adjustments_dict = {}
        for _, row in distributed_df.iterrows():
            key = (row["CODE_ARTICLE"], row["NUMERO_INVENTAIRE"], str(row["NUMERO_LOT"]).strip())
            adjustments_dict[key] = {
                "TYPE_LOT": row["TYPE_LOT"],
                "QUANTITE_CORRIGEE": row["QUANTITE_CORRIGEE"],
                "AJUSTEMENT": row["AJUSTEMENT"],
                "QUANTITE_REELLE_SAISIE": row["QUANTITE_REELLE_SAISIE"]
            }
        
        # Traiter chaque ligne originale
        for _, original_row in original_df.iterrows():
            parts = original_row["original_s_line_raw"].split(";")
            
            code_article = original_row["CODE_ARTICLE"]
            numero_inventaire = original_row["NUMERO_INVENTAIRE"]
            numero_lot = str(original_row["NUMERO_LOT"]).strip()
            
            key = (code_article, numero_inventaire, numero_lot)
            
            # Déterminer les quantités attendues
            completed_data = {}
            for _, row in completed_df.iterrows():
                if (row["Code Article"] == code_article and 
                    row["Numéro Inventaire"] == numero_inventaire and 
                    str(row["Numéro Lot"]).strip() == numero_lot):
                    completed_data = {
                        "qte_theo_originale": row["Quantité Théorique"],
                        "qte_reelle_saisie": row["Quantité Réelle"]
                    }
                    break
            
            adjustment_data = adjustments_dict.get(key, {})
            if adjustment_data:
                adjustment_data["qte_theo_ajustee"] = adjustment_data.get("QUANTITE_CORRIGEE", 0)
                adjustment_data["qte_reelle_saisie"] = adjustment_data.get("QUANTITE_REELLE_SAISIE", 0)
            
            # Récupérer les quantités
            quantite_theo_ajustee = adjustment_data.get("qte_theo_ajustee", completed_data.get("qte_theo_originale", 0))
            quantite_reelle_saisie = completed_data.get("qte_reelle_saisie", 0)
            
            if key in adjustments_dict:
                # 1. Mettre à jour la quantité théorique ajustée
                parts[5] = str(int(quantite_theo_ajustee))
                # 2. Mettre à jour la quantité réelle saisie (NOUVELLE FONCTIONNALITÉ)
                parts[6] = str(int(quantite_reelle_saisie))
                
                # 3. Vérifier s'il y a un ajustement
                if key in adjustments_dict:
                    adjustment = adjustments_dict[key]
                    
                    if adjustment["TYPE_LOT"] == "lotecart":
                        # LOTECART : qté théo = qté réelle
                        parts[5] = str(int(adjustment["QUANTITE_CORRIGEE"]))
                        parts[6] = str(int(adjustment.get("QUANTITE_REELLE_SAISIE", adjustment["QUANTITE_CORRIGEE"])))
                        parts[7] = "2"  # Indicateur
                        parts[14] = "LOTECART"
                    else:
                        # Ajustement normal
                        parts[5] = str(int(adjustment["QUANTITE_CORRIGEE"]))
                        parts[6] = str(int(adjustment.get("QUANTITE_REELLE_SAISIE", adjustment["QUANTITE_CORRIGEE"])))
                         
                # Écrire la ligne
                f.write(";".join(parts) + "\n")
            else:
                # Ligne standard sans ajustement
                parts[5] = str(int(quantite_theo_ajustee))
                parts[6] = str(int(quantite_reelle_saisie))
                f.write(";".join(parts) + "\n")
        
        # Ajouter les nouvelles lignes LOTECART
        for _, row in distributed_df.iterrows():
            if pd.isna(row["original_s_line_raw"]) and row["TYPE_LOT"] == "lotecart":
                new_line = f"S;SESSION001;INV001;1002;SITE01;{int(row['QUANTITE_CORRIGEE'])};{int(row['QUANTITE_REELLE_SAISIE'])};2;{row['CODE_ARTICLE']};EMP001;A;UN;0;ZONE1;LOTECART"
                f.write(new_line + "\n")
    
    return final_file_path

def verify_final_file(final_file_path, completed_df, distributed_df):
    """Vérifie que le fichier final contient les bonnes quantités"""
    
    # Créer les dictionnaires de référence
    completed_dict = {}
    for _, row in completed_df.iterrows():
        key = (row["Code Article"], row["Numéro Inventaire"], str(row["Numéro Lot"]).strip())
        completed_dict[key] = {
            "qte_theo_originale": row["Quantité Théorique"],
            "qte_reelle_saisie": row["Quantité Réelle"]
        }
    
    adjustments_dict = {}
    for _, row in distributed_df.iterrows():
        key = (row["CODE_ARTICLE"], row["NUMERO_INVENTAIRE"], str(row["NUMERO_LOT"]).strip())
        adjustments_dict[key] = {
            "TYPE_LOT": row["TYPE_LOT"],
            "QUANTITE_CORRIGEE": row["QUANTITE_CORRIGEE"],
            "AJUSTEMENT": row["AJUSTEMENT"],
            "qte_theo_ajustee": row["QUANTITE_CORRIGEE"],
            "qte_reelle_saisie": row["QUANTITE_REELLE_SAISIE"]
        }
    
    # Analyser le fichier final
    consistent_lines = 0
    total_lines = 0
    
    with open(final_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip().startswith('S;'):
                parts = line.strip().split(';')
                total_lines += 1
                
                code_article = parts[8]
                numero_inventaire = parts[2]
                numero_lot = parts[14].strip()
                
                key = (code_article, numero_inventaire, numero_lot)
                
                qte_theo_finale = float(parts[5])
                qte_reelle_finale = float(parts[6])
                
                # Déterminer les quantités attendues
                completed_data = completed_dict.get(key, {})
                adjustment_data = adjustments_dict.get(key, {})
                
                expected_qte_theo = completed_data.get("qte_theo_originale", 0)
                expected_qte_reelle = completed_data.get("qte_reelle_saisie", 0)
                expected_qte_reelle_input = completed_data.get("qte_reelle_saisie", 0)  # Même valeur
                
                if numero_lot == "LOTECART":
                    line_type = "LOTECART"
                    # LOTECART : théo = réelle
                    expected_qte_theo = completed_data.get("qte_reelle_saisie", 0)
                    expected_qte_reelle_input = expected_qte_theo
                elif adjustment_data:
                    line_type = "AJUSTÉ"
                    # Ajustement : théo = qté ajustée
                    expected_qte_theo = adjustment_data.get("qte_theo_ajustee", 0)
                    expected_qte_reelle_input = adjustment_data.get("qte_reelle_saisie", expected_qte_theo)
                else:
                    line_type = "STANDARD"
                    expected_qte_reelle_input = 0  # Pas de saisie = 0
                
                # Vérifier la cohérence
                theo_ok = abs(qte_theo_finale - expected_qte_theo) < 0.001
                reelle_ok = abs(qte_reelle_finale - expected_qte_reelle) < 0.001
                reelle_input_ok = abs(float(parts[6]) - expected_qte_reelle_input) < 0.001
                
                status = "✅" if (theo_ok and reelle_ok) else "❌"
                print(f"{status} Ligne {line_num:2d} | {code_article:15s} | {line_type:8s} | Théo: {qte_theo_finale:3.0f} (attendu: {expected_qte_theo:3.0f}) | Réelle: {qte_reelle_finale:3.0f} (attendu: {expected_qte_reelle:3.0f}) | Input: {parts[6]:3s} (attendu: {expected_qte_reelle_input:3.0f})")
                
                if theo_ok and reelle_ok and reelle_input_ok:
                    consistent_lines += 1
                else:
                    print(f"      ⚠️ Incohérence détectée!")
    
    print("-" * 40)
    print(f"📊 Résultat: {consistent_lines}/{total_lines} lignes cohérentes ({(consistent_lines/total_lines)*100:.1f}%)")
    
    return consistent_lines == total_lines

def main():
    """Fonction principale de test"""
    print("🧪 Test de vérification des quantités théoriques ajustées vs quantités réelles")
    print("=" * 80)
    
    # 1. Créer les données de test
    print("📋 Création des données de test...")
    original_df, completed_df, distributed_df = create_test_data()
    print(f"   - {len(original_df)} lignes originales")
    print(f"   - {len(completed_df)} lignes dans le template complété")
    print(f"   - {len(distributed_df)} ajustements calculés")
    
    # 2. Simuler la génération du fichier final
    print("\n🔧 Génération du fichier final simulé...")
    final_file_path = simulate_final_file_generation(original_df, completed_df, distributed_df)
    print(f"   Fichier généré: {final_file_path}")
    
    # 3. Vérifier le fichier final
    success = verify_final_file(final_file_path, completed_df, distributed_df)
    
    # 4. Afficher le contenu du fichier pour inspection
    print(f"\n📄 Contenu du fichier final:")
    print("-" * 40)
    with open(final_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            print(f"   {line_num:2d}: {line.strip()}")
    
    # 5. Nettoyage
    os.unlink(final_file_path)
    
    # 6. Résultat final
    print("\n" + "=" * 80)
    if success:
        print("🎉 TEST RÉUSSI !")
        print("    ✅ Quantités théoriques ajustées sont correctement appliquées!")
        print("    ✅ Quantités réelles saisies correspondent aux écarts calculés")
        print("    ✅ Lots LOTECART ont quantité théorique = quantité réelle")
        print("    ✅ Template complété du fichier final")
    else:
        print("❌ TEST ÉCHOUÉ : Des incohérences ont été détectées!")
        print("    ⚠️ Vérifiez la logique de génération du fichier final")

if __name__ == "__main__":
    main()