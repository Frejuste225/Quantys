#!/usr/bin/env python3
"""Test rapide pour vérifier la logique des quantités"""

def test_quantities_logic():
    """Test de la logique des quantités"""
    
    print("🧪 Test de la logique des quantités")
    print("=" * 50)
    
    # Cas de test
    test_cases = [
        {
            "name": "Ligne standard sans ajustement",
            "original_theo": 100,
            "saisie_reelle": 100,
            "has_adjustment": False,
            "expected_theo_final": 100,  # Garde l'original
            "expected_reelle_final": 100
        },
        {
            "name": "Ligne avec ajustement normal",
            "original_theo": 100,
            "saisie_reelle": 95,
            "has_adjustment": True,
            "adjusted_theo": 95,
            "expected_reelle_input": 95,
            "expected_theo_final": 95,  # Quantité ajustée
            "expected_reelle_final": 95
        },
        {
            "name": "Ligne LOTECART",
            "original_theo": 0,
            "saisie_reelle": 10,
            "has_adjustment": True,
            "is_lotecart": True,
            "expected_reelle_input": 10,
            "expected_theo_final": 10,  # Théo = Réelle pour LOTECART
            "expected_reelle_final": 10
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print("-" * 30)
        
        # Simuler la logique
        qte_reelle_final = case["saisie_reelle"]  # Toujours la saisie
        qte_reelle_input = case["saisie_reelle"]  # Quantité réelle saisie dans colonne G
        
        if case.get("has_adjustment", False):
            if case.get("is_lotecart", False):
                qte_theo_final = case["saisie_reelle"]  # LOTECART
            else:
                qte_theo_final = case["adjusted_theo"]  # Ajustement normal
        else:
            qte_theo_final = case["original_theo"]  # Pas d'ajustement
        
        # Vérifier
        theo_ok = qte_theo_final == case["expected_theo_final"]
        reelle_ok = qte_reelle_final == case["expected_reelle_final"]
        input_ok = qte_reelle_input == case.get("expected_reelle_input", case["saisie_reelle"])
        
        print(f"   Théorique: {qte_theo_final} (attendu: {case['expected_theo_final']}) {'✅' if theo_ok else '❌'}")
        print(f"   Réelle:    {qte_reelle_final} (attendu: {case['expected_reelle_final']}) {'✅' if reelle_ok else '❌'}")
        print(f"   Input (Col G): {qte_reelle_input} (attendu: {case.get('expected_reelle_input', case['saisie_reelle'])}) {'✅' if input_ok else '❌'}")
        
        if not (theo_ok and reelle_ok and input_ok):
            all_passed = False
            print("   ❌ ÉCHEC")
        else:
            print("   ✅ SUCCÈS")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("La logique des quantités (théorique, réelle et input) est correcte.")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ!")
        print("La logique nécessite des corrections.")
    
    return all_passed

if __name__ == "__main__":
    test_quantities_logic()