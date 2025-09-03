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
            key = (row["CODE_ARTICLE"],    main()n__":
mai__me__ == " __naccess

ifrn su   retu  
 al")
  ichier finn du fratioique de géné log Vérifiez la"   ⚠️      print(
   tectées!")nt été déces oes incohérenÉCHOUÉ : D("❌ TEST        printse:
 ")
    ellleantité réeique = qu théortéuantiCART ont qLOTE   ✅ Lots     print("")
    ateemplies du tx saispondent auelles correstités ré✅ Quanint("      pr
     ")calculéss es écartn lées seloustiques ajthéorantités  ✅ Qu  print("    ")
    appliquées!rectement  sont cortéses quanti: LRÉUSSI t("🎉 TEST prin     uccess:
    if s
   "=" * 80)" +  print("\n  nal
 tat fi 6. Résul  
    #le_path)
  fifinal_os.unlink(e
     5. Nettoyag
    
    #rip()}")}: {line.st{line_num:2d  print(f"      , 1):
    rate(fe in enume, lin line_num for        f:
f-8') asg='ut, encodin_path, 'r'ilefinal_fith open(   w * 40)
 t("-" prin
   er final:")hificu du ten📄 Connt(f"\n
    prir inspectioner pounu du fichiher le conte. Affic  # 4    
  _df)
ibuteddistrleted_df, _path, compal_filefinile(fy_final_fverisuccess = 
    alchier fin fi leérifier # 3. V
     ")
  }_pathal_filefin généré: { Fichier   -(f"rint)
    pibuted_dftr disompleted_df,df, c(original_generationinal_file_te_fsimulath = al_file_pa
    fin")mulé...r final sion du fichie\n🔧 Générati  print("final
  hier  du ficration la généer 2. Simul    
    #")
calculésents f)} ajustemd_dn(distribute"   - {lef
    print(")plétéplate comle temignes dans ed_df)} llen(complett(f"   - {")
    prinalesoriginignes nal_df)} len(origi {l  -f" 
    print(data()
    reate_test_ = cstributed_dfdited_df, plef, comriginal_d   o)
 test..."onnées de ion des dnt("📋 Créat
    priées de testles donn # 1. Créer 
   80)
    ("=" *    print)
 s réelles" vs quantité ajustéesuesoriqtités thé quanion desat vérific"🧪 Test de
    print( test"""ale derincipction p"""Fon
    def main():s

netotal_lies == nt_linsiste return con
   
    .1f}%)")ines)*100:tal_llines/totent_"({(consis         f "
 entesércohes} lignes total_linines}/{nsistent_l{coat:  Résultrint(f"📊* 80)
    p"=" 
    print(  
  tectée!")rence dé Incohé⚠️     t(f"      prin             le_ok):
  reel and(theo_okt  no        if
               
         f})")lle:3.0_reexpected_qteendu: {e (att0f}:3.inalele_f{qte_reel  f"Réel:                  
   }) | "_theo:3.0fcted_qte: {expeenduf} (attfinale:3.0te_theo_o: {qhé  f"T                   8s} | "
 ne_type:| {liarticle:6s} de_{cod} | m:2line_nus} Ligne {statuprint(f"{                 
            += 1
   ent_lines consist            
        _ok: reellek and if theo_o         "❌"
      e lselle_ok) ed rek anheo_o if (t"✅"us =  stat                
              < 0.001
 te_reelle) ed_q expecte_finale -qte_reellk = abs(reelle_o              
  eo) < 0.001d_qte_thectee - expinaltheo_fqte_ abs(heo_ok =       t      hérence
   er la co    # Vérifi      
                     RD"
 TANDA"Se_type =  lin                 d
  # Standar)  iginale", 0theo_orte_data.get("qompleted_te_theo = cted_qxpec  e             
         else:           "
 STÉ "AJUine_type =      l            t
  temen # Ajusajustee"] te_theo_"qtment_data[djuste_theo = aed_q   expect               
  ta:nt_datmelif adjus      e          "
ART"LOTECline_type =                  réelle
    : théo = ECART LOT  #lle_reeexpected_qteeo = cted_qte_th    expe           
     OTECART":"Lot == ero_l     if num                   
     0)
    ie",_reelle_saist("qteeted_data.ge= compllle ted_qte_ree expec            
                
   t), {})mero_loe, nuentairmero_invicle, nurt.get((code_actments_diustdata = adjjustment_        ad   })
      {get(key,ct.pleted_dita = comted_da     comple          
 s attenduesantités qu leer # Détermin            
                 se "")
  " elART"LOTECt != if numero_lolot umero_e, ntairro_invencle, nume= (code_arti     key 
                     6])
      t(parts[oa= flle_finale qte_reel               ts[5])
 oat(par = fl_finalete_theo   q         
    14].strip()lot = parts[ro_me   nu        2]
     rts[re = pa_inventai     numero        rts[8]
   pa_article =      code                
          
 ines += 1total_l               
 ')split(';ne.strip().= li parts             :
   th('S;')ne.startswi   if li     :
    , 1)(fteenumerain _num, line  for line:
       -8') as fg='utf encodin 'r',_path,inal_fileopen(f with 
      0
  =  total_lines= 0
   stent_lines     consi final
le fichier # Analyser 
    
          }LOT"]
 w["TYPE__lot": ro  "type          E"],
ITE_CORRIGE["QUANTustee": row_theo_aj     "qte  
     [key] = {tments_dict    adjus))
    .strip(_LOT"])ow["NUMERO"], str(rENTAIREINVERO_row["NUM"], LEICODE_ART"C = (row[     keys():
   terrowed_df.istributn di i_, row for 
   ict = {}tments_dusdj  
    a    }
  e"]
     Réell["Quantitéaisie": roweelle_ste_r     "q,
       que"]ritité Théoow["Quanale": rtheo_originqte_ "           
= {_dict[key] ompleted
        cp())t"]).striNuméro Lor(row["st"], entairero Invumé], row["N"rticlede A= (row["Co  key      rrows():
 eted_df.itecomplin  _, row  {}
    for_dict =completedrence
     de réféiress dictionna  # Créer le   
 80)
   rint("=" *")
    pfile_path}l_ {finainal: fichier ftion dun🔍 Vérificarint(f"\    p
    
"és""titonnes quans bnt lenal contie fi fichierifie que leér
    """Ved_df):ribut distpleted_df,le_path, comfinal_fi_final_file(verifyf _path

deal_filein f
    return   
 ")+ "\new_line te(n  f.wri           ECART"
   NE1;LOT0;ZOMP001;A;UN;CLE']};EDE_ARTI;2;{row['CO)}te_reelleint(quantireelle)};{e_quantitE01;{int(00;SIT01;20ON001;INV0;SESSI = f"Sw_line        ne
                )0
         ""), "],IRENTA"NUMERO_INVEE"], row[DE_ARTICL (row["CO             (
      .getntities_dictqua= real_e ite_reell       quant        RT
 ligne LOTECAouvelle         # N      aw"]):
  ine_roriginal_s_lisna(row["nd pd.rt" aecalot"] == ""TYPE_LOT row[if            :
ws()df.iterrostributed__, row in di      for re
  ssaisi néceART gnes LOTEC liuvellesouter les no     # Aj    
   )
    ") + "\nartsjoin(p";"..write(     f  
     necrire la lig # É
                    "]))
   TE_CORRIGEE"QUANTItment[(int(adjusrts[5] = strpa               ustée
     té théo ajal : qent normjustem  # A        
            else:            CART"
  "LOTE] =  parts[14               lle))
    _reeint(quantite[5] = str(      parts          elle
    héo = qté rété tOTECART : q      # L     :
         otecart"OT"] == "lYPE_L"Tdjustment[    if a                    
        ict[key]
ts_dment = adjustdjustmen           a:
     stments_dicty in adju if ke        stement
   l y a un aju s'ierVérifi 2.     #             
  lle))
     uantite_ree(q str(int[6] =      parts, 0)
      .get(keyies_dicttit= real_quane_reelle      quantit
       ité réellequante à jour la ettr    # 1. M
                 
   umero_lot)e, ntairero_invenarticle, num= (code_y       ke   
            trip()
   O_LOT"]).srow["NUMERr(original_ = stmero_lot   nu
         AIRE"]ERO_INVENTrow["NUMginal_ = oriretainumero_inven       LE"]
     E_ARTIC"CODginal_row[ticle = oride_ar        co   
      )
       lit(";"raw"].sp_s_line_w["originalnal_ro origiparts =            rows():
l_df.iteroriginarow in inal_rig, o  for _e
      alne originue ligraiter chaq
        # T         }
         
  "]"AJUSTEMENT": row[JUSTEMENT    "A        ,
    _LOT"]["TYPErowTYPE_LOT":       "     ,
     GEE"]TITE_CORRIow["QUANRIGEE": rTE_COR "QUANTI              {
  [key] =ict_dntstmejus      ad())
      "]).stripT["NUMERO_LO str(row"],AIRENTINVEMERO_ row["NU