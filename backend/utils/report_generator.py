"""
Générateur de rapports pou": {str(e)}rtreur expon f"Er    retur        : {e}")
ifications vérport donnéerreur exrror(f"E  logger.e          e:
 ption aspt Exce   exce   
                 )
 mat_type}"é: {forsupportat non "FormValueError(f  raise            
   :  else          n_data)
icatioeport(verife_text_r.generateratoreportGencationRn Verifiretur                :
== "text") ype.lower(t_tif forma       el)
     efault=strcii=False, dnsure_asnt=2, e, indeata_dationumps(verificson.dreturn j               json":
 = ".lower() =_typeormat  if f
              try:   
 """satts formenans différrification ds de vééee les donn"Export   ""r:
     son") -> st"je: str = t_typormany], fDict[str, Ata: ication_dan_data(verifatiofic_veriort exp
    deftaticmethod    
    @s        }

    .isoformat()ow()ime.natetmestamp": d"ti           ,
     : str(e)rror"         "e
       se,ss": Fal"succe       
          { return       )
    : {e}"pport résumé raérationErreur gén.error(f"     logger
       e:ption as ept Exce        exc 
       mary
    rn sum   retu       
         "
     "poor = tatus"]y["s      summar     lse:
           e    "
  ableept = "acc"]us"stat  summary[            :
   50re"] >=ty_scoali["quary  elif summ   "
       "] = "goodry["statusumma  s         0:
     "] >= 7eor"quality_scf summary[    eli
        ent"= "excelltatus"] "s  summary[            "]:
  issuess_lts"]["hassing_resuy["proceummarnd not s] >= 90 ascore"ality_mmary["qu   if su
         bal gloe statuter l Détermin           #     
  }
                         }
          0)
cators",ndirect_iort("incrt", {}).geet("loteca.gn_datacatioverificators": ect_indiorrinclotecart_       "             ", 0),
icatorsect_indt("corr.get", {})"lotecar.get(_dataerification": vcatorsrrect_inditecart_co"lo          
          ", [])),srningt("waion_data.geficat": len(verirnings       "wa   
          es", [])),.get("issuication_data: len(verifissues""critical_                  ": {
  _indicators  "quality           },
                   [])) > 0
 "issues",a.get(ation_datn(verifics": lehas_issue"                    
0),, "justmentswith_adet("lines_ {}).gts",ustmen"adjdata.get(n_ficatio veries":djusted_lin"a             ),
       ", 0total_lines}).get("art", {"lotec_data.get(fications": verirt_linecaote        "l           
 lue", 0),_va_discrepancyet("totaly", {}).gl_summarglobata.get("cation_dacy": verifial_discrepan   "tot                {
  ts":ng_resul "processi                     },
         que", 0)
 cles_uni_arti"total, {}).get(l_summary"get("globaa.ation_datverificcles": arti"total_                   
 ", 0),toriesinvenal_tot"", {}).get(summaryglobal_"on_data.get(tirificatories": vel_inven    "tota         
       nes_s", 0),ta_liget("da", {}).uctureta.get("stron_daficatirilines": ve    "total_       
          {ile_info":         "f      
 mp"),tatimeserification_data.get("vtion_ verificatamp":es   "tim           
   0),",scoreality_ta.get("qufication_daverie": scorquality_     "       se),
     Fal"success",(ata.getfication_d": veri"success       
         ry = {umma           s   try:
 "
     API""mé pour l'ort résuppGénère un ra     """]:
   nyict[str, A-> D]) Anytr, Dict[s: _dataerificationreport(vte_summary_eraen
    def godmethic 
    @statr(e)}"
   t: {strapportion rareur géné"Er f      return    
  l: {e}")uerapport textgénération rreur r.error(f"E   logge:
         s eeption aExc   except          
      ines)
  n(report_ljoiurn "\n".    ret
                  " * 80)
  nd("=_lines.appeport        re
    ion 1.0")rse X3 - Veagtaire SInvenulinette d'.append("Monesreport_li           ")
 )}:%S' à %H:%Mm/%Y/%ime('%dft.strow()ime.nle {datett généré ppord(f"Raes.appen report_lin          * 80)
 end("=" t_lines.appor       repe
      pag   # Pied de                  
   nd("")
pees.aplinrt_     repo
            ")
       autresngs) - 5} t {len(warnid(f"  ... e.appen_linesrt       repo            ) > 5:
 arningslen(w   if             
 warning}") • {(f" appendport_lines.   re                ngs[:5]:
 arnin w ior warning        f
        )}):")arnings{len(wnts (sseme\n⚠️ Avertid(f"appenlines.report_              
  rnings:       if wa       
      es")
     autrsues) - 5}. et {len(is  ..pend(f"nes.aprt_li      repo             
  > 5: len(issues)         if     )
   {issue}" •append(f" nes.port_li  re             [:5]:
     issuessue in    for is          :")
   (issues)})s ({lenitiquecres Problèm❌ \npend(f"lines.ap     report_     
      :issues  if 
                      fisante")
alité insufd("❌ Quines.appenport_l     re          else:
          )
   table"lité accepQuad("⚠️ _lines.appen     report         70:
   >=score if quality_      el     
 qualité")e nt✅ Excellepend("rt_lines.appo      re       :
   core >= 90 quality_s  if            
   )
       ore}/100"_scityité: {qual qual"Score dend(fs.appeline   report_       " * 40)
  "-nd(s.appeeport_line   r      LITÉ")
   UATION QUAÉVALd("🎯 lines.appen    report_    
       
         s", [])et("warningn_data.g verificatiogs =  warnin   ])
       ues", [ss"ita.get(on_datirifica ve  issues =
          ore", 0)ality_scet("qua.g_daton= verificatiity_score al qu         é
  itQual     # 
                  
 end("")t_lines.app       repor        f}%")
 ', 0):.1centageericator_2_pindget('ics.: {statist 2 Indicateurppend(f"%ines.a report_l         
      ")1f}%0):.rcentage', ero_theo_peics.get('zisttatque = 0: {sé théorind(f"% Qtes.appe  report_lin           )
   1f}%"tage', 0):.ercenjustment_p.get('adcs {statistitements:(f"% Ajusppendrt_lines.a   repo           
  0):.1f}%")centage', otecart_perics.get('l{statistT: "% LOTECARpend(ft_lines.ap       repor         
("-" * 40)ppendport_lines.a        re       QUES")
 TATISTIappend("📈 Sport_lines.   re       s:
      stic  if stati      })
    ", {s"statisticta.get(on_darificatis = veatistic         sttiques
    Statis   #        
           ("")
  nes.appendeport_li          r   ")
    0)} lignes)nes_count',a.get('liat{d  f"(                                       2f} "
 ncy', 0):+.epaal_discra.get('tot: {daticle}  • {artd(f"ppens.alinereport_                        ents[:5]:
stmted_adju in sorticle, data for ar            
          )                rse=True
        reve          ,
       ncy", 0))repatal_disc[1].get("to x: abs(xkey=lambda                      ems(),
  j_summary.it         ad            orted(
   ustments = sdj_a  sorted                 ")
 cle:ar artistements p📈 Top aju\nppend(".aort_lines         rep        y:
   mmaradj_su if           {})
     summary", justment_et("adts.g adjustmenmary =dj_sum   a          ts
   p ajustemenTo   #          
               ")
     ]))}adjusted', [articles_('nts.get(adjustmetés: {lenicles ajuspend(f"Artines.apport_lre            0)}")
    2', ndicator_th_i('lines_witments.get 2: {adjusicateurvec ind"Lignes and(f.appereport_lines               ', 0)}")
 djustmentsth_a_wis.get('linesnt: {adjustmetsenustemec ajs av(f"Lignelines.append report_             " * 40)
  .append("-esinreport_l              
  ENTS")STEMJUS A ANALYSE DEd("⚖️enrt_lines.appepo      r          :
 0 0) >ustments",es_with_adjet("lints.g adjustmen       if)
      {}",justments.get("adication_datas = veriftmentjus  ad          tements
    # Ajus                
 )
   d(""s.appen report_line           ")
    r', 'N/A')}icatot('indple.geteur: {sam"Indica   f                             
          0)} - "real', tity_le.get('quan"Qté: {samp     f                             
         - "/A')} 'N('article',get• {sample.(f"  ndnes.appe   report_li               
      les[:3]:ampe in spl for sam                   :")
ons LOTECARTillchantn📋 Éend("\es.appinort_l        rep          samples:
  if                 )
, []s"e_articlesamplet("cart.g= lote   samples             LOTECART
 illons ant      # Éch          
               )}")
 ors', 0ndicatct_irret('incotecart.georrects: {loncteurs i"Indicas.append(f report_line               
 0)}")ndicators',rect_it.get('corecarcts: {lotrecorateurs end(f"Indicappes. report_lin            
   )}"), 0ines'_l'totalcart.get(: {lote LOTECARTTotal ligness.append(f"t_line   repor             40)
 nd("-" *ines.appe   report_l            CART")
 NALYSE LOTE"🏷️ Apend(_lines.aport     rep        0:
    0) > ","total_linesecart.get( if lot    
       , {})"lotecart"ta.get(fication_daeri = votecart    lT
         # LOTECAR             
  
        pend("").apnesrt_li       repo)}")
     , 0ities'_quantt('negativentities.ge{quanégatives: ntités ua"Qs.append(fline     report_  ")
     )}t', 0_coun('zero_realgeties.ntit= 0: {quaqté réelle c nes aveend(f"Lig_lines.app     report      
 ")}t', 0)tical_coun'zero_theoreet(antities.gquique = 0: {té théorignes avec qnd(f"Lnes.appereport_li    
        ):,.2f}")repancy', 0isctal_d.get('toquantities total: {"Écartnd(f.appelines    report_  )
      f}"', 0):,.2l_real.get('totaesntitiéel: {quaf"Total rines.append(rt_l  repo
          :,.2f}")cal', 0)al_theoretiet('tots.gtieue: {quantiiqthéord(f"Total nes.appent_li    repor)
        * 40ppend("-" _lines.a   report)
         UANTITÉS"E DES QANALYSpend("📊 ort_lines.ap  rep     )
     es", {}uantitit("qata.geon_dti verificaities =       quanttés
     uanti# Q          
           d("")
   penes.apport_linre  
          , 0)}")s'ine('invalid_le.getructur{stnvalides: Lignes iend(f"s.appeport_line   r    ")
     s_s', 0)}ata_linee.get('d{structurnnées): gnes S (do"Lies.append(f_lin      report      , 0)}")
nes_l'ader_liet('hee.gstructurntaires): {inveignes L ((f"Lend_lines.apport      rep   )
   _e', 0)}"ineser_let('heade.g{structur: es)êtes E (en-tnd(f"Lign.appe_lines report        ")
   es', 0)}tal_line.get('to{structur lignes: "Totald(fappeneport_lines.      r    " * 40)
  d("-es.appen  report_lin         ER")
  DU FICHIUCTUREend("📁 STRppt_lines.aepor         r})
   ructure", {et("stdata.gcation_erifiucture = v    strer
        re du fichi   # Structu        
            ")
 pend("es.aprt_linpo        re  n'}")
  se '❌ Notments') el'has_adjusary.get(lobal_summ if g {'✅ Oui'ements:usttient ajConappend(f"lines.ort_ rep          )
 e '❌ Non'}"') elsas_lotecartet('hsummary.global_i' if g✅ Ou: {'nt LOTECARTd(f"Contiees.appen report_lin          0)}")
 cy_value', _discrepan'totalet(.gsummaryglobal_ {otal:t tf"Écarnd(appeport_lines.  re
          ', 0)}")niqueicles_uotal_artt('ty.geummarlobal_s uniques: {gticlesappend(f"Arlines.eport_      r
      )', 0)}"oriestotal_inventget('ry.al_summa: {globes traitésnventairappend(f"Is.ine_l    report       " * 40)
 .append("-es_lin  report           GLOBAL")
🎯 RÉSUMÉ"pend(nes.aprt_li      repo  ", {})
    aryobal_summ.get("glataion_dificaty = verobal_summar   gl
         é global Résum      #   
      )
         ppend(""ines.at_lrepor    )
        'N/A')}"imestamp', rification_tget('vea.fication_daterie: {v effectuéication(f"Vérifappends. report_line         )}")
  sed', 'N/A'trategy_uget('sion_info.{sesslisée: tégie utipend(f"Straap_lines.  report       ")
   'N/A')}tatus', info.get('sssion_se"Statut: {append(fines._l     report
       /A')}")ename', 'Nl_fil('originao.getinf{session_original: chier "Fi(fndlines.appet_       repor
     A')}")n_id', 'N/sessio('on_info.getssiID: {sef"Session d(ens.apport_line     rep  0)
      4pend("-" *ort_lines.ap        rep)
    LES"GÉNÉRATIONS ("📋 INFORMAppends.areport_line           fo", {})
 ssion_ina.get("se_datverificationion_info =   sess      ales
    nérs gémation # Infor             
        nd("")
  ppe.aesinreport_l           )
 ("=" * 80.appendrt_lines       repo  )
   AGE X3" SN - FICHIERIFICATIORT DE VÉR"📊 RAPPOappend(ort_lines.     rep       =" * 80)
ppend("es.a_lin     report    
   apporte du r   # En-têt           
      []
    es = port_lin        rey:
          tr"
  "le"tuel lisibpport texun ra""Génère    " str:
      Any]) ->[str,ata: Dictrification_dxt_report(veenerate_te
    def godticmeth    @sta
"""
    és détaillonicatits de vérifr de rapporérateu""Gén
    "ator:nReportGenercatioclass Verifime__)

naLogger(__ng.get = loggi

loggerjsone
import ort datetim impmeom dateti, List
frict, Anyort Dimptyping om ogging
frport l"""
im X3
agers S des fichiecationa vérifir l