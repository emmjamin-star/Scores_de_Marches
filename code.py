import streamlit as st
import ezc3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile
from math import sqrt

st.set_page_config(page_title="Scores de marches - Faps, eFaps, eGVI, GDI, GPS", layout="centered")
st.title("🦿 Scores de marches - Faps, eFaps, eGVI, GDI, GPS")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d dont au moins un fichier d'essai statique et trois d'essai dynamique")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)
st.header("2. Indiquer le score allant de 0 (aucune aide à la marche) à 5 (participant totalement dépendant) pour les aides ambulatoire et les dispositifs d'assistances")
df = pd.DataFrame({'Score' : [0,1,2,3,4,5]})

AmbulatoryAids = st.selectbox(
    "Pour l'aide ambulatoire :",
    df['Score'])
    
AssistiveDevice = st.selectbox(
    "Pour le dispositif d'assistance :",
    df['Score'])

if uploaded_files:
    selected_file_statique = st.selectbox("Choisissez un fichier statique pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique1 = st.selectbox("Choisissez un fichier dynamique 1 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique2 = st.selectbox("Choisissez un fichier dynamique 2 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique3 = st.selectbox("Choisissez un fichier dynamique 3 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique4 = st.selectbox("Choisissez un fichier dynamique 4 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique5 = st.selectbox("Choisissez un fichier dynamique 5 pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_statique.read())
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique1.read())
        tmp1_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique2.read())
        tmp2_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique3.read())
        tmp3_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique4.read())
        tmp4_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique5.read())
        tmp5_path = tmp.name
        
if st.button("Lancer le calcul des scores de marche"):
    try:
      # Score eFAPS
        mval = 1.3/(sqrt(9.81*0.85)) #Chiffre de l'INRETS
        def calculate_faps(trials, static_file, walking_aids=False, assistive_devices=False):
            # 1. PARAMÈTRES ANTHROPOMÉTRIQUES
            try:
                statique = ezc3d.c3d(static_file)
                labelsStat = statique['parameters']['POINT']['LABELS']['value']
        
                def get_static_point(label):
                    if label not in labelsStat: return None
                    idx = labelsStat.index(label)
                    pt = statique['data']['points'][:3, idx, :]
                    mask = ~np.isnan(pt[0, :])
                    return pt[:, mask][:, 0] if np.any(mask) else None
        
                # Calcul de la longueur de jambe (Leg Length) en METRES
                p1 = get_static_point('LPSI')
                p2 = get_static_point('LANK')
                if p1 is None or p2 is None:
                    print("Erreur : Marqueurs LPSI ou LANK introuvables.")
                    return
        
                leg_length = np.linalg.norm(p1 - p2) / 1000.0 
            except Exception as e:
                print(f"Erreur statique : {e}")
                return
        
            results = {'sl_r': [], 'sl_l': [], 'st_r': [], 'st_l': [], 'dbs': []}
        
            # 2. TRAITEMENT DES ESSAIS
            for trial_path in trials:
                try:
                    acq = ezc3d.c3d(trial_path)
                    labels = acq['parameters']['POINT']['LABELS']['value']
                    freq = acq['header']['points']['frame_rate']
                    data = acq['data']['points']
        
                    def get_clean_marker(label):
                        if label not in labels: return None
                        idx = labels.index(label)
                        m = data[:3, idx, :].copy()
                        for i in range(3):
                            mask = np.isnan(m[i, :])
                            if np.any(mask) and not np.all(mask):
                                m[i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), m[i, ~mask])
                        return m
        
                    r_he = get_clean_marker('RHEE')
                    l_he = get_clean_marker('LHEE')
                    if r_he is None or l_he is None: continue
        
                    # Détection des Heel Strikes (Axe Z)
                    hs_r, _ = find_peaks(-r_he[2, :], distance=int(freq*0.4), prominence=2)
                    hs_l, _ = find_peaks(-l_he[2, :], distance=int(freq*0.4), prominence=2)
        
                    # Temps de Pas (Step Time) et Longueur de Pas (Step Length)
                    # Jambe Droite (LHS -> RHS)
                    for t0 in hs_l:
                        next_r = hs_r[hs_r > t0]
                        if len(next_r) > 0:
                            t1 = next_r[0]
                            results['st_r'].append((t1 - t0) / freq) # Temps en secondes
                            results['sl_r'].append(np.abs(r_he[0, t1] - l_he[0, t0]) / 1000.0) # Distance en mètres
        
                    # Jambe Gauche (RHS -> LHS)
                    for t0 in hs_r:
                        next_l = hs_l[hs_l > t0]
                        if len(next_l) > 0:
                            t1 = next_l[0]
                            results['st_l'].append((t1 - t0) / freq)
                            results['sl_l'].append(np.abs(l_he[0, t1] - r_he[0, t0]) / 1000.0)
        
                    # Base de soutien dynamique (cm) - Ecart Y moyen
                    results['dbs'].append(np.abs(np.mean(r_he[1, :]) - np.mean(l_he[1, :])) / 10.0)
        
                except Exception as e:
                    print(f"Erreur essai {trial_path}: {e}")
        
            # 3. CALCULS FINAUX ET SCORING
            if not results['sl_r'] or not results['sl_l']:
                print("Calcul impossible : Données de pas manquantes.")
                return
        
            # Moyennes
            avg_sl_r = np.mean(results['sl_r'])
            avg_sl_l = np.mean(results['sl_l'])
            avg_st_r = np.mean(results['st_r'])
            avg_st_l = np.mean(results['st_l'])
            avg_dbs = np.mean(results['dbs'])
        
            # --- NORMALISATION SELON FAPS ---
            # GSL (Ratio Longueur pas / Longueur Jambe)
            gsl_r = avg_sl_r / leg_length
            gsl_l = avg_sl_l / leg_length
            
            # GV (Vitesse normalisée par jambe = GSL / Step Time)
            gv_r = gsl_r / avg_st_r
            gv_l = gsl_l / avg_st_l
        
            # --- ALGORITHME DE DÉDUCTION ---
            def get_step_function_penalty(gv_val, gsl_val, st_val):
                # Pénalité progressive si hors des normes (Max ~7.33 pts par paramètre pour atteindre 22)
                p_v = 0 if 1.1 <= gv_val <= 1.5 else min(min(abs(gv_val - 1.1), abs(gv_val - 1.5)) / 0.4 * 7.33, 7.33)
                p_sl = 0 if 0.69 <= gsl_val <= 0.86 else min(min(abs(gsl_val - 0.69), abs(gsl_val - 0.86)) / 0.2 * 7.33, 7.33)
                p_st = 0 if 0.50 <= st_val <= 0.63 else min(min(abs(st_val - 0.50), abs(st_val - 0.63)) / 0.2 * 7.33, 7.33)
                return min(p_v + p_sl + p_st, 22)
        
            # Déductions A et B (Fonctions de pas)
            deduction_A = get_step_function_penalty(gv_l, gsl_l, avg_st_l)
            deduction_B = get_step_function_penalty(gv_r, gsl_r, avg_st_r)
        
            # Déduction C : Asymétrie (Max 8 points)
            diff_asy = np.abs(gsl_r - gsl_l)
            deduction_C = 0 if diff_asy < 0.03 else min(((diff_asy - 0.03) / 0.15) * 8, 8)
        
            # Déduction D : Base de Support Dynamique (Max 8 points)
            # Norme typique assumée entre 5cm et 10cm de large
            if 5 <= avg_dbs <= 10:
                deduction_D = 0
            else:
                dbs_diff = min(abs(avg_dbs - 5), abs(avg_dbs - 10))
                deduction_D = min((dbs_diff / 8) * 8, 8) 
        
            # Déductions E et F : Aides et Dispositifs
            deduction_E = 5 if AmbulatoryAids>0 else 0
            deduction_F = 5 if AssistiveDevice>0 else 0
        
            # Formule Finale
            total_deductions = deduction_A + deduction_B + deduction_C + deduction_D + deduction_E + deduction_F
            score_faps = 100 - total_deductions
            
            # Plancher théorique du FAPS
            score_min = 30 if (walking_aids or assistive_devices) else 40
            score_faps = max(score_min, score_faps)
            st.markdown("### 📊 Résultats du score FAPS")
            st.write(f"Score FAPS : {score_faps:.2f}")
            st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 95 et 100. Tout score en-dehors indique une atteinte à la fonctionnalité de la marche.")
            
        trials_list = [tmp1_path, tmp2_path, tmp3_path, tmp4_path, tmp5_path]
        calculate_faps(trials_list, tmp_path, walking_aids=False, assistive_devices=False)
      
        # Score eFAPS
        def calculate_efaps(trials, static_file, walking_aids=False, assistive_devices=False):
            # 1. PARAMÈTRES ANTHROPOMÉTRIQUES
            try:
                statique = ezc3d.c3d(static_file)
                labelsStat = statique['parameters']['POINT']['LABELS']['value']
        
                def get_static_point(label):
                    if label not in labelsStat: return None
                    idx = labelsStat.index(label)
                    pt = statique['data']['points'][:3, idx, :]
                    mask = ~np.isnan(pt[0, :])
                    return pt[:, mask][:, 0] if np.any(mask) else None
        
                # Calcul de la longueur de jambe (Leg Length) en METRES
                p1 = get_static_point('LPSI')
                p2 = get_static_point('LANK')
                if p1 is None or p2 is None:
                    print("Erreur : Marqueurs LPSI ou LANK introuvables.")
                    return
        
                leg_length = np.linalg.norm(p1 - p2) / 1000.0 
            except Exception as e:
                print(f"Erreur statique : {e}")
                return
        
            results = {'sl_r': [], 'sl_l': [], 'st_r': [], 'st_l': [], 'dbs': []}
        
            # 2. TRAITEMENT DES ESSAIS
            for trial_path in trials:
                try:
                    acq = ezc3d.c3d(trial_path)
                    labels = acq['parameters']['POINT']['LABELS']['value']
                    freq = acq['header']['points']['frame_rate']
                    data = acq['data']['points']
        
                    def get_clean_marker(label):
                        if label not in labels: return None
                        idx = labels.index(label)
                        m = data[:3, idx, :].copy()
                        for i in range(3):
                            mask = np.isnan(m[i, :])
                            if np.any(mask) and not np.all(mask):
                                m[i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), m[i, ~mask])
                        return m
        
                    r_he = get_clean_marker('RHEE')
                    l_he = get_clean_marker('LHEE')
                    if r_he is None or l_he is None: continue
        
                    # Détection des Heel Strikes (Axe Z)
                    hs_r, _ = find_peaks(-r_he[2, :], distance=int(freq*0.4), prominence=2)
                    hs_l, _ = find_peaks(-l_he[2, :], distance=int(freq*0.4), prominence=2)
        
                    # Temps de Pas (Step Time) et Longueur de Pas (Step Length)
                    # Jambe Droite (LHS -> RHS)
                    for t0 in hs_l:
                        next_r = hs_r[hs_r > t0]
                        if len(next_r) > 0:
                            t1 = next_r[0]
                            results['st_r'].append((t1 - t0) / freq) # Temps en secondes
                            results['sl_r'].append(np.abs(r_he[0, t1] - l_he[0, t0]) / 1000.0) # Distance en mètres
        
                    # Jambe Gauche (RHS -> LHS)
                    for t0 in hs_r:
                        next_l = hs_l[hs_l > t0]
                        if len(next_l) > 0:
                            t1 = next_l[0]
                            results['st_l'].append((t1 - t0) / freq)
                            results['sl_l'].append(np.abs(l_he[0, t1] - r_he[0, t0]) / 1000.0)
        
                    # Base de soutien dynamique (cm) - Ecart Y moyen
                    results['dbs'].append(np.abs(np.mean(r_he[1, :]) - np.mean(l_he[1, :])) / 10.0)
        
                except Exception as e:
                    print(f"Erreur essai {trial_path}: {e}")
        
            # 3. CALCULS FINAUX ET SCORING
            if not results['sl_r'] or not results['sl_l']:
                print("Calcul impossible : Données de pas manquantes.")
                return
        
            # Moyennes
            avg_sl_r = np.mean(results['sl_r'])
            avg_sl_l = np.mean(results['sl_l'])
            avg_st_r = np.mean(results['st_r'])
            avg_st_l = np.mean(results['st_l'])
            avg_dbs = np.mean(results['dbs'])
        
            # --- NORMALISATION SELON FAPS ---
            # GSL (Ratio Longueur pas / Longueur Jambe)
            gsl_r = avg_sl_r / leg_length
            gsl_l = avg_sl_l / leg_length
            
            # GV (Vitesse normalisée par jambe = GSL / Step Time)
            gv_r = gsl_r / avg_st_r
            gv_l = gsl_l / avg_st_l
        
            # --- ALGORITHME DE DÉDUCTION ---
            def get_step_function_penalty(gv_val, gsl_val, st_val):
                # Pénalité progressive si hors des normes (Max ~7.33 pts par paramètre pour atteindre 22)
                p_v = 0 if 1.1 <= gv_val <= 1.5 else min(min(abs(gv_val - 1.1), abs(gv_val - 1.5)) / 0.4 * 7.33, 7.33)
                p_sl = 0 if 0.69 <= gsl_val <= 0.86 else min(min(abs(gsl_val - 0.69), abs(gsl_val - 0.86)) / 0.2 * 7.33, 7.33)
                p_st = 0 if 0.50 <= st_val <= 0.63 else min(min(abs(st_val - 0.50), abs(st_val - 0.63)) / 0.2 * 7.33, 7.33)
                return min(p_v + p_sl + p_st, 22)
        
            # Déductions A et B (Fonctions de pas)
            deduction_A = get_step_function_penalty(gv_l, gsl_l, avg_st_l)
            deduction_B = get_step_function_penalty(gv_r, gsl_r, avg_st_r)
        
            # Déduction C : Asymétrie (Max 8 points)
            diff_asy = np.abs(gsl_r - gsl_l)
            deduction_C = 0 if diff_asy < 0.03 else min(((diff_asy - 0.03) / 0.15) * 8, 8)
        
            # Déduction D : Base de Support Dynamique (Max 8 points)
            # Norme typique assumée entre 5cm et 10cm de large
            if 5 <= avg_dbs <= 10:
                deduction_D = 0
            else:
                dbs_diff = min(abs(avg_dbs - 5), abs(avg_dbs - 10))
                deduction_D = min((dbs_diff / 8) * 8, 8) 
        
            # Déductions E et F : Aides et Dispositifs
            deduction_E = AmbulatoryAids
            deduction_F = AssistiveDevice
        
            # Formule Finale
            total_deductions = deduction_A + deduction_B + deduction_C + deduction_D + deduction_E + deduction_F
            score_faps = 100 - total_deductions
            
            # Plancher théorique du eFAPS
            score_min = 30 if (walking_aids or assistive_devices) else 40
            score_faps = max(score_min, score_faps)
            st.markdown("### 📊 Résultats du score eFAPS")
            st.write(f"Score eFAPS : {score_faps:.2f}")
            st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 95 et 100. Tout score en-dehors indique une atteinte à la fonctionnalité de la marche.")
            
        trials_list = [tmp1_path, tmp2_path, tmp3_path, tmp4_path, tmp5_path]
        calculate_efaps(trials_list, tmp_path, walking_aids=False, assistive_devices=False)

    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
