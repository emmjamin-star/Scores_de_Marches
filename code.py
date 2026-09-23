import streamlit as st
import ezc3d
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
matplotlib.use('Agg') # Rendement graphique sans interface
from scipy.signal import savgol_filter, hilbert, find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile
from math import sqrt
import seaborn as sns

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
        calculate_efaps(trials_list, tmp_path, walking_aids=False, assistive_devices=False)
        # calcul EGVI
        acq_stat = ezc3d.c3d(tmp_path)
        pts_stat = acq_stat['data']['points']
        lbl_stat = acq_stat['parameters']['POINT']['LABELS']['value']
    
        iLPSI, iRPSI = lbl_stat.index('LPSI'), lbl_stat.index('RPSI')
        iLANK, iRANK = lbl_stat.index('LANK'), lbl_stat.index('RANK')
        iLASI, iRASI = lbl_stat.index('LASI'), lbl_stat.index('RASI')
    
        LgJambeR = np.mean(np.linalg.norm(pts_stat[:, iRANK, :] - pts_stat[:, iRPSI, :], axis=0))
        LgJambeL = np.mean(np.linalg.norm(pts_stat[:, iLANK, :] - pts_stat[:, iLPSI, :], axis=0))
        LargeurBassin = np.mean(np.abs(pts_stat[1, iLASI, :] - pts_stat[1, iRASI, :]))
        LgJambe_Moy_m = ((LgJambeL + LgJambeR) / 2) / 1000
        # ==============================================================================
        # CONFIGURATION : LISTE DES FICHIERS A TRAITER
        # ==============================================================================
        # Remplacez les chemins ci-dessous par vos 3 fichiers c3d
        liste_fichiers = trials_list
        
        # Dictionnaires pour stocker les "Différences"
        global_diffs_left = {
            'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []
        }
        global_diffs_right = {
            'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []
        }
        
        # ==============================================================================
        # FONCTIONS UTILITAIRES
        # ==============================================================================
        
        def find_toe_offs_from_toes(z_toe_data, cycle_tuples, threshold_clearance=20.0):
            """ Détecte le Toe Off en cherchant quand l'orteil remonte après avoir été au sol. """
            detected_tos = []
            for (start_frame, end_frame) in cycle_tuples:
                z_segment = z_toe_data[start_frame:end_frame]
                if len(z_segment) == 0: continue
                idx_min = np.argmin(z_segment)
                min_val = z_segment[idx_min]
                post_min_segment = z_segment[idx_min:]
                candidates = np.where(post_min_segment > (min_val + threshold_clearance))[0]
                if len(candidates) > 0:
                    to_frame = start_frame + idx_min + candidates[0]
                    detected_tos.append(to_frame)
            return detected_tos
        
        def calculate_diffs(raw_values, normalization_mean=None):
            """ Étape 1 : Calcule les différences absolues entre pas consécutifs normalisés. """
            if len(raw_values) < 2:
                return []
        
            arr = np.array(raw_values)
            if normalization_mean is None:
                normalization_mean = np.mean(arr)
        
            normalized_p = []
            diffs = []
            for val in arr:
                normalized_p.append((val) / normalization_mean * 100)
        
            for i in range(len(normalized_p) - 1):
                diff = abs(normalized_p[i+1] - normalized_p[i])
                diffs.append(diff)
            return diffs
        
        # ==============================================================================
        # BOUCLE PRINCIPALE SUR LES FICHIERS
        # ==============================================================================
        
        print(f"Début du traitement de {len(liste_fichiers)} fichiers...")
        
        for fichier in liste_fichiers:
            if not os.path.exists(fichier):
                print(f"ATTENTION : Fichier introuvable {fichier}, passage au suivant.")
                continue
        
            # 1. Chargement
            acq1 = ezc3d.c3d(fichier)
            labels = acq1['parameters']['POINT']['LABELS']['value']
            freq = acq1['header']['points']['frame_rate']
            points = acq1['data']['points']
            axis_ap = 0
        
            # 2. Détection Cycles (LHEE / RHEE)
            lhee_valid_cycles, rhee_valid_cycles = [], []
            lhee_cycle_start_indices, rhee_cycle_start_indices = [], []
        
            if "LHEE" in labels:
                idx_lhee = labels.index("LHEE")
                peaks, _ = find_peaks(-points[2, idx_lhee, :], distance=int(freq * 0.8), prominence=1)
                lhee_valid_cycles = [(s, e) for s, e in zip(peaks[:-1], peaks[1:]) if (e - s) >= int(0.5 * freq)]
                lhee_cycle_start_indices = peaks[:-1]
        
            if "RHEE" in labels:
                idx_rhee = labels.index("RHEE")
                peaks, _ = find_peaks(-points[2, idx_rhee, :], distance=int(freq * 0.8), prominence=1)
                rhee_valid_cycles = [(s, e) for s, e in zip(peaks[:-1], peaks[1:]) if (e - s) >= int(0.5 * freq)]
                rhee_cycle_start_indices = peaks[:-1]
        
            # 3. Calcul Step Length
            step_lengths_left, step_lengths_right = [], []
            all_events = sorted([(f, 'Left') for f in lhee_cycle_start_indices] + [(f, 'Right') for f in rhee_cycle_start_indices], key=lambda x: x[0])
        
            if "LHEE" in labels and "RHEE" in labels:
                idx_lhee, idx_rhee = labels.index("LHEE"), labels.index("RHEE")
                for i in range(1, len(all_events)):
                    current_frame, current_side = all_events[i]
                    prev_frame, prev_side = all_events[i-1]
                    if current_side != prev_side:
                        step_len = abs(points[axis_ap, idx_lhee, current_frame] - points[axis_ap, idx_rhee, current_frame])
                        if current_side == 'Left': step_lengths_left.append(step_len/LgJambe_Moy_m*100)
                        else: step_lengths_right.append(step_len/LgJambe_Moy_m*100)
        
            # 4. Calcul Step Time
            step_times_left, step_times_right = [], []
            for i in range(1, len(all_events)):
                current_frame, current_side = all_events[i]
                prev_frame, prev_side = all_events[i-1]
                if current_side != prev_side:
                    step_time_seconds = (current_frame - prev_frame) / freq
                    if current_side == 'Left': step_times_left.append(step_time_seconds)
                    else: step_times_right.append(step_time_seconds)
        
            # 5. Détection Toe Offs
            lhee_toe_offs, rhee_toe_offs = [], []
            if "LTOE" in labels and len(lhee_valid_cycles) > 0:
                lhee_toe_offs = find_toe_offs_from_toes(points[2, labels.index("LTOE"), :], lhee_valid_cycles)
            if "RTOE" in labels and len(rhee_valid_cycles) > 0:
                rhee_toe_offs = find_toe_offs_from_toes(points[2, labels.index("RTOE"), :], rhee_valid_cycles)
        
            # 6. Single Support & Stance Time (Normalisés en % du cycle de marche)
            sst_left, sst_right = [], []
            stance_time_left, stance_time_right = [], []
        
            # --- GAUCHE ---
            # Temps d'appui (Stance Time) Gauche
            for (start, end) in lhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in lhee_toe_offs if start < to < end]
                if next_tos:
                    stance_dur_frames = next_tos[0] - start
                    stance_time_left.append((stance_dur_frames / cycle_dur_frames) * 100)
        
            # Simple Appui Gauche (= Phase oscillante/Swing Droite)
            for (start, end) in rhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in rhee_toe_offs if start < to < end]
                if next_tos:
                    swing_dur_frames = end - next_tos[0]
                    sst_left.append((swing_dur_frames / cycle_dur_frames) * 100)
        
            # --- DROITE ---
            # Temps d'appui (Stance Time) Droit
            for (start, end) in rhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in rhee_toe_offs if start < to < end]
                if next_tos:
                    stance_dur_frames = next_tos[0] - start
                    stance_time_right.append((stance_dur_frames / cycle_dur_frames) * 100)
        
            # Simple Appui Droit (= Phase oscillante/Swing Gauche)
            for (start, end) in lhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in lhee_toe_offs if start < to < end]
                if next_tos:
                    swing_dur_frames = end - next_tos[0]
                    sst_right.append((swing_dur_frames / cycle_dur_frames) * 100)
        
            # 7. Velocity (via Stride)
            def get_velocity(cycles, marker_idx):
                vels = []
                for (start, end) in cycles:
                    dur = (end - start) / freq
                    dist = abs(points[axis_ap, marker_idx, end] - points[axis_ap, marker_idx, start]) / 10
                    if dur > 0: vels.append((dist/dur)/(sqrt(9.81*LgJambe_Moy_m)))
                return vels
        
            velocity_left, velocity_right = [], []
            if len(lhee_valid_cycles) > 0 and "LHEE" in labels:
                velocity_left = get_velocity(lhee_valid_cycles, labels.index("LHEE"))
            if len(rhee_valid_cycles) > 0 and "RHEE" in labels:
                velocity_right = get_velocity(rhee_valid_cycles, labels.index("RHEE"))
        
            # AGGRÉGATION DES DIFFÉRENCES
            global_diffs_left['StepLen'].extend(calculate_diffs(step_lengths_left))
            global_diffs_left['StepTime'].extend(calculate_diffs(step_times_left))
            global_diffs_left['StanceTime'].extend(calculate_diffs(stance_time_left))
            global_diffs_left['SingleSup'].extend(calculate_diffs(sst_left))
            global_diffs_left['Velocity'].extend(calculate_diffs(velocity_left))
        
            global_diffs_right['StepLen'].extend(calculate_diffs(step_lengths_right))
            global_diffs_right['StepTime'].extend(calculate_diffs(step_times_right))
            global_diffs_right['StanceTime'].extend(calculate_diffs(stance_time_right))
            global_diffs_right['SingleSup'].extend(calculate_diffs(sst_right))
            global_diffs_right['Velocity'].extend(calculate_diffs(velocity_right))
        
        print("\n" + "="*30)
        print(" CALCUL GLOBAL EGVI (Tous fichiers)")
        print("="*30)
        
        # ==============================================================================
        # CALCUL FINAL EGVI
        # ==============================================================================
        
        def calculer_egvi(donnees_sujet):
            coeffs = np.array([0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90])
            mean_control_s_alpha = 20.37541132570365
            mean_ln_d_control = 1.3865728033714733
            sd_ln_d_control = 0.6193340454665202
        
            if len(donnees_sujet) != 10: return 0.0, 0.0, 0.0
            donnees_propres = [0.0 if np.isnan(x) else x for x in donnees_sujet]
        
            s_alpha_sujet = np.dot(coeffs, donnees_propres)
            diff = s_alpha_sujet - mean_control_s_alpha
            
            distance_absolue = abs(diff)
            ln_d = math.log(1 + distance_absolue)
        
            z_score = (ln_d - mean_ln_d_control) / sd_ln_d_control
            egvi = 100 + (10 * z_score)
            return egvi, ln_d, s_alpha_sujet
        
        def get_stats_from_diffs(diff_list):
            arr = np.array(diff_list)
            if len(arr) == 0: return 0.0, 0.0
            return np.mean(arr), np.std(arr, ddof=1)
        
        # Construction des vecteurs
        m_sl_r, sd_sl_r = get_stats_from_diffs(global_diffs_right['StepLen'])
        m_st_r, sd_st_r = get_stats_from_diffs(global_diffs_right['StepTime'])
        m_sta_r, sd_sta_r = get_stats_from_diffs(global_diffs_right['StanceTime'])
        m_ss_r, sd_ss_r = get_stats_from_diffs(global_diffs_right['SingleSup'])
        m_vel_r, sd_vel_r = get_stats_from_diffs(global_diffs_right['Velocity'])
        
        valeurs_sujet_D = [m_sl_r, m_st_r, m_sta_r, m_ss_r, m_vel_r, sd_sl_r, sd_st_r, sd_sta_r, sd_ss_r, sd_vel_r]
        
        m_sl_l, sd_sl_l = get_stats_from_diffs(global_diffs_left['StepLen'])
        m_st_l, sd_st_l = get_stats_from_diffs(global_diffs_left['StepTime'])
        m_sta_l, sd_sta_l = get_stats_from_diffs(global_diffs_left['StanceTime'])
        m_ss_l, sd_ss_l = get_stats_from_diffs(global_diffs_left['SingleSup'])
        m_vel_l, sd_vel_l = get_stats_from_diffs(global_diffs_left['Velocity'])
        
        valeurs_sujet_G = [m_sl_l, m_st_l, m_sta_l, m_ss_l, m_vel_l, sd_sl_l, sd_st_l, sd_sta_l, sd_ss_l, sd_vel_l]
        
        # Calcul Final
        egvi_resultat_D, ln_dd, s_alpha_sujetd = calculer_egvi(valeurs_sujet_D)
        egvi_resultat_G, ln_dg, s_alpha_sujetg = calculer_egvi(valeurs_sujet_G)
        EGVItot = (egvi_resultat_D + egvi_resultat_G) / 2
        
        st.markdown("### 📊 Résultats du score eGVI")
        st.write(f"\nRÉSULTATS EGVI AGREGÉS (sur {len(global_diffs_left['StepLen'])} G et {len(global_diffs_right['StepLen'])} D pas cumulés) :")
        st.write(f"Score EGVI Gauche : {egvi_resultat_G:.2f}")
        st.write(f"Score EGVI Droit  : {egvi_resultat_D:.2f}")
        st.write(f"Score EGVI Global : {EGVItot:.2f}")
        st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 95 et 105. Tout score en-dehors indique une atteinte à la variabilité de la marche.")
    
        # Calcul GDP & GVI
        # Thème clinique Seaborn
        sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
        
        class MasterGaitAnalyzer:
            """
            Analyseur maître bilatéral de la marche cinématique :
            1. Extraction avec correction Vicon (LPelvis utilisé pour le bassin G et D).
            2. GDI et GPS séparés pour le côté Gauche et le côté Droit + Moyennes.
            3. Movement Analysis Profile (MAP) bilatéral (Barres groupées).
            """
            def __init__(self, healthy_matrix_path, n_points_per_curve=51):
                self.healthy_matrix_path = healthy_matrix_path
                self.n_points_per_curve = n_points_per_curve
        
                # Configuration des canaux (Nom de base, Index de l'axe spatial 0=X, 1=Y, 2=Z)
                self.gdi_channels_config = [
                    ('PelvisAngles', 0),       # 1. Tilt (X)
                    ('PelvisAngles', 1),       # 2. Obliquité (Y)
                    ('PelvisAngles', 2),       # 3. Rotation (Z)
                    ('HipAngles', 0),          # 4. Flexion (X)
                    ('HipAngles', 1),          # 5. Adduction (Y)
                    ('HipAngles', 2),          # 6. Rotation de hanche (Z)
                    ('KneeAngles', 0),         # 7. Flexion de genou (X)
                    ('AnkleAngles', 0),        # 8. Dorsiflexion de cheville (X)
                    ('FootProgressAngles', 2)  # 9. Angle de progression du pied (Z)
                ]
        
                self.gvs_labels = [
                    'Pelvic Tilt', 'Pelvic Obliquity', 'Pelvic Rotation',
                    'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
                    'Knee Flexion', 'Ankle Dorsiflexion', 'Foot Progression'
                ]
        
                self.expected_rows = len(self.gdi_channels_config) * self.n_points_per_curve
                self._load_and_fit_healthy_reference()
        
            def _load_and_fit_healthy_reference(self):
                """Charge la matrice saine (459 points) servant de référence normative."""
                if not os.path.exists(self.healthy_matrix_path):
                    raise FileNotFoundError(f"Matrice saine '{self.healthy_matrix_path}' introuvable.")
        
                if self.healthy_matrix_path.endswith('.npy'):
                    self.healthy_matrix = np.load(self.healthy_matrix_path)
                elif self.healthy_matrix_path.endswith('.csv'):
                    self.healthy_matrix = np.loadtxt(self.healthy_matrix_path, delimiter=',')
        
                n_temoins = self.healthy_matrix.shape[1]
        
                # Moyennes normatives pour le GPS
                self.mean_healthy_vector = np.mean(self.healthy_matrix, axis=1, keepdims=True)
                self.healthy_mean_curves = self.mean_healthy_vector.reshape(len(self.gdi_channels_config), self.n_points_per_curve)
        
                # Modèle SVD pour le GDI
                centered_healthy = self.healthy_matrix - self.mean_healthy_vector
                U, S, Vt = np.linalg.svd(centered_healthy, full_matrices=False)
                self.feature_base = U[:, :15]
        
                projected_healthy = np.dot(self.feature_base.T, centered_healthy)
                raw_distances_healthy = np.sqrt(np.sum(projected_healthy**2, axis=0))
        
                self.mean_raw_healthy_gdi = np.mean(raw_distances_healthy)
                self.std_raw_healthy_gdi = np.std(raw_distances_healthy)
        
                print(f"✅ Modèle bilatéral GDI & GPS initialisé ({n_temoins} témoins).\n")
        
            def _resample_curve(self, curve_data):
                """Rééchantillonne sur 51 points et gère les potentiels NaN."""
                if np.isnan(curve_data).any():
                    curve_data = np.nan_to_num(curve_data, nan=np.nanmean(curve_data))
                x_old = np.linspace(0, 100, len(curve_data))
                x_new = np.linspace(0, 100, self.n_points_per_curve)
                return interp1d(x_old, curve_data, kind='cubic', fill_value="extrapolate")(x_new)
        
            def _extract_and_trim_channel(self, points_data, point_labels, target_label, axis_idx):
                """Trouve le canal, extrait l'axe demandé, et supprime les zéros de fin (padding)."""
                if target_label not in point_labels:
                    raise KeyError(f"Canal '{target_label}' introuvable.")
        
                idx = point_labels.index(target_label)
                raw_curve = points_data[axis_idx, idx, :]
        
                non_zero_indices = np.nonzero(raw_curve)[0]
                if len(non_zero_indices) == 0:
                    raise ValueError(f"Canal '{target_label}' entièrement vide.")
        
                return raw_curve[:non_zero_indices[-1] + 1]
        
            def extract_kinematics_from_c3d(self, c3d_filepath):
                """Extrait et sépare la cinématique Gauche et Droite."""
                if ezc3d is None:
                    raise ImportError("Installez 'ezc3d' (`pip install ezc3d`).")
        
                c3d = ezc3d.c3d(c3d_filepath)
                point_labels = [label.strip() for label in c3d['parameters']['POINT']['LABELS']['value']]
                points_data = c3d['data']['points']
        
                curves_L = np.zeros((len(self.gdi_channels_config), self.n_points_per_curve))
                curves_R = np.zeros((len(self.gdi_channels_config), self.n_points_per_curve))
        
                for i, (base_label, axis_idx) in enumerate(self.gdi_channels_config):
                    # CORRECTION BASSIN : Utilisation exclusive de LPelvis pour G et D
                    if base_label == 'PelvisAngles':
                        label_L = "LPelvisAngles"
                        label_R = "LPelvisAngles"
                    else:
                        label_L = f"L{base_label}"
                        label_R = f"R{base_label}"
        
                    raw_L = self._extract_and_trim_channel(points_data, point_labels, label_L, axis_idx)
                    raw_R = self._extract_and_trim_channel(points_data, point_labels, label_R, axis_idx)
        
                    curves_L[i, :] = self._resample_curve(raw_L)
                    curves_R[i, :] = self._resample_curve(raw_R)
        
                return curves_L, curves_R
        
            def compute_gdi_trial(self, patient_vector_col):
                """Calcule le Z-score GDI."""
                centered = patient_vector_col - self.mean_healthy_vector
                projected = np.dot(self.feature_base.T, centered)
                raw_dist = np.sqrt(np.sum(projected**2))
                z_score = (raw_dist - self.mean_raw_healthy_gdi) / self.std_raw_healthy_gdi
                return 100.0 - (10.0 * z_score)
        
            def compute_gps_trial(self, patient_curves_matrix):
                """Calcule les 9 GVS et le GPS."""
                gvs = np.sqrt(np.mean((patient_curves_matrix - self.healthy_mean_curves)**2, axis=1))
                gps = np.sqrt(np.mean(gvs**2))
                return gps, gvs
        
            def generate_bilateral_map_chart(self, gvs_L, gvs_R, gps_L, gps_R, output_filename):
                """Génère le graphique MAP comparant la Gauche et la Droite."""
                labels = self.gvs_labels[::-1]
                val_L = gvs_L[::-1]
                val_R = gvs_R[::-1]
        
                y = np.arange(len(labels))
                height = 0.38
        
                fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
                # Tracé des barres (Rouge = Gauche, Vert = Droit) convention clinique
                bars_L = ax.barh(y + height/2, val_L, height, label=f'Côté Gauche (GPS: {gps_L:.1f}°)', color='#d62728')
                bars_R = ax.barh(y - height/2, val_R, height, label=f'Côté Droit (GPS: {gps_R:.1f}°)', color='#2ca02c')
        
                # Ajout des valeurs numériques
                for bar in bars_L:
                    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}',
                            va='center', fontsize=9, color='#d62728', fontweight='bold')
                for bar in bars_R:
                    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}',
                            va='center', fontsize=9, color='#2ca02c', fontweight='bold')
        
                # Ligne de référence
                ax.axvline(5.4, color='gray', linestyle='--', linewidth=1.5, label='Référence Normative saine (5.4°)')
        
                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
                ax.set_xlabel('Gait Variable Score - GVS (°)', fontsize=11, fontweight='bold')
                ax.set_title("Movement Analysis Profile (MAP) Bilatéral", fontsize=14, fontweight='bold', pad=15)
        
                ax.set_xlim(0, max(max(val_L), max(val_R)) + 3.5)
                ax.legend(loc='lower right', frameon=True, facecolor='white')
                sns.despine(left=True)
        
                plt.tight_layout()
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"📊 Graphique MAP sauvegardé sous '{output_filename}'")
        
            def run_full_analysis(self, c3d_files_list, output_chart_path='map_profile_bilateral.png'):
                """Analyse complète moyennée sur les 5 essais."""
                scores = {'GDI_L': [], 'GDI_R': [], 'GPS_L': [], 'GPS_R': [], 'GVS_L': [], 'GVS_R': []}
        
                st.write("=========================================================")
                st.write("     ANALYSE GLOBALE BILATÉRALE DE LA MARCHE (L / R)     ")
                st.write("=========================================================")
        
                for i, filepath in enumerate(c3d_files_list, 1):
                    curves_L, curves_R = self.extract_kinematics_from_c3d(filepath)
        
                    # GDI
                    gdi_L = self.compute_gdi_trial(curves_L.reshape(-1, 1))
                    gdi_R = self.compute_gdi_trial(curves_R.reshape(-1, 1))
        
                    # GPS / GVS
                    gps_L, gvs_L = self.compute_gps_trial(curves_L)
                    gps_R, gvs_R = self.compute_gps_trial(curves_R)
        
                    scores['GDI_L'].append(gdi_L); scores['GDI_R'].append(gdi_R)
                    scores['GPS_L'].append(gps_L); scores['GPS_R'].append(gps_R)
                    scores['GVS_L'].append(gvs_L); scores['GVS_R'].append(gvs_R)
        
                    st.write(f"Essai {i} ({os.path.basename(filepath)}) :")
                    st.write(f"  • Gauche -> GDI: {gdi_L:5.1f} | GPS: {gps_L:4.1f}°")
                    st.write(f"  • Droit  -> GDI: {gdi_R:5.1f} | GPS: {gps_R:4.1f}°")
        
                # Moyennes Globales
                m_gdi_L = np.mean(scores['GDI_L']); m_gdi_R = np.mean(scores['GDI_R'])
                m_gps_L = np.mean(scores['GPS_L']); m_gps_R = np.mean(scores['GPS_R'])
        
                m_gvs_L = np.mean(np.array(scores['GVS_L']), axis=0)
                m_gvs_R = np.mean(np.array(scores['GVS_R']), axis=0)
        
                mean_gdi_overall = (m_gdi_L + m_gdi_R) / 2.0
                mean_gps_overall = (m_gps_L + m_gps_R) / 2.0
                
                st.markdown("### 📊 GAIT DEVIATION INDEX (GDI)")
                st.write(f"Gauche : {m_gdi_L:.1f}  |  Droit : {m_gdi_R:.1f}  |  Moyenne Globale : {mean_gdi_overall:.1f}")
                st.markdown("### 📐 GAIT PROFILE SCORE (GPS")
                st.write(f"Gauche : {m_gps_L:.1f}° |  Droit : {m_gps_R:.1f}° |  Moyenne Globale : {mean_gps_overall:.1f}")
                st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 95 et 105. Tout score en-dehors indique une atteinte à la variabilité de la marche.")
                self.generate_bilateral_map_chart(m_gvs_L, m_gvs_R, m_gps_L, m_gps_R, output_chart_path)
    
                return {
                    'GDI': {'Left': m_gdi_L, 'Right': m_gdi_R, 'Overall': mean_gdi_overall},
                    'GPS': {'Left': m_gps_L, 'Right': m_gps_R, 'Overall': mean_gps_overall},
                    'GVS_Left': dict(zip(self.gvs_labels, m_gvs_L)),
                    'GVS_Right': dict(zip(self.gvs_labels, m_gvs_R))
                }
        if __name__ == "__main__":
            matrice_saine = "/Scores_de_Marches/matrice_temoins_459.npy"
    
        # Remplacez par vos fichiers
            essais_patient = trials_list
            analyzer = MasterGaitAnalyzer(healthy_matrix_path=matrice_saine)
            results = analyzer.run_full_analysis(essais_patient, output_chart_path="map_profile_bilateral.png")
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
