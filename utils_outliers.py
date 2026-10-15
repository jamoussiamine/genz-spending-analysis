"""
Module centralisé pour la gestion des outliers
Évite la redondance de code dans les différents fichiers du projet
"""

import pandas as pd
import numpy as np
from scipy import stats


def detect_outliers_iqr(df, columns, factor=1.5):
    """
    Détecte les outliers en utilisant la méthode IQR (Interquartile Range)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list
        Liste des colonnes numériques à analyser
    factor : float, default=1.5
        Facteur multiplicatif pour définir les limites (1.5 = standard, 3 = strict)
    
    Returns:
    --------
    dict : Dictionnaire avec les indices des outliers par colonne
    """
    outliers_dict = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            outliers_dict[col] = outliers
    
    return outliers_dict


def detect_outliers_zscore(df, columns, threshold=3):
    """
    Détecte les outliers en utilisant la méthode Z-score
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list
        Liste des colonnes numériques à analyser
    threshold : float, default=3
        Seuil Z-score (3 = standard, 2.5 = moins strict)
    
    Returns:
    --------
    dict : Dictionnaire avec les indices des outliers par colonne
    """
    outliers_dict = {}
    
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[z_scores > threshold].index.tolist()
            outliers_dict[col] = outliers
    
    return outliers_dict


def remove_outliers_iqr(df, columns, factor=1.5, inplace=False):
    """
    Supprime les outliers en utilisant la méthode IQR
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list
        Liste des colonnes numériques à traiter
    factor : float, default=1.5
        Facteur multiplicatif pour définir les limites
    inplace : bool, default=False
        Si True, modifie le DataFrame directement
    
    Returns:
    --------
    pandas.DataFrame : DataFrame sans outliers
    """
    if inplace:
        df_clean = df
    else:
        df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Supprimer les outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean


def cap_outliers_iqr(df, columns, factor=1.5, inplace=False):
    """
    Limite (cap) les outliers en utilisant la méthode IQR (winsorization)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list
        Liste des colonnes numériques à traiter
    factor : float, default=1.5
        Facteur multiplicatif pour définir les limites
    inplace : bool, default=False
        Si True, modifie le DataFrame directement
    
    Returns:
    --------
    pandas.DataFrame : DataFrame avec outliers limités
    """
    if inplace:
        df_clean = df
    else:
        df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Limiter les valeurs extrêmes
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean


def get_outlier_summary(df, columns, method='iqr', factor=1.5, threshold=3):
    """
    Génère un résumé des outliers détectés
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list
        Liste des colonnes numériques à analyser
    method : str, default='iqr'
        Méthode de détection ('iqr' ou 'zscore')
    factor : float, default=1.5
        Facteur pour IQR
    threshold : float, default=3
        Seuil pour Z-score
    
    Returns:
    --------
    pandas.DataFrame : Résumé des outliers par colonne
    """
    if method == 'iqr':
        outliers_dict = detect_outliers_iqr(df, columns, factor)
    else:
        outliers_dict = detect_outliers_zscore(df, columns, threshold)
    
    summary = []
    for col, indices in outliers_dict.items():
        summary.append({
            'Column': col,
            'Outliers_Count': len(indices),
            'Outliers_Percentage': (len(indices) / len(df)) * 100 if len(df) > 0 else 0,
            'Outlier_Indices': indices[:10]  # Limiter à 10 pour l'affichage
        })
    
    return pd.DataFrame(summary)

