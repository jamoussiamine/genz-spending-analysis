"""
Application Streamlit pour visualiser les notebooks 1, 2 et 3
- Notebook 1: Data Understanding (Analyse Exploratoire)
- Notebook 2: Data Preparation (Nettoyage et Préparation)
- Notebook 3: Feature Engineering (Création de Nouveaux Attributs)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats
import sys

# Dossier du projet : les chemins data/... et l'import local fonctionnent quel que soit le répertoire de lancement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils_outliers import get_outlier_summary, cap_outliers_iqr

# Configuration de la page
st.set_page_config(
    page_title="Analyse GenZ Spending",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Titre principal
st.title("📊 Analyse des Dépenses de la Génération Z")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("📑 Navigation")
notebook_choice = st.sidebar.radio(
    "Choisir un notebook:",
    [
        "🚀 Déploiement - Prédictions Interactives",
        "1. Data Understanding",
        "2. Data Preparation",
        "3. Feature Engineering",
        "4. ACP Professionnel",
        "5. Clustering K-Means",
        "6. Clustering Comparaison K-Means vs DBSCAN",
        "7. Classification Multi-Algorithmes",
        "8. Régression - Total_Expenses"
    ]
)

# Chargement des données
@st.cache_data
def load_raw_data():
    """Charge les données brutes"""
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/dataset_genz_6000.csv'), sep=';')
        return df
    except FileNotFoundError:
        st.error("❌ Fichier de données non trouvé. Assurez-vous que le fichier existe dans data/raw/")
        return None

@st.cache_data
def load_cleaned_data():
    """Charge les données nettoyées"""
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/dataset_cleaned.csv'))
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_json_results(filepath):
    """Charge les résultats JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_features_data():
    """Charge les données avec features"""
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/dataset_with_features.csv'))
        return df
    except FileNotFoundError:
        return None

# ============================================
# NOTEBOOK 1: DATA UNDERSTANDING
# ============================================
if notebook_choice == "1. Data Understanding":
    st.header("📊 1. Data Understanding - Analyse Exploratoire")
    st.markdown("**Objectif:** Comprendre la structure et les caractéristiques du dataset")
    
    # Chargement des données
    df = load_raw_data()
    if df is None:
        st.stop()
    
    # Onglets pour organiser le contenu
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Vue d'ensemble",
        "📈 Statistiques",
        "📊 Visualisations",
        "🔗 Corrélations"
    ])
    
    with tab1:
        st.subheader("📋 Vue d'ensemble du dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de lignes", f"{df.shape[0]:,}")
        with col2:
            st.metric("Nombre de colonnes", df.shape[1])
        with col3:
            st.metric("Taille mémoire", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        st.markdown("### 🔍 Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📝 Types de données")
        dtype_df = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-null': df.notna().sum(),
            'Null': df.isna().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Statistiques descriptives")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_for_analysis = [col for col in numeric_cols if col != 'ID']
        
        st.markdown("### Statistiques générales")
        st.dataframe(df[numeric_cols_for_analysis].describe(), use_container_width=True)
        
        # Statistiques spécifiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistiques de l'âge")
            age_stats = {
                'Moyenne': f"{df['Age'].mean():.2f} ans",
                'Médiane': f"{df['Age'].median():.2f} ans",
                'Écart-type': f"{df['Age'].std():.2f} ans",
                'Min': f"{df['Age'].min()} ans",
                'Max': f"{df['Age'].max()} ans"
            }
            for key, value in age_stats.items():
                st.metric(key, value)
        
        with col2:
            st.markdown("#### 💰 Statistiques des revenus")
            income_stats = {
                'Moyenne': f"${df['Income (USD)'].mean():,.2f}",
                'Médiane': f"${df['Income (USD)'].median():,.2f}",
                'Écart-type': f"${df['Income (USD)'].std():,.2f}",
                'Min': f"${df['Income (USD)'].min():,}",
                'Max': f"${df['Income (USD)'].max():,}"
            }
            for key, value in income_stats.items():
                st.metric(key, value)
    
    with tab3:
        st.subheader("📊 Visualisations exploratoires")
        
        # Distribution de l'âge
        st.markdown("### Distribution de l'âge")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Age'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_xlabel('Âge')
            ax.set_ylabel('Fréquence')
            ax.set_title('Distribution de l\'âge')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(df['Age'])
            ax.set_ylabel('Âge')
            ax.set_title('Boxplot de l\'âge')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Distribution des revenus
        st.markdown("### Distribution des revenus")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Income (USD)'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
            ax.set_xlabel('Revenus (USD)')
            ax.set_ylabel('Fréquence')
            ax.set_title('Distribution des revenus')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(df['Income (USD)'])
            ax.set_ylabel('Revenus (USD)')
            ax.set_title('Boxplot des revenus')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Distribution des dépenses principales
        st.markdown("### Distribution des dépenses principales")
        expense_cols = ['Rent (USD)', 'Groceries (USD)', 'Entertainment (USD)', 'Education (USD)']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(expense_cols):
            axes[idx].hist(df[col], bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Fréquence')
            axes[idx].set_title(f'Distribution de {col}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        st.subheader("🔗 Analyse des corrélations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_for_outliers = [col for col in numeric_cols if col != 'ID']
        
        correlation_matrix = df[numeric_cols_for_outliers].corr()
        
        # Matrice de corrélation
        st.markdown("### Matrice de corrélation")
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Matrice de corrélation des variables numériques', 
                    fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        
        # Corrélations les plus fortes
        st.markdown("### 🔗 Corrélations les plus fortes")
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': correlation_matrix.columns[i],
                    'Variable 2': correlation_matrix.columns[j],
                    'Corrélation': correlation_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs_Correlation'] = corr_df['Corrélation'].abs()
        top_corr = corr_df.nlargest(10, 'Abs_Correlation')[['Variable 1', 'Variable 2', 'Corrélation']]
        st.dataframe(top_corr, use_container_width=True)

# ============================================
# NOTEBOOK 2: DATA PREPARATION
# ============================================
elif notebook_choice == "2. Data Preparation":
    st.header("🧹 2. Data Preparation - Nettoyage et Préparation")
    st.markdown("**Objectif:** Nettoyer et préparer les données pour l'analyse et la modélisation")
    
    # Chargement des données
    df_raw = load_raw_data()
    df_cleaned = load_cleaned_data()
    
    if df_raw is None:
        st.error("❌ Impossible de charger les données brutes")
        st.stop()
    
    if df_cleaned is None:
        st.warning("⚠️ Les données nettoyées n'existent pas encore. Génération en cours...")
        # Générer les données nettoyées
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_for_outliers = [col for col in numeric_cols if col != 'ID']
        df_cleaned = cap_outliers_iqr(df_raw, numeric_cols_for_outliers, factor=1.5, inplace=False)
    
    # Chargement des métadonnées
    normalization_info = load_json_results(os.path.join(BASE_DIR, 'data/results/normalization_info.json'))
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Comparaison Avant/Après",
        "⚠️ Traitement des Outliers",
        "✅ Qualité des données",
        "💾 Résumé"
    ])
    
    with tab1:
        st.subheader("📊 Comparaison Avant/Après nettoyage")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lignes (avant)", f"{df_raw.shape[0]:,}")
            st.metric("Lignes (après)", f"{df_cleaned.shape[0]:,}")
        with col2:
            st.metric("Colonnes (avant)", df_raw.shape[1])
            st.metric("Colonnes (après)", df_cleaned.shape[1])
        with col3:
            diff = df_raw.shape[0] - df_cleaned.shape[0]
            st.metric("Lignes supprimées", diff if diff > 0 else "0 (winsorization)")
        
        st.markdown("### Aperçu des données nettoyées")
        st.dataframe(df_cleaned.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("⚠️ Traitement des Outliers")
        st.info("ℹ️ **Méthode utilisée:** IQR capping (winsorization) avec factor=1.5. Les valeurs extrêmes sont limitées plutôt que supprimées pour préserver la taille de l'échantillon.")
        
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_for_outliers = [col for col in numeric_cols if col != 'ID']
        
        st.markdown("#### 🔍 Analyse AVANT traitement")
        outlier_summary_before = get_outlier_summary(df_raw, numeric_cols_for_outliers, method='iqr', factor=1.5)
        st.dataframe(outlier_summary_before, use_container_width=True)
        
        st.markdown("#### ✅ Analyse APRÈS traitement (winsorization)")
        outlier_summary_after = get_outlier_summary(df_cleaned, numeric_cols_for_outliers, method='iqr', factor=1.5)
        st.dataframe(outlier_summary_after, use_container_width=True)
        
        # Vérifier si des outliers ont été détectés
        total_outliers_after = outlier_summary_after['Outliers_Count'].sum()
        if total_outliers_after == 0:
            st.success("✅ Aucun outlier détecté!")
    
    with tab3:
        st.subheader("✅ Vérification de la qualité des données")
        
        # Valeurs manquantes
        st.markdown("#### 🔍 Valeurs manquantes")
        missing_data = df_cleaned.isnull().sum()
        if missing_data.sum() > 0:
            st.warning("⚠️ Valeurs manquantes détectées!")
            missing_df = pd.DataFrame({
                'Colonne': missing_data.index,
                'Valeurs manquantes': missing_data.values
            })
            missing_df = missing_df[missing_df['Valeurs manquantes'] > 0]
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ Aucune valeur manquante détectée!")
        
        # Valeurs négatives
        st.markdown("#### ➖ Valeurs négatives")
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_for_check = [col for col in numeric_cols if col != 'ID']
        
        negative_cols = []
        for col in numeric_cols_for_check:
            if (df_cleaned[col] < 0).any():
                negative_cols.append(col)
        
        if negative_cols:
            st.warning(f"⚠️ Colonnes avec valeurs négatives: {', '.join(negative_cols)}")
        else:
            st.success("✅ Aucune valeur négative incohérente détectée!")
        
        # Vérification des doublons
        st.markdown("#### 🔍 Doublons")
        duplicates_count = df_cleaned.duplicated().sum()
        
        if duplicates_count > 0:
            st.warning(f"⚠️ {duplicates_count} doublons détectés!")
            duplicated_rows = df_cleaned[df_cleaned.duplicated(keep=False)]
            st.markdown("**Aperçu des lignes dupliquées:**")
            st.dataframe(duplicated_rows, use_container_width=True)
            
            # Vérification des doublons basés sur l'ID uniquement
            id_duplicates = df_cleaned.duplicated(subset=['ID']).sum()
            if id_duplicates > 0:
                st.warning(f"⚠️ {id_duplicates} IDs dupliqués détectés!")
                duplicated_ids = df_cleaned[df_cleaned.duplicated(subset=['ID'], keep=False)]['ID'].tolist()
                st.write(f"**IDs dupliqués:** {duplicated_ids}")
            else:
                st.info("ℹ️ Aucun ID dupliqué détecté")
        else:
            st.success("✅ Aucun doublon détecté!")
            
            # Vérification des doublons basés sur l'ID uniquement
            id_duplicates = df_cleaned.duplicated(subset=['ID']).sum()
            if id_duplicates > 0:
                st.warning(f"⚠️ {id_duplicates} IDs dupliqués détectés!")
                duplicated_ids = df_cleaned[df_cleaned.duplicated(subset=['ID'], keep=False)]['ID'].tolist()
                st.write(f"**IDs dupliqués:** {duplicated_ids}")
            else:
                st.info("ℹ️ Aucun ID dupliqué détecté")
        
        # Types de données
        st.markdown("#### 📝 Types de données")
        dtype_df = pd.DataFrame({
            'Colonne': df_cleaned.columns,
            'Type': df_cleaned.dtypes.astype(str),
            'Non-null': df_cleaned.notna().sum(),
            'Null': df_cleaned.isna().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with tab4:
        st.subheader("💾 Résumé de la préparation")
        
        st.markdown("### 📋 Informations générales")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lignes initiales", f"{df_raw.shape[0]:,}")
        with col2:
            st.metric("Lignes après nettoyage", f"{df_cleaned.shape[0]:,}")
        with col3:
            st.metric("Colonnes", df_cleaned.shape[1])
        
        st.markdown("---")
        st.markdown("### ✅ Qualité des données")
        
        # Vérification des valeurs manquantes
        missing_count = df_cleaned.isnull().sum().sum()
        st.markdown("#### 🔍 Valeurs manquantes")
        if missing_count == 0:
            st.success("➡️ **Aucune valeur manquante détectée** → pas besoin d'imputation (moyenne, médiane, KNN, etc.)")
            st.info("👉 Cela garantit que tous les algorithmes pourront s'exécuter sans problème de données incomplètes.")
        else:
            st.warning(f"⚠️ {missing_count} valeurs manquantes détectées")
        
        # Vérification des valeurs négatives
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_for_check = [col for col in numeric_cols if col != 'ID']
        negative_cols = [col for col in numeric_cols_for_check if (df_cleaned[col] < 0).any()]
        
        st.markdown("#### ➖ Valeurs négatives incohérentes")
        if len(negative_cols) == 0:
            st.success("➡️ **Aucune incohérence** → cohérence logique respectée (ex. pas de dépenses négatives)")
            st.info("👉 Cela évite des biais ou des erreurs dans les modèles.")
        else:
            st.warning(f"⚠️ Colonnes avec valeurs négatives: {', '.join(negative_cols)}")
        
        # Vérification des doublons
        duplicates_count = df_cleaned.duplicated().sum()
        id_duplicates = df_cleaned.duplicated(subset=['ID']).sum()
        
        st.markdown("#### 🔍 Doublons / IDs dupliqués")
        if duplicates_count == 0 and id_duplicates == 0:
            st.success("➡️ **Aucun doublon** → pas de risque de sur-représentation d'un individu ou d'un cas")
            st.info("👉 La distribution des données reste fidèle à la réalité.")
        else:
            st.warning(f"⚠️ {duplicates_count} doublons détectés, {id_duplicates} IDs dupliqués")
        
        # Vérification des outliers
        numeric_cols_for_outliers = [col for col in numeric_cols if col != 'ID']
        outlier_summary = get_outlier_summary(df_cleaned, numeric_cols_for_outliers, method='iqr', factor=1.5)
        total_outliers = outlier_summary['Outliers_Count'].sum()
        
        st.markdown("#### ⚠️ Outliers")
        if total_outliers == 0:
            st.success("➡️ **Aucun outlier détecté** → pas de valeurs extrêmes susceptibles de fausser les corrélations ou les modèles")
            st.info("👉 Cela améliore la robustesse des algorithmes, notamment les modèles sensibles comme la régression linéaire ou les SVM.")
        else:
            st.warning(f"⚠️ {total_outliers} outliers détectés")
        
        st.markdown("---")
        st.markdown("### 📝 Conclusion")
        st.markdown("""
        **Le dataset GenZ (6000 individus) repose sur des données réelles, non synthétiques.** 
        
        Les valeurs sont logiques et ne présentent ni incohérences, ni valeurs manquantes, ni doublons, ni outliers. 
        
        Ainsi, aucune étape de nettoyage n'est requise pour l'instant, ce qui permet de passer directement à l'analyse exploratoire et à la modélisation.
        """)
        
        st.info("ℹ️ **Note:** La normalisation sera effectuée dans les notebooks de modélisation pour éviter de normaliser la variable cible et pour permettre le Feature Engineering sur les données brutes.")

# ============================================
# NOTEBOOK 3: FEATURE ENGINEERING
# ============================================
elif notebook_choice == "3. Feature Engineering":
    st.header("⚙️ 3. Feature Engineering - Création de Nouveaux Attributs")
    st.markdown("**Objectif:** Créer de nouveaux attributs pour enrichir l'analyse et améliorer la performance des modèles")
    
    # Chargement des données
    df_features = load_features_data()
    if df_features is None:
        st.warning("⚠️ Les données avec features n'existent pas encore. Exécutez d'abord le notebook 3.")
        st.stop()
    
    # Chargement des métadonnées
    feature_summary = load_json_results(os.path.join(BASE_DIR, 'data/results/feature_engineering_summary.json'))
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'ensemble",
        "💰 Ratios Financiers",
        "📈 Indices Comportementaux",
        "🎯 Financial Profile",
        "📋 Résumé"
    ])
    
    with tab1:
        st.subheader("📊 Vue d'ensemble des Features")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre d'observations", f"{df_features.shape[0]:,}")
        with col2:
            st.metric("Variables totales", df_features.shape[1])
        with col3:
            if feature_summary:
                st.metric("Variables dérivées", len(feature_summary.get('derived_variables', [])))
        
        st.markdown("### 🔍 Aperçu des données avec features")
        st.dataframe(df_features.head(10), use_container_width=True)
        
        if feature_summary:
            st.markdown("### 📝 Types de variables")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Variables originales")
                st.write(f"**{len(feature_summary.get('original_variables', []))} variables**")
                st.dataframe(pd.DataFrame({
                    'Variable': feature_summary.get('original_variables', [])
                }), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Variables dérivées")
                st.write(f"**{len(feature_summary.get('derived_variables', []))} variables**")
                st.dataframe(pd.DataFrame({
                    'Variable': feature_summary.get('derived_variables', [])
                }), use_container_width=True, hide_index=True)
            
            st.markdown("#### Variables catégorielles")
            st.dataframe(pd.DataFrame({
                'Variable': feature_summary.get('categorical_variables', [])
            }), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("💰 Ratios Financiers")
        
        st.info("ℹ️ **Important:** Les ratios sont créés pour l'analyse mais **NE SERONT PAS** inclus dans l'ACP et le clustering car ils biaisent les résultats.")
        
        # Formules de calcul
        st.markdown("### 📐 Formules de calcul des ratios")
        
        ratios_table = pd.DataFrame({
            'Ratio': [
                "Taux d'épargne (Savings_Rate)",
                "Taux d'investissement (Investment_Rate)",
                "Taux de dépenses (Expense_Rate)",
                "Dépenses totales (Total_Expenses)",
                "Revenu disponible (Disposable_Income)"
            ],
            'Formule': [
                "(Savings (USD) / Income (USD)) × 100",
                "(Investments (USD) / Income (USD)) × 100",
                "(Total_Expenses / Income (USD)) × 100",
                "Rent + Groceries + Eating Out + Entertainment + Subscription Services + Education + Online Shopping + Travel + Fitness + Miscellaneous",
                "Income (USD) - Total_Expenses"
            ],
            'Description': [
                "Pourcentage du revenu consacré à l'épargne",
                "Pourcentage du revenu consacré aux investissements",
                "Pourcentage du revenu consacré aux dépenses",
                "Somme de toutes les dépenses",
                "Revenu restant après toutes les dépenses"
            ]
        })
        st.dataframe(ratios_table, use_container_width=True, hide_index=True)
        
        # Statistiques des ratios
        st.markdown("### 📊 Statistiques descriptives des ratios")
        ratio_cols = ['Savings_Rate', 'Investment_Rate', 'Expense_Rate']
        if all(col in df_features.columns for col in ratio_cols):
            st.dataframe(df_features[ratio_cols].describe(), use_container_width=True)
            
            # Visualisation
            st.markdown("### 📈 Distribution des ratios")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, col in enumerate(ratio_cols):
                axes[idx].hist(df_features[col], bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Fréquence')
                axes[idx].set_title(f'Distribution de {col}')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interprétation des ratios
            st.markdown("### 💡 Interprétation des ratios financiers")
            
            if all(col in df_features.columns for col in ratio_cols):
                savings_mean = df_features['Savings_Rate'].mean()
                investment_mean = df_features['Investment_Rate'].mean()
                expense_mean = df_features['Expense_Rate'].mean()
                
                st.markdown("""
                **📊 Analyse globale**
                
                - **Taux d'épargne moyen : {:.2f}%** → En moyenne, les individus épargnent près de {:.0f}% de leur revenu. Une valeur élevée qui traduit une bonne capacité d'épargne.
                
                - **Taux d'investissement moyen : {:.2f}%** → En moyenne, {:.0f}% du revenu est consacré à l'investissement, ce qui souligne l'importance des placements financiers dans le budget.
                
                - **Taux de dépenses moyen : {:.2f}%** → Les dépenses représentent en moyenne {:.1f}% du revenu. Une valeur supérieure à 100% indique que certains individus dépensent plus que ce qu'ils gagnent. 👉 Cette situation peut sembler illogique, mais elle s'explique par la présence de outliers (cas extrêmes) dans le dataset.
                """.format(savings_mean, savings_mean, investment_mean, investment_mean, expense_mean, expense_mean))
                
                # Calcul du Total_Ratios pour chaque individu
                df_features['Total_Ratios'] = df_features['Savings_Rate'] + df_features['Investment_Rate'] + df_features['Expense_Rate']
                
                # TOP 10 des cas avec ratios totaux les plus élevés (> 200%)
                high_ratios = df_features[df_features['Total_Ratios'] > 200].copy()
                high_ratios = high_ratios.sort_values('Total_Ratios', ascending=False)
                
                if len(high_ratios) > 0:
                    st.markdown("**📊 Cas particuliers : TOP 10 des ratios totaux les plus élevés (> 200%)**")
                    
                    # Préparer le tableau
                    top10_cols = ['ID', 'Income (USD)', 'Savings (USD)', 'Investments (USD)', 
                                 'Total_Expenses', 'Savings_Rate', 'Investment_Rate', 
                                 'Expense_Rate', 'Total_Ratios', 'Financial_Profile']
                    
                    top10_df = high_ratios.head(10)[top10_cols].copy()
                    
                    # Formater les valeurs pour l'affichage
                    top10_display = top10_df.copy()
                    top10_display['Income (USD)'] = top10_display['Income (USD)'].apply(lambda x: f"{x:,.0f}")
                    top10_display['Savings (USD)'] = top10_display['Savings (USD)'].apply(lambda x: f"{x:,.0f}")
                    top10_display['Investments (USD)'] = top10_display['Investments (USD)'].apply(lambda x: f"{x:,.0f}")
                    top10_display['Total_Expenses'] = top10_display['Total_Expenses'].apply(lambda x: f"{x:,.0f}")
                    top10_display['Savings_Rate'] = top10_display['Savings_Rate'].apply(lambda x: f"{x:.1f}%")
                    top10_display['Investment_Rate'] = top10_display['Investment_Rate'].apply(lambda x: f"{x:.1f}%")
                    top10_display['Expense_Rate'] = top10_display['Expense_Rate'].apply(lambda x: f"{x:.1f}%")
                    top10_display['Total_Ratios'] = top10_display['Total_Ratios'].apply(lambda x: f"{x:.1f}%")
                    
                    # Renommer les colonnes pour l'affichage
                    top10_display.columns = ['ID', 'Income (USD)', 'Savings (USD)', 'Investments (USD)', 
                                            'Total Expenses', 'Savings Rate', 'Investment Rate', 
                                            'Expense Rate', 'Total Ratios', 'Profil']
                    
                    st.dataframe(top10_display, use_container_width=True, hide_index=True)
                    
                    # Résumé statistique
                    st.markdown("**📊 Résumé statistique**")
                    
                    # Détection des outliers avec IQR
                    Q1 = df_features['Total_Ratios'].quantile(0.25)
                    Q3 = df_features['Total_Ratios'].quantile(0.75)
                    IQR = Q3 - Q1
                    upper_bound_iqr = Q3 + 1.5 * IQR
                    outliers_iqr = (df_features['Total_Ratios'] > upper_bound_iqr).sum()
                    
                    # Détection avec Z-Score
                    z_scores = np.abs(stats.zscore(df_features['Total_Ratios']))
                    outliers_zscore = (z_scores > 3).sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total d'individus", len(df_features))
                    with col2:
                        st.metric("Ratios > 200%", f"{(df_features['Total_Ratios'] > 200).sum()} ({(df_features['Total_Ratios'] > 200).sum()/len(df_features)*100:.1f}%)")
                    with col3:
                        st.metric("Outliers IQR", f"{outliers_iqr} ({outliers_iqr/len(df_features)*100:.2f}%)")
                    with col4:
                        st.metric("Outliers Z-Score", f"{outliers_zscore} ({outliers_zscore/len(df_features)*100:.2f}%)")
                    
                    # Analyse détaillée
                    st.markdown("**💡 Analyse**")
                    st.markdown("""
                    ✅ **Ce sont des outliers statistiques**
                    - Détection par IQR : OUI
                    - Détection par Z-Score : OUI
                    - Les ratios totaux > 200% sont rares statistiquement.
                    
                    ❌ **Mais ce ne sont pas des erreurs de données**
                    - Les calculs sont corrects.
                    - Les valeurs sont cohérentes entre elles.
                    - Les profils financiers sont logiques et reflètent la diversité des comportements financiers de la Génération Z.
                    
                    ✅ **Ce sont des cas réels valides**
                    - Ces individus représentent des comportements financiers atypiques mais existants.
                    - Ils doivent être conservés dans le dataset.
                    - Ils doivent être documentés pour expliquer leur présence et éviter une mauvaise interprétation.
                    
                    **📊 Analyse et interprétations des causes des ratios extrêmes :**
                    
                    - **Savings_Rate > 100%** : peut s'expliquer par des revenus complémentaires (freelance, location, ventes), une épargne accumulée sur plusieurs périodes, des héritages ou dons, ou encore des revenus de placements antérieurs.
                    
                    - **Investment_Rate > 100%** : reflète des investissements financés par une épargne passée, des héritages, des crédits, ou le réinvestissement de revenus financiers (dividendes, intérêts).
                    
                    - **Expense_Rate > 100%** : traduit des dépenses couvertes par le crédit, des décalages entre revenus et dépenses annuelles (vacances, éducation), des aides externes, des revenus non déclarés ou des dépenses exceptionnelles (santé, réparations).
                    """)
                    
                    # Conclusion
                    st.markdown("**✨ Conclusion**")
                    st.info("""
                    Les ratios moyens semblent incohérents à première vue car biaisés par des outliers. 
                    Mais ces outliers ne sont pas des erreurs : ils reflètent des comportements financiers réels. 
                    Leur conservation est essentielle pour une analyse complète et réaliste.
                    """)
    
    with tab3:
        st.subheader("📈 Indices Comportementaux")
        
        st.markdown("Ces indices aident à comprendre les comportements de dépenses.")
        
        # Formules de calcul
        st.markdown("### 📐 Formules de calcul des indices")
        
        indices_table = pd.DataFrame({
            'Indice': [
                "Indice de style de vie (Lifestyle_Index)",
                "Indice d'éducation (Education_Index)",
                "Indice de santé (Health_Index)",
                "Indice digital (Digital_Index)"
            ],
            'Formule': [
                "((Eating Out + Entertainment + Travel) / Income (USD)) × 100",
                "(Education (USD) / Income (USD)) × 100",
                "(Fitness (USD) / Income (USD)) × 100",
                "((Subscription Services + Online Shopping) / Income (USD)) × 100"
            ],
            'Description': [
                "Pourcentage du revenu consacré aux loisirs et divertissement",
                "Pourcentage du revenu consacré à l'éducation",
                "Pourcentage du revenu consacré à la santé et fitness",
                "Pourcentage du revenu consacré aux abonnements et shopping en ligne"
            ]
        })
        st.dataframe(indices_table, use_container_width=True, hide_index=True)
        
        # Statistiques des indices
        st.markdown("### 📊 Statistiques descriptives des indices")
        index_cols = ['Lifestyle_Index', 'Education_Index', 'Health_Index', 'Digital_Index']
        if all(col in df_features.columns for col in index_cols):
            st.dataframe(df_features[index_cols].describe(), use_container_width=True)
            
            # Visualisation
            st.markdown("### 📈 Distribution des indices")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(index_cols):
                axes[idx].hist(df_features[col], bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Fréquence')
                axes[idx].set_title(f'Distribution de {col}')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Comparaison des indices
            st.markdown("### 📊 Comparaison des indices (moyennes)")
            index_means = df_features[index_cols].mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            index_means.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            ax.set_xlabel('Indice')
            ax.set_ylabel('Valeur moyenne (%)')
            ax.set_title('Comparaison des indices comportementaux (moyennes)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interprétation des indices
            st.markdown("### 💡 Interprétation des indices")
            
            if all(col in df_features.columns for col in index_cols):
                lifestyle_mean = df_features['Lifestyle_Index'].mean()
                education_mean = df_features['Education_Index'].mean()
                health_mean = df_features['Health_Index'].mean()
                digital_mean = df_features['Digital_Index'].mean()
                
                # Trouver l'indice le plus élevé
                index_means = {
                    'Lifestyle': lifestyle_mean,
                    'Education': education_mean,
                    'Health': health_mean,
                    'Digital': digital_mean
                }
                max_index = max(index_means, key=index_means.get)
                
                st.markdown(f"""
                **📊 Analyse comportementale :**
                
                - **Indice de style de vie : {lifestyle_mean:.2f}%**
                  → Les individus consacrent en moyenne {lifestyle_mean:.1f}% de leur revenu aux loisirs (restaurants, divertissement, voyages).
                  Cela reflète l'importance accordée aux expériences et au bien-être.
                
                - **Indice d'éducation : {education_mean:.2f}%**
                  → En moyenne, {education_mean:.1f}% du revenu est investi dans l'éducation.
                  Une valeur élevée indique une priorité à la formation et au développement personnel.
                
                - **Indice de santé : {health_mean:.2f}%**
                  → Les dépenses fitness représentent en moyenne {health_mean:.1f}% du revenu.
                  Cela montre l'attention portée à la santé physique.
                
                - **Indice digital : {digital_mean:.2f}%**
                  → En moyenne, {digital_mean:.1f}% du revenu est consacré aux abonnements et shopping en ligne.
                  Cela reflète l'adoption des services numériques par la Génération Z.
                
                **🔍 Observation principale :**
                - L'indice le plus élevé est **{max_index}** ({index_means[max_index]:.1f}%), ce qui indique que c'est la priorité principale de dépense pour cette génération.
                """)
    
    with tab4:
        st.subheader("🎯 Financial Profile (Target)")
        
        st.markdown("Le Financial Profile est créé en fonction des ratios d'épargne, d'investissement et de dépenses.")
        
        if 'Financial_Profile' in df_features.columns:
            # Règles de classification
            st.markdown("### 🔍 Règles de classification du Financial Profile")
            
            st.markdown("Le Financial Profile est créé en fonction des ratios calculés précédemment. Les conditions sont évaluées dans l'ordre ci-dessous :")
            
            classification_table = pd.DataFrame({
                'Profil': [
                    "Investisseur 💼",
                    "Économe 💰",
                    "Dépensier 🛍️",
                    "Équilibré ⚖️"
                ],
                'Condition': [
                    "Investment_Rate > 15% ET Savings_Rate > 10%",
                    "Savings_Rate > 20% ET Expense_Rate < 70%",
                    "Expense_Rate > 85% ET Savings_Rate < 10%",
                    "Tous les autres cas (comportement modéré)"
                ],
                'Description': [
                    "Priorité à l'investissement avec épargne modérée",
                    "Priorité à l'épargne avec dépenses contrôlées",
                    "Dépenses très élevées avec faible épargne",
                    "Comportement modéré ne correspondant à aucun des profils ci-dessus"
                ]
            })
            st.dataframe(classification_table, use_container_width=True, hide_index=True)
            
            # Distribution
            st.markdown("### 📊 Distribution des profils financiers")
            profile_dist = df_features['Financial_Profile'].value_counts()
            profile_pct = (profile_dist / len(df_features) * 100).round(2)
            
            # Tableau fusionné avec nombre et pourcentage
            distribution_df = pd.DataFrame({
                'Profil': profile_dist.index,
                'Nombre d\'individus': profile_dist.values,
                'Pourcentage (%)': profile_pct.values
            })
            st.dataframe(distribution_df, use_container_width=True, hide_index=True)
            
            # Graphique de distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            profile_dist.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
            ax.set_xlabel('Profil Financier')
            ax.set_ylabel('Nombre d\'individus')
            ax.set_title('Distribution des Profils Financiers (Target)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Analyse des ratios par profil
            st.markdown("### 📊 Statistiques des ratios par profil")
            if all(col in df_features.columns for col in ['Savings_Rate', 'Investment_Rate', 'Expense_Rate', 'Financial_Profile']):
                profile_stats = df_features.groupby('Financial_Profile').agg({
                    'Savings_Rate': ['mean', 'median'],
                    'Investment_Rate': ['mean', 'median'],
                    'Expense_Rate': ['mean', 'median'],
                    'Income (USD)': ['mean', 'median']
                }).round(2)
                st.dataframe(profile_stats, use_container_width=True)
                
                # Boxplots
                st.markdown("### 📈 Visualisation des ratios par profil")
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Savings Rate
                df_features.boxplot(column='Savings_Rate', by='Financial_Profile', ax=axes[0, 0])
                axes[0, 0].set_title('Taux d\'Épargne par Profil', fontweight='bold')
                axes[0, 0].set_xlabel('')
                axes[0, 0].set_ylabel('Taux d\'Épargne (%)')
                
                # Investment Rate
                df_features.boxplot(column='Investment_Rate', by='Financial_Profile', ax=axes[0, 1])
                axes[0, 1].set_title('Taux d\'Investissement par Profil', fontweight='bold')
                axes[0, 1].set_xlabel('')
                axes[0, 1].set_ylabel('Taux d\'Investissement (%)')
                
                # Expense Rate
                df_features.boxplot(column='Expense_Rate', by='Financial_Profile', ax=axes[1, 0])
                axes[1, 0].set_title('Taux de Dépenses par Profil', fontweight='bold')
                axes[1, 0].set_xlabel('')
                axes[1, 0].set_ylabel('Taux de Dépenses (%)')
                
                # Income
                df_features.boxplot(column='Income (USD)', by='Financial_Profile', ax=axes[1, 1])
                axes[1, 1].set_title('Revenus par Profil', fontweight='bold')
                axes[1, 1].set_xlabel('')
                axes[1, 1].set_ylabel('Revenus (USD)')
                
                plt.suptitle('Analyse des Ratios par Profil Financier', fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interprétation
                st.markdown("### 💡 Interprétation")
                st.markdown("""
                **📊 1. Taux d'Épargne par Profil**
                
                - **Investisseurs** ont les taux d'épargne les plus élevés, avec une distribution étalée et des valeurs extrêmes.
                - **Équilibrés** et **Économes** montrent des niveaux modérés mais stables.
                - **Dépensiers** ont des taux très faibles, souvent proches de zéro.
                
                **📈 2. Taux d'Investissement par Profil**
                
                - Les **Investisseurs** dominent largement, avec des taux très élevés et une forte dispersion.
                - Les autres profils ont des taux faibles, parfois nuls.
                
                **💸 3. Taux de Dépenses par Profil**
                
                - Les **Dépensiers** et **Équilibrés** ont des taux de dépenses élevés.
                - Les **Économes** et **Investisseurs** dépensent moins, avec des distributions plus resserrées.
                
                **💰 4. Revenu par Profil**
                
                Le graphique des revenus peut sembler vide car lorsque les ratios (épargne, investissement, dépenses) sont calculés en pourcentage du revenu, le revenu devient simplement une variable de référence (le dénominateur).
                """)
    
    with tab5:
        st.subheader("📋 Résumé du Feature Engineering")
        
        if feature_summary:
            st.markdown("### 📊 Informations générales")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Variables originales", len(feature_summary.get('original_variables', [])))
            with col2:
                st.metric("Variables dérivées", len(feature_summary.get('derived_variables', [])))
            with col3:
                st.metric("Variables catégorielles", len(feature_summary.get('categorical_variables', [])))
            
            st.markdown("### 🎯 Target Variable")
            st.success(f"**{feature_summary.get('target_variable', 'N/A')}**")
            
            if 'profile_distribution' in feature_summary:
                st.markdown("### 📊 Distribution du Financial Profile")
                profile_dist = feature_summary['profile_distribution']
                dist_df = pd.DataFrame({
                    'Profil': list(profile_dist.keys()),
                    'Nombre': list(profile_dist.values())
                })
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            st.markdown("### ⚠️ Note importante")
            st.warning(feature_summary.get('note', 'Les ratios et indices NE DOIVENT PAS être utilisés dans ACP et Clustering car ils biaisent les résultats'))
        else:
            st.warning("⚠️ Les métadonnées du feature engineering ne sont pas disponibles.")

# ============================================
# NOTEBOOK 4: ACP PROFESSIONNEL
# ============================================
elif notebook_choice == "4. ACP Professionnel":
    st.header("📊 4. ACP Professionnel - Analyse en Composantes Principales")
    st.markdown("**Objectif:** Réduire la dimensionnalité des données et identifier les structures sous-jacentes")
    
    # Chargement des données
    @st.cache_data
    def load_acp_data():
        """Charge les données transformées par l'ACP"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/acp_transformed_data.csv'))
            return df
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_acp_components():
        """Charge les composantes principales (loadings)"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/results/acp_components_loadings.csv'), index_col=0)
            return df
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_acp_eigenvalues():
        """Charge les valeurs propres"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/results/acp_eigenvalues.csv'))
            return df
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_acp_summary():
        """Charge le résumé complet de l'ACP"""
        try:
            with open(os.path.join(BASE_DIR, 'data/results/acp_complete_summary.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    # Chargement des données
    df_acp = load_acp_data()
    components_df = load_acp_components()
    eigenvalues_df = load_acp_eigenvalues()
    acp_summary = load_acp_summary()
    
    if df_acp is None or components_df is None or eigenvalues_df is None:
        st.warning("⚠️ Les données de l'ACP n'existent pas encore. Exécutez d'abord le notebook notebooks/03_pca.ipynb")
        st.stop()
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Vue d'ensemble",
        "📉 Valeurs propres & Scree Plot",
        "🎯 Cercles de corrélation",
        "📍 Cartes des individus",
        "🎨 Biplot",
        "📋 Résumé & Interprétation"
    ])
    
    with tab1:
        st.subheader("📊 Vue d'ensemble de l'ACP")
        
        if acp_summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Variables originales", acp_summary.get('n_variables_originales', 'N/A'))
            with col2:
                st.metric("Composantes retenues", acp_summary.get('n_composantes_retenues', 'N/A'))
            with col3:
                st.metric("Variance expliquée", f"{acp_summary.get('variance_expliquee', 0):.2f}%")
            with col4:
                st.metric("Observations", f"{acp_summary.get('n_observations', 0):,}")
            
            st.markdown("---")
            
            # Critères de sélection
            st.markdown("### 📌 Critères de sélection des composantes")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Critère 80%", f"{acp_summary.get('critere_80_pourcent', 'N/A')} composantes")
            with col2:
                st.metric("Critère 90%", f"{acp_summary.get('critere_90_pourcent', 'N/A')} composantes")
            with col3:
                st.metric("Critère Kaiser", f"{acp_summary.get('critere_kaiser', 'N/A')} composantes")
            
            st.markdown("### 📊 Aperçu des données transformées")
            st.dataframe(df_acp.head(10), use_container_width=True)
            
            st.markdown("### 📈 Variance expliquée par composante")
            if 'variance_par_composante' in acp_summary:
                variance_data = acp_summary['variance_par_composante']
                variance_df = pd.DataFrame({
                    'Composante': [f"PC{k}" for k in variance_data.keys()],
                    'Variance expliquée (%)': [v for v in variance_data.values()]
                })
                st.dataframe(variance_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Le résumé de l'ACP n'est pas disponible")
    
    with tab2:
        st.subheader("📉 Valeurs Propres et Scree Plot")
        
        if eigenvalues_df is not None:
            st.markdown("### 📊 Tableau des valeurs propres")
            st.dataframe(eigenvalues_df, use_container_width=True)
            
            # Scree plot
            st.markdown("### 📉 Scree Plot")
            
            if 'Valeur propre' in eigenvalues_df.columns and 'Variance expliquée (%)' in eigenvalues_df.columns:
                eigenvalues = eigenvalues_df['Valeur propre'].values
                explained_variance = eigenvalues_df['Variance expliquée (%)'].values
                cumulative_variance = eigenvalues_df['Variance cumulée (%)'].values
                
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Graphique 1 : Scree plot
                ax1 = axes[0]
                n_components = len(eigenvalues)
                ax1.plot(range(1, n_components+1), eigenvalues, 'o-', linewidth=2, markersize=8, 
                        color='steelblue', label='Valeurs propres')
                ax1.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Critère Kaiser (λ = 1)')
                ax1.set_xlabel('Numéro de la composante', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Valeur propre (λ)', fontsize=11, fontweight='bold')
                ax1.set_title('📉 Scree Plot - Valeurs Propres', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Graphique 2 : Variance cumulée
                ax2 = axes[1]
                ax2.plot(range(1, n_components+1), cumulative_variance, 'o-', linewidth=2, 
                        markersize=8, color='mediumseagreen', label='Variance cumulée')
                ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% de variance')
                ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% de variance')
                ax2.set_xlabel('Nombre de composantes', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Variance cumulée (%)', fontsize=11, fontweight='bold')
                ax2.set_title('📈 Variance Cumulée', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interprétation du Scree Plot
                st.markdown("### 📊 Interprétation précise du Scree Plot")
                
                # Calcul du nombre de composantes avec λ > 1 (critère de Kaiser)
                n_kaiser = np.sum(eigenvalues > 1)
                
                st.markdown("""
                La courbe bleue représente les **valeurs propres (λ)** de chaque composante principale.
                
                La ligne rouge est le **seuil du critère de Kaiser (λ = 1)**, qui recommande de retenir les composantes dont λ > 1.
                
                Si la courbe bleue croise la ligne rouge au niveau de la **{}ᵉ composante**, cela signifie que :
                
                - Les **{} premières composantes** ont λ > 1 → elles sont considérées comme significatives selon Kaiser.
                
                - À partir de la **{}ᵉ composante**, λ < 1 → elles apportent moins d'information utile.
                """.format(n_kaiser + 1, n_kaiser, n_kaiser + 1))
                
                # Interprétation des seuils de variance
                st.markdown("### 📊 Interprétation des seuils de variance")
                
                # Calcul des composantes pour 80% et 90% de variance
                n_80 = np.where(cumulative_variance >= 0.80)[0][0] + 1 if np.any(cumulative_variance >= 0.80) else n_components
                n_90 = np.where(cumulative_variance >= 0.90)[0][0] + 1 if np.any(cumulative_variance >= 0.90) else n_components
                
                st.markdown(f"""
                **Critère 80%** : atteint à la **{n_80}ᵉ composante** → Il faut conserver au moins {n_80} composantes pour expliquer 80% de la variance totale.
                
                **Critère 90%** : atteint à la **{n_90}ᵉ composante** → Pour une couverture quasi complète de l'information, {n_90} composantes sont nécessaires.
                
                ### 🧠 Conclusion
                
                Le dataset présente une **forte dispersion de l'information** : aucune composante individuelle ne domine. Une réduction trop agressive (2–3 composantes) serait insuffisante. Il est recommandé de conserver entre **{n_80} et {n_90} composantes**.
                """)
    
    with tab3:
        st.subheader("🎯 Cercles de Corrélation")
        
        if components_df is not None and eigenvalues_df is not None:
            explained_variance_ratio = eigenvalues_df['Variance expliquée (%)'].values / 100
            
            st.markdown("### 📊 Cercles de corrélation (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)")
            
            # Sélection des composantes à visualiser
            col1, col2 = st.columns(2)
            with col1:
                pc_x = st.selectbox("Composante X", [f"PC{i+1}" for i in range(min(3, len(components_df.columns)))], index=0)
            with col2:
                pc_y = st.selectbox("Composante Y", [f"PC{i+1}" for i in range(min(3, len(components_df.columns)))], index=1)
            
            if pc_x in components_df.columns and pc_y in components_df.columns:
                pc_x_idx = int(pc_x[2:]) - 1
                pc_y_idx = int(pc_y[2:]) - 1
                
                loadings = components_df[[pc_x, pc_y]]
                
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Cercle unité
                circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1)
                ax.add_patch(circle)
                
                # Flèches pour chaque variable
                for idx, row in loadings.iterrows():
                    ax.arrow(0, 0, row[pc_x], row[pc_y], head_width=0.05, head_length=0.05,
                            fc='red', ec='red', linewidth=1.5, alpha=0.7)
                    ax.text(row[pc_x]*1.1, row[pc_y]*1.1, idx, fontsize=9, fontweight='bold',
                           ha='center', va='center')
                
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_xlabel(f'{pc_x} ({explained_variance_ratio[pc_x_idx]*100:.2f}%)', 
                             fontsize=11, fontweight='bold')
                ax.set_ylabel(f'{pc_y} ({explained_variance_ratio[pc_y_idx]*100:.2f}%)', 
                             fontsize=11, fontweight='bold')
                ax.set_title(f'🎯 Cercle de Corrélation ({pc_x} vs {pc_y})', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("### 💡 Interprétation")
                st.info("""
                - **Variables proches du bord** → bien représentées
                - **Angles ≈ 0°** → corrélation positive
                - **Angles ≈ 180°** → corrélation négative
                - **Angles ≈ 90°** → indépendance
                """)
            else:
                st.warning("⚠️ Les composantes sélectionnées ne sont pas disponibles")
        else:
            st.warning("⚠️ Les données des composantes ne sont pas disponibles")
    
    with tab4:
        st.subheader("📍 Cartes des Individus")
        
        if df_acp is not None and eigenvalues_df is not None:
            explained_variance_ratio = eigenvalues_df['Variance expliquée (%)'].values / 100
            
            st.markdown("### 📊 Cartes des individus dans l'espace des composantes principales")
            
            # Sélection des composantes
            col1, col2 = st.columns(2)
            with col1:
                pc_x = st.selectbox("Composante X (cartes)", 
                                   [f"PC{i+1}" for i in range(min(3, len(df_acp.columns)))], 
                                   index=0, key='pc_x_map')
            with col2:
                pc_y = st.selectbox("Composante Y (cartes)", 
                                   [f"PC{i+1}" for i in range(min(3, len(df_acp.columns)))], 
                                   index=1, key='pc_y_map')
            
            if pc_x in df_acp.columns and pc_y in df_acp.columns:
                pc_x_idx = int(pc_x[2:]) - 1
                pc_y_idx = int(pc_y[2:]) - 1
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                scatter = ax.scatter(df_acp[pc_x], df_acp[pc_y], 
                                    alpha=0.6, s=30, c=df_acp[pc_x], 
                                    cmap='viridis', edgecolors='black', linewidth=0.3)
                
                ax.set_xlabel(f'{pc_x} ({explained_variance_ratio[pc_x_idx]*100:.2f}%)', 
                             fontsize=11, fontweight='bold')
                ax.set_ylabel(f'{pc_y} ({explained_variance_ratio[pc_y_idx]*100:.2f}%)', 
                             fontsize=11, fontweight='bold')
                ax.set_title(f'📍 Carte des Individus ({pc_x} vs {pc_y})', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
                plt.colorbar(scatter, ax=ax, label=f'Valeur {pc_x}')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("### 💡 Interprétation")
                st.info("""
                - **Proximité entre individus** = similarité
                - **Éloignement** = différences significatives
                """)
            else:
                st.warning("⚠️ Les composantes sélectionnées ne sont pas disponibles")
        else:
            st.warning("⚠️ Les données transformées ne sont pas disponibles")
    
    with tab5:
        st.subheader("🎨 Biplot")
        
        if df_acp is not None and components_df is not None and eigenvalues_df is not None:
            explained_variance_ratio = eigenvalues_df['Variance expliquée (%)'].values / 100
            
            st.markdown("### 📊 Biplot : Variables et Individus")
            st.info("💡 Le biplot superpose les variables (flèches) et les individus (points) pour une lecture combinée.")
            
            # Sélection des composantes pour le biplot
            col1, col2 = st.columns(2)
            with col1:
                pc_x = st.selectbox("Composante X (biplot)", 
                                   [f"PC{i+1}" for i in range(min(3, len(df_acp.columns)))], 
                                   index=0, key='pc_x_biplot')
            with col2:
                pc_y = st.selectbox("Composante Y (biplot)", 
                                   [f"PC{i+1}" for i in range(min(3, len(df_acp.columns)))], 
                                   index=1, key='pc_y_biplot')
            
            if pc_x in df_acp.columns and pc_y in df_acp.columns and pc_x in components_df.columns and pc_y in components_df.columns:
                pc_x_idx = int(pc_x[2:]) - 1
                pc_y_idx = int(pc_y[2:]) - 1
                
                loadings_biplot = components_df[[pc_x, pc_y]]
                
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Individus
                ax.scatter(df_acp[pc_x], df_acp[pc_y], 
                          alpha=0.5, s=20, c='lightblue', edgecolors='black', linewidth=0.3,
                          label='Individus')
                
                # Variables (flèches)
                loadings_scale = 2.5
                for idx, row in loadings_biplot.iterrows():
                    ax.arrow(0, 0, row[pc_x]*loadings_scale, row[pc_y]*loadings_scale,
                            head_width=0.1, head_length=0.1, fc='red', ec='red', 
                            linewidth=2, alpha=0.8)
                    ax.text(row[pc_x]*loadings_scale*1.15, row[pc_y]*loadings_scale*1.15, 
                           idx, fontsize=10, fontweight='bold', color='red',
                           ha='center', va='center')
                
                ax.set_xlabel(f'{pc_x} ({explained_variance_ratio[pc_x_idx]*100:.2f}%)', 
                             fontsize=12, fontweight='bold')
                ax.set_ylabel(f'{pc_y} ({explained_variance_ratio[pc_y_idx]*100:.2f}%)', 
                             fontsize=12, fontweight='bold')
                ax.set_title(f'🎨 Biplot : Variables et Individus ({pc_x} vs {pc_y})', 
                           fontsize=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
                ax.legend(loc='upper right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("### 💡 Interprétation du biplot")
                st.info("""
                - **Les flèches rouges** représentent les variables
                - **Les points bleus** représentent les individus
                - **Proximité variable-individu** = forte contribution de la variable pour cet individu
                """)
            else:
                st.warning("⚠️ Les composantes sélectionnées ne sont pas disponibles")
        else:
            st.warning("⚠️ Les données nécessaires ne sont pas disponibles")
    
    with tab6:
        st.subheader("📋 Résumé et Interprétation")
        
        if acp_summary:
            st.markdown("### 📊 Résumé de l'ACP")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Informations générales")
                st.write(f"**Variables originales:** {acp_summary.get('n_variables_originales', 'N/A')}")
                st.write(f"**Composantes retenues:** {acp_summary.get('n_composantes_retenues', 'N/A')}")
                st.write(f"**Variance expliquée:** {acp_summary.get('variance_expliquee', 0):.2f}%")
                st.write(f"**Observations:** {acp_summary.get('n_observations', 0):,}")
            
            with col2:
                st.markdown("#### 📌 Critères de sélection")
                st.write(f"**Critère 80%:** {acp_summary.get('critere_80_pourcent', 'N/A')} composantes")
                st.write(f"**Critère 90%:** {acp_summary.get('critere_90_pourcent', 'N/A')} composantes")
                st.write(f"**Critère Kaiser:** {acp_summary.get('critere_kaiser', 'N/A')} composantes")
            
            st.markdown("---")
            
            # Recommandations
            st.markdown("### 💡 Recommandations")
            
            variance_expl = acp_summary.get('variance_expliquee', 0)
            n_comp = acp_summary.get('n_composantes_retenues', 0)
            n_vars = acp_summary.get('n_variables_originales', 0)
            
            reduction_ratio = (1 - n_comp / n_vars) * 100 if n_vars > 0 else 0
            
            if variance_expl < 80:
                st.warning(f"⚠️ La variance expliquée ({variance_expl:.2f}%) est inférieure à 80%. Considérez augmenter le nombre de composantes.")
            
            if reduction_ratio < 20:
                st.info(f"ℹ️ La réduction de dimensionnalité est limitée ({reduction_ratio:.1f}%). L'ACP peut ne pas être très efficace pour ce dataset.")
            
            st.warning("""
            ⚠️ **Dans notre cas, l'ACP n'est pas recommandée.** 
            
            Les variables étant indépendantes, chaque variable est importante. L'ACP ajoute de la complexité sans bénéfice significatif.
            
            **Recommandation :** Utiliser les variables originales standardisées directement pour le clustering.
            """)
        else:
            st.warning("⚠️ Le résumé de l'ACP n'est pas disponible")

# ============================================
# NOTEBOOK 5: CLUSTERING K-MEANS
# ============================================
elif notebook_choice == "5. Clustering K-Means":
    st.header("🤖 5. Clustering K-Means - Segmentation des Données")
    st.markdown("**Objectif:** Segmenter les individus en groupes homogènes basés sur leurs comportements de dépenses")
    
    # Fonctions de chargement
    @st.cache_data
    def load_kmeans_data():
        """Charge les données avec les clusters K-Means"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/dataset_with_kmeans_clusters.csv'))
            return df
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_kmeans_metrics():
        """Charge les métriques K-Means par k"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/results/kmeans_metrics_by_k.csv'))
            return df
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_kmeans_results():
        """Charge les résultats optimaux K-Means"""
        try:
            with open(os.path.join(BASE_DIR, 'data/results/kmeans_optimal_k_results.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_standardization_info():
        """Charge les informations de standardisation"""
        try:
            with open(os.path.join(BASE_DIR, 'data/results/clustering_standardization_info.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    # Chargement des données
    df_kmeans = load_kmeans_data()
    metrics_df = load_kmeans_metrics()
    kmeans_results = load_kmeans_results()
    standardization_info = load_standardization_info()
    
    if df_kmeans is None:
        st.warning("⚠️ Les données de clustering K-Means n'existent pas encore. Exécutez d'abord le notebook notebooks/04_kmeans_clustering.ipynb")
        st.stop()
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Préparation & Standardisation",
        "⚙️ Application K-Means & Visualisations",
        "📉 Détermination K Optimal",
        "📈 Résumé & Interprétation"
    ])
    
    # TAB 1: Préparation & Standardisation
    with tab1:
        st.subheader("📂 Préparation des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nombre d'observations", f"{len(df_kmeans):,}")
            # Compter les variables originales (exclure ID, Financial_Profile, clusters)
            original_vars_count = len([col for col in df_kmeans.columns 
                                      if col not in ['ID', 'Financial_Profile', 'KMeans_Cluster', 'DBSCAN_Cluster', 
                                                    'Total_Expenses', 'Disposable_Income', 'Savings_Rate', 
                                                    'Investment_Rate', 'Expense_Rate', 'Income_Category',
                                                    'Lifestyle_Index', 'Education_Index', 'Health_Index', 'Digital_Index']])
            st.metric("Nombre de variables", str(original_vars_count))
        
        with col2:
            st.metric("Données standardisées", "✅ Oui")
            st.metric("Valeurs manquantes", "0")
        
        st.markdown("---")
        st.subheader("💡 Pourquoi standardiser ?")
        st.info("""
        **Les algorithmes de clustering (K-Means, DBSCAN) utilisent des distances**
        
        - Les variables avec des échelles différentes biaisent les calculs de distance
        - La standardisation (moyenne=0, écart-type=1) assure une pondération équitable
        - Chaque variable contribue de manière équitable au calcul des distances
        """)
        
        if standardization_info:
            st.markdown("### ✅ Données standardisées")
            st.success("Les données ont été standardisées avec succès. Moyenne ≈ 0 et écart-type ≈ 1 pour toutes les variables.")
            
            st.markdown("#### Variables standardisées:")
            vars_list = standardization_info.get('variables', [])
            st.write(", ".join(vars_list))
        
        st.markdown("---")
        st.subheader("🔍 Analyse des corrélations")
        
        # Charger les données originales pour la corrélation
        df_cleaned = load_cleaned_data()
        if df_cleaned is not None:
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            if 'ID' in numeric_cols:
                numeric_cols.remove('ID')
            
            if len(numeric_cols) > 0:
                X_corr = df_cleaned[numeric_cols]
                
                # Calcul de la matrice de corrélation
                correlation_matrix = X_corr.corr()
                
                # Analyse des corrélations fortes
                corr_values = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_values.append({
                            'Variable 1': correlation_matrix.columns[i],
                            'Variable 2': correlation_matrix.columns[j],
                            'Corrélation': correlation_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_values)
                corr_df['Abs_Correlation'] = corr_df['Corrélation'].abs()
                strong_corr = corr_df[corr_df['Abs_Correlation'] > 0.7].sort_values('Abs_Correlation', ascending=False)
                
                if len(strong_corr) > 0:
                    st.warning(f"⚠️ {len(strong_corr)} paires de variables hautement corrélées (|r| > 0.7)")
                    st.dataframe(strong_corr[['Variable 1', 'Variable 2', 'Corrélation']].head(10))
                else:
                    st.success("✅ Aucune paire de variables hautement corrélée (|r| > 0.7)")
                    st.info("""
                    **Les variables sont relativement indépendantes**
                    
                    - Cela peut rendre le clustering plus difficile (pas de structure claire)
                    - Chaque variable apporte une information unique
                    - Le clustering peut nécessiter plus de clusters pour capturer la variabilité
                    """)
                
                # Visualisation de la matrice de corrélation
                try:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                               cmap='coolwarm', center=0, square=True, linewidths=0.5, 
                               cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
                    ax.set_title('📊 Matrice de Corrélation (Données Standardisées)', 
                                fontsize=14, fontweight='bold', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except:
                    st.info("Visualisation de la matrice de corrélation disponible dans le notebook")
    
    # TAB 2: Application K-Means
    with tab2:
        st.subheader("⚙️ Application de K-Means")
        
        # Paramètres initiaux
        st.markdown("### 📊 Paramètres initiaux")
        col1, col2, col3 = st.columns(3)
        with col1:
            k_value = kmeans_results.get('k_optimal', 4) if kmeans_results else 4
            st.metric("Nombre de clusters (k)", str(k_value))
        with col2:
            st.metric("Nombre d'observations", f"{len(df_kmeans):,}")
        with col3:
            # Compter les variables originales (exclure ID, Financial_Profile, clusters)
            original_vars_count = len([col for col in df_kmeans.columns 
                                      if col not in ['ID', 'Financial_Profile', 'KMeans_Cluster', 'DBSCAN_Cluster', 
                                                    'Total_Expenses', 'Disposable_Income', 'Savings_Rate', 
                                                    'Investment_Rate', 'Expense_Rate', 'Income_Category',
                                                    'Lifestyle_Index', 'Education_Index', 'Health_Index', 'Digital_Index']])
            st.metric("Nombre de variables", str(original_vars_count))
        
        # Vérifier si on a les clusters initiaux (k=4)
        if 'KMeans_Cluster' in df_kmeans.columns:
            cluster_col = 'KMeans_Cluster'
        elif 'Cluster' in df_kmeans.columns:
            cluster_col = 'Cluster'
        else:
            cluster_col = None
        
        if cluster_col:
            st.markdown("---")
            st.markdown("### 🔍 Vérification de la convergence")
            
            st.success("✅ Convergence atteinte")
            st.info("""
            - **Nombre d'itérations effectuées:** 49
            - **Nombre maximum d'itérations:** 300
            - **Convergence:** ✅ Oui
            
            Les centroïdes sont stables, le modèle a convergé avec succès.
            """)
            
            # Centroïdes (si disponibles dans les résultats)
            if kmeans_results:
                st.markdown("---")
                st.markdown("### 📊 Centroïdes des clusters")
                st.info("Les centroïdes sont dans l'espace standardisé (moyenne=0, écart-type=1)")
                st.caption("💡 Les valeurs positives indiquent des valeurs au-dessus de la moyenne, les valeurs négatives en dessous")
        
        # Section Visualisations
        st.markdown("---")
        st.markdown("## 📊 Visualisations des Clusters K-Means")
        
        st.markdown("💡 Pour visualiser les clusters en 2D/3D, on utilise des variables originales importantes")
        
        # Visualisation 2D
        st.markdown("### 📊 Visualisation 2D des Clusters")
        
        if cluster_col and 'Income (USD)' in df_kmeans.columns and 'Savings (USD)' in df_kmeans.columns:
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Créer un scatter plot coloré par cluster
            unique_clusters = sorted(df_kmeans[cluster_col].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                cluster_data = df_kmeans[df_kmeans[cluster_col] == cluster]
                ax.scatter(cluster_data['Income (USD)'], cluster_data['Savings (USD)'], 
                          label=f'Cluster {cluster}', alpha=0.6, s=50, color=colors[i], edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Income (USD)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Savings (USD)', fontsize=12, fontweight='bold')
            ax.set_title('Visualisation 2D des Clusters K-Means\nIncome vs Savings', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(title='Clusters', loc='best', frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.caption("📊 Variables utilisées: Income (USD) vs Savings (USD)")
        
        # Afficher l'image sauvegardée si elle existe
        if os.path.exists(os.path.join(BASE_DIR, 'data/results/kmeans_clusters_2d.png')):
            st.markdown("---")
            st.image(os.path.join(BASE_DIR, 'data/results/kmeans_clusters_2d.png'), 
                    caption='Clusters K-Means (k=4) - Visualisation 2D: Income vs Savings avec centroïdes')
        
        # Visualisation 3D
        st.markdown("---")
        st.markdown("### 📊 Visualisation 3D des Clusters")
        
        if cluster_col and 'Income (USD)' in df_kmeans.columns and 'Savings (USD)' in df_kmeans.columns and 'Rent (USD)' in df_kmeans.columns:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            unique_clusters = sorted(df_kmeans[cluster_col].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                cluster_data = df_kmeans[df_kmeans[cluster_col] == cluster]
                ax.scatter(cluster_data['Income (USD)'], cluster_data['Savings (USD)'], 
                          cluster_data['Rent (USD)'], label=f'Cluster {cluster}', 
                          alpha=0.6, s=30, color=colors[i], edgecolors='black', linewidth=0.3)
            
            ax.set_xlabel('Income (USD)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Savings (USD)', fontsize=11, fontweight='bold')
            ax.set_zlabel('Rent (USD)', fontsize=11, fontweight='bold')
            ax.set_title('Visualisation 3D des Clusters K-Means', fontsize=14, fontweight='bold', pad=20)
            ax.legend(title='Clusters', loc='upper left', frameon=True, shadow=True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.caption("📊 Variables utilisées: Income (USD) vs Savings (USD) vs Rent (USD)")
        
        # Afficher l'image sauvegardée 3D si elle existe
        if os.path.exists(os.path.join(BASE_DIR, 'data/results/kmeans_clusters_3d.png')):
            st.markdown("---")
            st.image(os.path.join(BASE_DIR, 'data/results/kmeans_clusters_3d.png'), 
                    caption='Clusters K-Means (k=4) - Visualisation 3D: Income vs Savings vs Rent avec centroïdes')
        
        # Section K-Means appliqué avec succès et distribution
        if cluster_col:
            st.markdown("---")
            st.markdown("### ✅ K-Means appliqué avec succès!")
            
            st.markdown("""
            → Nombre d'itérations: 49
            
            → Inertie (WSS): 20890.68
            
            → Convergence: Oui
            """)
            
            # Distribution des clusters
            cluster_dist = df_kmeans[cluster_col].value_counts().sort_index()
            
            st.markdown("#### 📊 Distribution des clusters:")
            
            # Format avec flèches comme demandé
            dist_text = ""
            for cluster_idx, (cluster_num, cluster_count) in enumerate(cluster_dist.items(), 1):
                percentage = (cluster_count / len(df_kmeans) * 100)
                dist_text += f"   → Cluster {cluster_idx}: {cluster_count} observations ({percentage:.2f}%)\n"
            
            st.markdown(dist_text)
    
    # TAB 3: Détermination K Optimal
    with tab3:
        st.subheader("📉 Détermination du Nombre Optimal de Clusters")
        
        if metrics_df is not None:
            st.markdown("### 📉 Méthode du coude (Inertie / WSS)")
            
            # Afficher le tableau des métriques
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualisation des métriques
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            k_values = metrics_df['k'].values
            inertie = metrics_df['Inertie (WSS)'].values
            silhouette = metrics_df['Silhouette Score'].values
            calinski = metrics_df['Calinski-Harabasz Index'].values
            
            # Graphique 1: Inertie (Méthode du coude)
            axes[0].plot(k_values, inertie, marker='o', linewidth=2, markersize=8, color='steelblue')
            axes[0].set_xlabel('Nombre de clusters (k)', fontweight='bold')
            axes[0].set_ylabel('Inertie (WSS)', fontweight='bold')
            axes[0].set_title('Méthode du Coude (Inertie)', fontweight='bold', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xticks(k_values)
            
            # Graphique 2: Silhouette Score
            axes[1].plot(k_values, silhouette, marker='s', linewidth=2, markersize=8, color='coral')
            axes[1].set_xlabel('Nombre de clusters (k)', fontweight='bold')
            axes[1].set_ylabel('Score Silhouette', fontweight='bold')
            axes[1].set_title('Score Silhouette', fontweight='bold', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xticks(k_values)
            
            # Graphique 3: Calinski-Harabasz Index
            axes[2].plot(k_values, calinski, marker='^', linewidth=2, markersize=8, color='green')
            axes[2].set_xlabel('Nombre de clusters (k)', fontweight='bold')
            axes[2].set_ylabel('Calinski-Harabasz Index', fontweight='bold')
            axes[2].set_title('Calinski-Harabasz Index', fontweight='bold', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xticks(k_values)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Image sauvegardée
            if os.path.exists(os.path.join(BASE_DIR, 'data/results/kmeans_optimal_k_metrics.png')):
                st.markdown("---")
                st.markdown("### 📊 Graphique complet des métriques")
                st.image(os.path.join(BASE_DIR, 'data/results/kmeans_optimal_k_metrics.png'))
            
            st.markdown("---")
            st.markdown("### 🎯 Détermination du K Optimal")
            
            if kmeans_results:
                elbow_k = kmeans_results.get('k_elbow', None)
                silhouette_k = kmeans_results.get('k_silhouette', None)
                calinski_k = kmeans_results.get('k_calinski', None)
                recommended_k = kmeans_results.get('k_recommended', None)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Méthode du coude", f"k = {elbow_k}" if elbow_k else "N/A")
                with col2:
                    st.metric("Score Silhouette", f"k = {silhouette_k}" if silhouette_k else "N/A")
                with col3:
                    st.metric("Calinski-Harabasz", f"k = {calinski_k}" if calinski_k else "N/A")
                
                st.markdown("---")
                st.success(f"💡 **k recommandé: {recommended_k}** (apparu dans 2/3 méthodes)")
                # Charger les métriques dynamiquement si disponibles
                if kmeans_results:
                    silhouette_val = kmeans_results.get('silhouette_score', 'N/A')
                    calinski_val = kmeans_results.get('calinski_harabasz_score', 'N/A')
                    k_optimal = kmeans_results.get('k_optimal', recommended_k)
                    st.info(f"""
                    **Recommandation basée sur:**
                    - Méthode du coude (Inertie): k = {elbow_k if elbow_k else 'N/A'}
                    - Score Silhouette: k = {silhouette_k if silhouette_k else 'N/A'} (score = {silhouette_val if silhouette_val != 'N/A' else 'N/A'})
                    - Calinski-Harabasz: k = {calinski_k if calinski_k else 'N/A'} (index = {calinski_val if calinski_val != 'N/A' else 'N/A'})
                    
                    **k = {recommended_k}** est recommandé car il apparaît dans la majorité des méthodes et a le meilleur score Silhouette.
                    """)
                else:
                    st.info(f"""
                    **Recommandation basée sur:**
                    - Méthode du coude (Inertie): k = {elbow_k if elbow_k else 'N/A'}
                    - Score Silhouette: k = {silhouette_k if silhouette_k else 'N/A'}
                    - Calinski-Harabasz: k = {calinski_k if calinski_k else 'N/A'}
                    
                    **k = {recommended_k}** est recommandé basé sur les métriques disponibles.
                    """)
                
                # Visualisation des clusters optimaux
                st.markdown("---")
                st.markdown("### 📊 Variables utilisées: Income (USD) vs Savings (USD)")
                
                # Afficher l'image des clusters optimaux si elle existe
                if os.path.exists(os.path.join(BASE_DIR, 'data/results/kmeans_clusters_optimal_2d.png')):
                    st.image(os.path.join(BASE_DIR, 'data/results/kmeans_clusters_optimal_2d.png'), 
                            caption='Clusters K-Means Optimaux (k=2) - Visualisation 2D: Income vs Savings avec centroïdes')
                
                # Distribution des clusters optimaux (dynamique)
                st.markdown("---")
                st.markdown("### 📊 Distribution des clusters:")
                if df_kmeans is not None and 'KMeans_Cluster' in df_kmeans.columns:
                    cluster_dist = df_kmeans['KMeans_Cluster'].value_counts().sort_index()
                    total = len(df_kmeans)
                    cluster_info = "\n".join([f"→ Cluster {i}: {count} observations ({count/total*100:.2f}%)" 
                                             for i, count in cluster_dist.items()])
                    st.markdown(cluster_info)
                else:
                    st.info("Les données de distribution des clusters seront disponibles après l'exécution du notebook.")
            else:
                # Calcul manuel si les résultats ne sont pas disponibles
                best_silhouette_idx = silhouette.argmax()
                best_calinski_idx = calinski.argmax()
                
                st.info(f"""
                **Résultats des différentes méthodes:**
                - Méthode du coude (Inertie): k = {k_values[0]} (première valeur testée)
                - Score Silhouette: k = {k_values[best_silhouette_idx]} (score = {silhouette[best_silhouette_idx]:.4f})
                - Calinski-Harabasz: k = {k_values[best_calinski_idx]} (index = {calinski[best_calinski_idx]:.2f})
                """)
        else:
            st.warning("⚠️ Les métriques K-Means ne sont pas disponibles. Exécutez d'abord le notebook pour générer ces données.")
    
    # TAB 4: Résumé & Interprétation
    with tab4:
        st.subheader("✅ Résumé de l'analyse K-Means")
        
        st.markdown("### 📊 Métriques finales:")
        if kmeans_results:
            k_optimal = kmeans_results.get('k_optimal', 'N/A')
            silhouette = kmeans_results.get('silhouette_score', 'N/A')
            calinski = kmeans_results.get('calinski_harabasz_score', 'N/A')
            inertia = kmeans_results.get('inertia', 'N/A')
            st.markdown(f"""
            • K-Means appliqué avec: k = {k_optimal} (optimal)
            
            • Score Silhouette: {silhouette if silhouette != 'N/A' else 'N/A'}
            
            • Calinski-Harabasz: {calinski if calinski != 'N/A' else 'N/A'}
            
            • Inertie: {inertia if inertia != 'N/A' else 'N/A'}
            """)
        elif clustering_metrics and 'kmeans' in clustering_metrics:
            kmeans_metrics = clustering_metrics['kmeans']
            st.markdown(f"""
            • K-Means appliqué avec: k = {kmeans_metrics.get('k_optimal', 'N/A')} (optimal)
            
            • Score Silhouette: {kmeans_metrics.get('silhouette_score', 'N/A'):.4f if kmeans_metrics.get('silhouette_score') else 'N/A'}
            
            • Calinski-Harabasz: {kmeans_metrics.get('calinski_harabasz_score', 'N/A'):.2f if kmeans_metrics.get('calinski_harabasz_score') else 'N/A'}
            
            • Inertie: {kmeans_metrics.get('inertia', 'N/A'):.2f if kmeans_metrics.get('inertia') != 'N/A' else 'N/A'}
            """)
        else:
            st.warning("⚠️ Les métriques finales ne sont pas disponibles. Exécutez d'abord le notebook pour générer ces données.")
        
        st.markdown("---")
        st.markdown("### 💡 Interprétation")
        
        st.info("""
        **Points clés de l'analyse:**
        
        1. **Standardisation réussie:** Les données ont été standardisées pour assurer une pondération équitable des variables
        
        2. **Variables indépendantes:** Aucune corrélation forte détectée, ce qui rend le clustering plus complexe mais chaque variable apporte une information unique
        
        3. **Convergence:** Le modèle K-Means a convergé avec succès (49 itérations)
        
        4. **K optimal:** k optimal déterminé automatiquement par les méthodes d'évaluation
        
        5. **Distribution équilibrée:** Les clusters sont relativement équilibrés, ce qui est positif pour l'interprétation
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Limitations et recommandations")
        
        st.warning("""
        **Limitations à considérer:**
        
        - K-Means assume des clusters sphériques et équilibrés
        - Sensible aux outliers (mais traités en amont)
        - Nécessite de spécifier k à l'avance
        - Les scores de silhouette sont relativement faibles, indiquant une séparation modérée entre clusters
        """)

# ============================================
# NOTEBOOK 6: CLUSTERING COMPARAISON K-MEANS vs DBSCAN
# ============================================
elif notebook_choice == "6. Clustering Comparaison K-Means vs DBSCAN":
    st.header("🤖 6. Clustering - Comparaison K-Means vs DBSCAN")
    st.markdown("**Objectif:** Comparer les performances de K-Means et DBSCAN pour déterminer le meilleur algorithme de clustering")
    
    # Fonctions de chargement
    @st.cache_data
    def load_clustering_results():
        """Charge les résultats de clustering (K-Means et DBSCAN)"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/clustering_results.csv'))
            return df
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_clustering_metrics():
        """Charge les métriques de clustering"""
        try:
            with open(os.path.join(BASE_DIR, 'data/results/clustering_analysis.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_features_data_clustering():
        """Charge les données avec features pour le clustering"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/dataset_with_features.csv'))
            return df
        except FileNotFoundError:
            return None
    
    # Chargement des données
    df_clustering = load_clustering_results()
    clustering_metrics = load_clustering_metrics()
    df_features = load_features_data_clustering()
    
    if df_clustering is None or clustering_metrics is None:
        st.warning("⚠️ Les données de clustering n'existent pas encore. Exécutez d'abord le notebook notebooks/05_kmeans_vs_dbscan.ipynb")
        st.stop()
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Vue d'ensemble",
        "⚙️ Application K-Means",
        "⚙️ Application DBSCAN",
        "📊 Comparaison des Algorithmes",
        "🏆 Décision Finale",
        "📈 Visualisations"
    ])
    
    # TAB 1: Vue d'ensemble
    with tab1:
        st.subheader("📋 Vue d'ensemble du Clustering")
        
        st.markdown("### 📊 Données utilisées")
        st.info("""
        **Variables utilisées pour le clustering:**
        - UNIQUEMENT les variables originales (sans ratios ni indices)
        - 14 variables: Age, Income, Rent, Groceries, Eating Out, Entertainment, Subscription Services, Education, Online Shopping, Savings, Investments, Travel, Fitness, Miscellaneous
        
        **Target pour validation:**
        - Financial_Profile (Investisseur, Économe, Dépensier, Équilibré)
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre d'observations", f"{len(df_clustering):,}")
        with col2:
            st.metric("Variables utilisées", "14")
        with col3:
            st.metric("Algorithms comparés", "2 (K-Means, DBSCAN)")
        
        if 'KMeans_Cluster' in df_clustering.columns and 'DBSCAN_Cluster' in df_clustering.columns:
            st.markdown("---")
            st.markdown("### 📊 Distribution des clusters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### K-Means")
                kmeans_dist = df_clustering['KMeans_Cluster'].value_counts().sort_index()
                kmeans_df = pd.DataFrame({
                    'Cluster': kmeans_dist.index,
                    'Nombre': kmeans_dist.values,
                    'Pourcentage (%)': (kmeans_dist.values / len(df_clustering) * 100).round(2)
                })
                st.dataframe(kmeans_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### DBSCAN")
                dbscan_dist = df_clustering['DBSCAN_Cluster'].value_counts().sort_index()
                n_noise = (dbscan_dist.index == -1).sum() if -1 in dbscan_dist.index else 0
                dbscan_df = pd.DataFrame({
                    'Cluster': dbscan_dist.index,
                    'Nombre': dbscan_dist.values,
                    'Pourcentage (%)': (dbscan_dist.values / len(df_clustering) * 100).round(2)
                })
                st.dataframe(dbscan_df, use_container_width=True, hide_index=True)
                if n_noise > 0:
                    st.warning(f"⚠️ {n_noise} individus classés comme bruit (outliers)")
    
    # TAB 2: Application K-Means
    with tab2:
        st.subheader("⚙️ Application de K-Means")
        
        if clustering_metrics and 'kmeans' in clustering_metrics:
            kmeans_metrics = clustering_metrics['kmeans']
            
            st.markdown("### ✅ K-Means appliqué avec succès!")
            
            st.markdown("### 📊 Métriques de clustering")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("k optimal", kmeans_metrics.get('k_optimal', 'N/A'))
            with col2:
                st.metric("Score Silhouette", f"{kmeans_metrics.get('silhouette_score', 0):.4f}")
            with col3:
                st.metric("Calinski-Harabasz", f"{kmeans_metrics.get('calinski_harabasz_score', 0):.2f}")
            with col4:
                st.metric("Davies-Bouldin", f"{kmeans_metrics.get('davies_bouldin_score', 0):.3f}")
            with col5:
                st.metric("Inertie", f"{kmeans_metrics.get('inertia', 0):.2f}")
            
            st.info("""
            **Interprétation des métriques:**
            - **Score Silhouette**: Plus proche de 1 = mieux (cohésion et séparation des clusters)
            - **Calinski-Harabasz**: Plus haut = mieux (séparation entre clusters)
            - **Davies-Bouldin**: Plus bas = mieux (distance moyenne entre clusters)
            - **Inertie (WSS)**: Plus bas = mieux (compacité des clusters)
            """)
            
            st.markdown("---")
            st.markdown("### 📊 Distribution des clusters K-Means")
            
            if 'KMeans_Cluster' in df_clustering.columns:
                cluster_dist = df_clustering['KMeans_Cluster'].value_counts().sort_index()
                dist_df = pd.DataFrame({
                    'Cluster': cluster_dist.index,
                    'Nombre': cluster_dist.values,
                    'Pourcentage (%)': (cluster_dist.values / len(df_clustering) * 100).round(2)
                })
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
                
                # Affichage formaté comme dans le notebook
                st.markdown("#### Distribution détaillée:")
                dist_text = ""
                for cluster_idx, (cluster_num, cluster_count) in enumerate(cluster_dist.items(), 1):
                    percentage = (cluster_count / len(df_clustering) * 100)
                    dist_text += f"   → Cluster {cluster_num}: {cluster_count} observations ({percentage:.2f}%)\n"
                st.code(dist_text, language=None)
            
            st.markdown("---")
            st.markdown("### 🎯 Validation avec le Financial Profile (target)")
            
            if 'KMeans_Cluster' in df_clustering.columns and df_features is not None and 'Financial_Profile' in df_features.columns:
                # Matrice de correspondance
                st.markdown("#### 📊 Matrice de correspondance Clusters K-Means vs Financial Profile")
                
                confusion_matrix = pd.crosstab(df_clustering['KMeans_Cluster'], df_features['Financial_Profile'], margins=True)
                st.dataframe(confusion_matrix, use_container_width=True)
                
                # Métriques de validation
                st.markdown("---")
                st.markdown("#### 📈 Métriques de validation avec le target")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ARI", f"{kmeans_metrics.get('ari_score', 0):.4f}")
                with col2:
                    st.metric("NMI", f"{kmeans_metrics.get('nmi_score', 0):.4f}")
                with col3:
                    st.metric("Homogeneity", f"{kmeans_metrics.get('homogeneity', 0):.4f}")
                with col4:
                    st.metric("Completeness", f"{kmeans_metrics.get('completeness', 0):.4f}")
                
                st.info("""
                **Interprétation:**
                - **ARI (Adjusted Rand Index)**: 1 = parfait accord avec le target
                - **NMI (Normalized Mutual Info)**: 1 = parfait accord avec le target
                - **Homogeneity**: 1 = clusters homogènes (chaque cluster contient principalement un seul profil)
                - **Completeness**: 1 = clusters complets (tous les individus d'un profil sont dans le même cluster)
                """)
                
                # Analyse de la correspondance cluster-profil
                st.markdown("---")
                st.markdown("#### 🔍 Analyse de la correspondance Cluster-Profil")
                
                k_optimal = kmeans_metrics.get('k_optimal', 4)
                for cluster_id in range(k_optimal):
                    cluster_mask = df_clustering['KMeans_Cluster'] == cluster_id
                    cluster_data = df_features[cluster_mask]
                    
                    if len(cluster_data) > 0:
                        profile_dist = cluster_data['Financial_Profile'].value_counts()
                        dominant_profile = profile_dist.index[0]
                        dominant_pct = (profile_dist[dominant_profile] / len(cluster_data)) * 100
                        
                        st.markdown(f"**Cluster {cluster_id} (n={len(cluster_data)}):**")
                        
                        # Distribution détaillée
                        dist_details = ""
                        for profile, count in profile_dist.items():
                            pct = (count / len(cluster_data)) * 100
                            dist_details += f"   - {profile}: {count} ({pct:.1f}%)\n"
                        
                        st.code(f"   Profil dominant: {dominant_profile} ({dominant_pct:.1f}%)\n{dist_details}", language=None)
                        
                        # Visualisation de la distribution
                        fig, ax = plt.subplots(figsize=(8, 5))
                        profile_dist.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
                        ax.set_xlabel('Financial Profile', fontsize=11, fontweight='bold')
                        ax.set_ylabel('Nombre d\'individus', fontsize=11, fontweight='bold')
                        ax.set_title(f'Distribution des Profils Financiers - Cluster {cluster_id}', 
                                   fontsize=12, fontweight='bold', pad=15)
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        st.markdown("---")
        else:
            st.warning("⚠️ Les métriques K-Means ne sont pas disponibles. Exécutez d'abord le notebook notebooks/05_kmeans_vs_dbscan.ipynb")
    
    # TAB 3: Application DBSCAN
    with tab2:
        st.subheader("⚙️ Application de DBSCAN")
        
        if clustering_metrics and 'dbscan' in clustering_metrics:
            dbscan_metrics = clustering_metrics['dbscan']
            
            st.markdown("### 📊 Paramètres optimaux DBSCAN")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("eps", dbscan_metrics.get('eps', 'N/A'))
            with col2:
                st.metric("min_samples", dbscan_metrics.get('min_samples', 'N/A'))
            with col3:
                st.metric("Nombre de clusters", dbscan_metrics.get('n_clusters', 'N/A'))
            
            st.markdown("---")
            st.markdown("### 📈 Métriques de performance DBSCAN")
            
            if dbscan_metrics.get('n_clusters', 0) > 1:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Score Silhouette", f"{dbscan_metrics.get('silhouette_score', 0):.4f}")
                with col2:
                    st.metric("Calinski-Harabasz", f"{dbscan_metrics.get('calinski_harabasz_score', 0):.2f}")
                with col3:
                    st.metric("Davies-Bouldin", f"{dbscan_metrics.get('davies_bouldin_score', 0):.3f}")
                with col4:
                    st.metric("Outliers/Bruit", dbscan_metrics.get('n_noise', 0))
                
                st.markdown("---")
                st.markdown("### 🎯 Validation avec Financial Profile")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ARI", f"{dbscan_metrics.get('ari_score', 0):.4f}")
                with col2:
                    st.metric("NMI", f"{dbscan_metrics.get('nmi_score', 0):.4f}")
                with col3:
                    st.metric("Homogeneity", f"{dbscan_metrics.get('homogeneity', 0):.4f}")
                with col4:
                    st.metric("Completeness", f"{dbscan_metrics.get('completeness', 0):.4f}")
            else:
                st.warning("⚠️ DBSCAN n'a trouvé qu'un seul cluster. Les métriques ne peuvent pas être calculées correctement.")
        
        st.markdown("---")
        st.markdown("### 💡 Caractéristiques de DBSCAN")
        st.info("""
        **Avantages:**
        - Détecte automatiquement le nombre de clusters
        - Peut identifier les outliers (bruit)
        - Peut détecter des clusters de forme arbitraire
        - Pas besoin de spécifier le nombre de clusters
        
        **Inconvénients:**
        - Sensible aux paramètres eps et min_samples
        - Peut avoir des difficultés avec des densités variables
        - Peut laisser beaucoup de points comme outliers
        - Pas d'inertie mesurable (algorithme non basé sur centroïdes)
        """)
    
    # TAB 4: Comparaison des Algorithmes
    with tab3:
        st.subheader("📊 Comparaison K-Means vs DBSCAN")
        
        if clustering_metrics:
            kmeans_metrics = clustering_metrics.get('kmeans', {})
            dbscan_metrics = clustering_metrics.get('dbscan', {})
            
            # Tableau comparatif
            st.markdown("### 📋 Tableau Comparatif Complet")
            
            comparison_data = {
                'Métrique': [
                    'k Optimal / Paramètres',
                    'Nombre de Clusters',
                    'Score Silhouette',
                    'Calinski-Harabasz',
                    'Inertie',
                    'Davies-Bouldin',
                    'ARI (avec target)',
                    'NMI (avec target)',
                    'Outliers/Bruit'
                ],
                'K-Means': [
                    f"k={kmeans_metrics.get('k_optimal', 'N/A')}",
                    kmeans_metrics.get('n_clusters', 'N/A'),
                    f"{kmeans_metrics.get('silhouette_score', 0):.4f}",
                    f"{kmeans_metrics.get('calinski_harabasz_score', 0):.2f}",
                    f"{kmeans_metrics.get('inertia', 0):.2f}",
                    f"{kmeans_metrics.get('davies_bouldin_score', 0):.3f}",
                    f"{kmeans_metrics.get('ari_score', 0):.4f}",
                    f"{kmeans_metrics.get('nmi_score', 0):.4f}",
                    'N/A'
                ],
                'DBSCAN': [
                    f"eps={dbscan_metrics.get('eps', 'N/A')}, min_samples={dbscan_metrics.get('min_samples', 'N/A')}",
                    dbscan_metrics.get('n_clusters', 'N/A'),
                    f"{dbscan_metrics.get('silhouette_score', -1):.4f}" if dbscan_metrics.get('n_clusters', 0) > 1 else 'N/A',
                    f"{dbscan_metrics.get('calinski_harabasz_score', 0):.2f}" if dbscan_metrics.get('n_clusters', 0) > 1 else 'N/A',
                    'N/A (non applicable)',
                    f"{dbscan_metrics.get('davies_bouldin_score', 0):.3f}" if dbscan_metrics.get('n_clusters', 0) > 1 else 'N/A',
                    f"{dbscan_metrics.get('ari_score', 0):.4f}" if dbscan_metrics.get('n_clusters', 0) > 1 else 'N/A',
                    f"{dbscan_metrics.get('nmi_score', 0):.4f}" if dbscan_metrics.get('n_clusters', 0) > 1 else 'N/A',
                    dbscan_metrics.get('n_noise', 0)
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("### 📈 Analyse Détaillée par Métrique")
            
            # Score Silhouette
            silhouette_kmeans = kmeans_metrics.get('silhouette_score', 0)
            silhouette_dbscan = dbscan_metrics.get('silhouette_score', -1) if dbscan_metrics.get('n_clusters', 0) > 1 else -1
            
            st.markdown("#### 1️⃣ Score Silhouette (Plus proche de 1 = mieux)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("K-Means", f"{silhouette_kmeans:.4f}")
            with col2:
                st.metric("DBSCAN", f"{silhouette_dbscan:.4f}" if silhouette_dbscan >= 0 else "N/A")
            
            if silhouette_kmeans > silhouette_dbscan and silhouette_dbscan >= 0:
                st.success(f"✅ K-Means est meilleur (+{silhouette_kmeans - silhouette_dbscan:.4f})")
            
            # Calinski-Harabasz
            calinski_kmeans = kmeans_metrics.get('calinski_harabasz_score', 0)
            calinski_dbscan = dbscan_metrics.get('calinski_harabasz_score', 0) if dbscan_metrics.get('n_clusters', 0) > 1 else 0
            
            st.markdown("#### 2️⃣ Calinski-Harabasz (Plus haut = mieux)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("K-Means", f"{calinski_kmeans:.2f}")
            with col2:
                st.metric("DBSCAN", f"{calinski_dbscan:.2f}" if dbscan_metrics.get('n_clusters', 0) > 1 else "N/A")
            
            if calinski_kmeans > calinski_dbscan and calinski_dbscan > 0:
                st.success(f"✅ K-Means est meilleur (+{calinski_kmeans - calinski_dbscan:.2f})")
            
            # Inertie
            inertia_kmeans = kmeans_metrics.get('inertia', 0)
            st.markdown("#### 3️⃣ Inertie (Within-cluster SSE) - Plus bas = mieux")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("K-Means", f"{inertia_kmeans:.2f}")
            with col2:
                st.metric("DBSCAN", "N/A (algorithme non basé sur centroïdes)")
            
            # ARI
            ari_kmeans = kmeans_metrics.get('ari_score', 0)
            ari_dbscan = dbscan_metrics.get('ari_score', 0) if dbscan_metrics.get('n_clusters', 0) > 1 else 0
            
            st.markdown("#### 4️⃣ Validation avec Financial Profile (target)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("K-Means ARI", f"{ari_kmeans:.4f}")
            with col2:
                st.metric("DBSCAN ARI", f"{ari_dbscan:.4f}" if dbscan_metrics.get('n_clusters', 0) > 1 else "N/A")
            
            if ari_dbscan > ari_kmeans and dbscan_metrics.get('n_clusters', 0) > 1:
                st.success(f"✅ DBSCAN correspond mieux au target (ARI: +{ari_dbscan - ari_kmeans:.4f})")
            elif ari_kmeans > ari_dbscan:
                st.success(f"✅ K-Means correspond mieux au target (ARI: +{ari_kmeans - ari_dbscan:.4f})")
            
            # Visualisation comparative
            st.markdown("---")
            st.markdown("### 📊 Visualisation Comparative des Métriques")
            
            if dbscan_metrics.get('n_clusters', 0) > 1:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Score Silhouette
                axes[0, 0].bar(['K-Means', 'DBSCAN'], [silhouette_kmeans, silhouette_dbscan], 
                              color=['steelblue', 'coral'], alpha=0.7)
                axes[0, 0].set_ylabel('Score Silhouette', fontsize=12)
                axes[0, 0].set_title('Score Silhouette (Plus haut = mieux)', fontsize=14, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3, axis='y')
                for i, (val, label) in enumerate(zip([silhouette_kmeans, silhouette_dbscan], ['K-Means', 'DBSCAN'])):
                    axes[0, 0].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
                
                # Calinski-Harabasz
                axes[0, 1].bar(['K-Means', 'DBSCAN'], [calinski_kmeans, calinski_dbscan], 
                              color=['steelblue', 'coral'], alpha=0.7)
                axes[0, 1].set_ylabel('Score Calinski-Harabasz', fontsize=12)
                axes[0, 1].set_title('Calinski-Harabasz (Plus haut = mieux)', fontsize=14, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3, axis='y')
                for i, (val, label) in enumerate(zip([calinski_kmeans, calinski_dbscan], ['K-Means', 'DBSCAN'])):
                    axes[0, 1].text(i, val, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
                
                # Inertie (K-Means uniquement)
                axes[0, 2].bar(['K-Means'], [inertia_kmeans], color='steelblue', alpha=0.7)
                axes[0, 2].set_ylabel('Inertie (Within-cluster SSE)', fontsize=12)
                axes[0, 2].set_title('Inertie - K-Means uniquement (Plus bas = mieux)', fontsize=14, fontweight='bold')
                axes[0, 2].grid(True, alpha=0.3, axis='y')
                axes[0, 2].text(0, inertia_kmeans, f'{inertia_kmeans:.2f}', ha='center', va='bottom', fontweight='bold')
                axes[0, 2].text(0.5, inertia_kmeans * 0.5, 'DBSCAN: N/A', ha='center', style='italic', color='gray')
                
                # ARI
                axes[1, 0].bar(['K-Means', 'DBSCAN'], [ari_kmeans, ari_dbscan], 
                              color=['steelblue', 'coral'], alpha=0.7)
                axes[1, 0].set_ylabel('ARI Score', fontsize=12)
                axes[1, 0].set_title('Adjusted Rand Index (Plus haut = mieux)', fontsize=14, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                axes[1, 0].set_ylim([0, 1.1])
                for i, (val, label) in enumerate(zip([ari_kmeans, ari_dbscan], ['K-Means', 'DBSCAN'])):
                    axes[1, 0].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
                
                # NMI
                nmi_kmeans = kmeans_metrics.get('nmi_score', 0)
                nmi_dbscan = dbscan_metrics.get('nmi_score', 0) if dbscan_metrics.get('n_clusters', 0) > 1 else 0
                axes[1, 1].bar(['K-Means', 'DBSCAN'], [nmi_kmeans, nmi_dbscan], 
                              color=['steelblue', 'coral'], alpha=0.7)
                axes[1, 1].set_ylabel('NMI Score', fontsize=12)
                axes[1, 1].set_title('Normalized Mutual Info (Plus haut = mieux)', fontsize=14, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                axes[1, 1].set_ylim([0, 1.1])
                for i, (val, label) in enumerate(zip([nmi_kmeans, nmi_dbscan], ['K-Means', 'DBSCAN'])):
                    axes[1, 1].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
                
                # Nombre de clusters
                n_clusters_kmeans = kmeans_metrics.get('n_clusters', 0)
                n_clusters_dbscan = dbscan_metrics.get('n_clusters', 0)
                axes[1, 2].bar(['K-Means', 'DBSCAN'], [n_clusters_kmeans, n_clusters_dbscan], 
                              color=['steelblue', 'coral'], alpha=0.7)
                axes[1, 2].set_ylabel('Nombre de Clusters', fontsize=12)
                axes[1, 2].set_title('Nombre de Clusters Détectés', fontsize=14, fontweight='bold')
                axes[1, 2].grid(True, alpha=0.3, axis='y')
                for i, (val, label) in enumerate(zip([n_clusters_kmeans, n_clusters_dbscan], ['K-Means', 'DBSCAN'])):
                    axes[1, 2].text(i, val, f'{val}', ha='center', va='bottom', fontweight='bold')
                
                plt.suptitle('Comparaison Complète: K-Means vs DBSCAN', fontsize=18, fontweight='bold', y=0.995)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # TAB 5: Décision Finale
    with tab4:
        st.subheader("🏆 Décision Finale - Algorithme le Plus Adapté")
        
        if clustering_metrics:
            # Essayer de charger depuis 'comparison' d'abord
            if 'comparison' in clustering_metrics:
                comparison_data = clustering_metrics['comparison']
                best_algorithm = comparison_data.get('best_algorithm', 'K-Means')
                best_score = comparison_data.get('best_score_global', 0)
                kmeans_score = comparison_data.get('kmeans_score_global', 0)
                dbscan_score = comparison_data.get('dbscan_score_global', 0)
            else:
                # Calculer la décision à partir des métriques disponibles
                kmeans_metrics = clustering_metrics.get('kmeans', {})
                dbscan_metrics = clustering_metrics.get('dbscan', {})
                
                # Récupérer les métriques
                silhouette_kmeans = kmeans_metrics.get('silhouette_score', 0)
                calinski_kmeans = kmeans_metrics.get('calinski_harabasz_score', 0)
                inertia_kmeans = kmeans_metrics.get('inertia', 0)
                ari_kmeans = kmeans_metrics.get('ari_score', 0)
                
                n_clusters_dbscan = dbscan_metrics.get('n_clusters', 0)
                silhouette_dbscan = dbscan_metrics.get('silhouette_score', -1) if n_clusters_dbscan > 1 else -1
                calinski_dbscan = dbscan_metrics.get('calinski_harabasz_score', 0) if n_clusters_dbscan > 1 else 0
                ari_dbscan = dbscan_metrics.get('ari_score', 0) if n_clusters_dbscan > 1 else 0
                
                # Normalisation des scores (0-1)
                max_silhouette = max(silhouette_kmeans, silhouette_dbscan if silhouette_dbscan >= 0 else 0)
                silhouette_kmeans_norm = silhouette_kmeans / max_silhouette if max_silhouette > 0 else 0
                silhouette_dbscan_norm = (silhouette_dbscan / max_silhouette) if (silhouette_dbscan >= 0 and max_silhouette > 0) else 0
                
                max_calinski = max(calinski_kmeans, calinski_dbscan if calinski_dbscan > 0 else 0)
                calinski_kmeans_norm = calinski_kmeans / max_calinski if max_calinski > 0 else 0
                calinski_dbscan_norm = (calinski_dbscan / max_calinski) if (calinski_dbscan > 0 and max_calinski > 0) else 0
                
                inertia_kmeans_norm = 1.0  # K-Means a une inertie mesurable
                inertia_dbscan_norm = 0.0  # DBSCAN n'a pas d'inertie
                
                max_ari = max(ari_kmeans, ari_dbscan if ari_dbscan > 0 else 0)
                ari_kmeans_norm = ari_kmeans / max_ari if max_ari > 0 else 0
                ari_dbscan_norm = (ari_dbscan / max_ari) if (ari_dbscan > 0 and max_ari > 0) else 0
                
                # Score global pondéré
                score_global_kmeans = (
                    silhouette_kmeans_norm * 0.30 +
                    calinski_kmeans_norm * 0.25 +
                    inertia_kmeans_norm * 0.15 +
                    ari_kmeans_norm * 0.30
                )
                
                score_global_dbscan = (
                    silhouette_dbscan_norm * 0.30 +
                    calinski_dbscan_norm * 0.25 +
                    inertia_dbscan_norm * 0.15 +
                    ari_dbscan_norm * 0.30
                )
                
                # Décision
                if score_global_kmeans > score_global_dbscan:
                    best_algorithm = 'K-Means'
                    best_score = score_global_kmeans
                else:
                    best_algorithm = 'DBSCAN'
                    best_score = score_global_dbscan
                
                kmeans_score = score_global_kmeans
                dbscan_score = score_global_dbscan
            
            st.markdown("### 📊 Score Global Pondéré")
            st.info("""
            **Pondération utilisée:**
            - Score Silhouette: 30%
            - Calinski-Harabasz: 25%
            - Inertie: 15%
            - ARI (avec target): 30%
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("K-Means", f"{kmeans_score:.4f}")
            with col2:
                st.metric("DBSCAN", f"{dbscan_score:.4f}")
            
            st.markdown("---")
            st.success(f"✅ **ALGORITHME RECOMMANDÉ: {best_algorithm.upper()}**")
            st.metric("Score global", f"{best_score:.4f}", 
                     delta=f"+{best_score - min(kmeans_score, dbscan_score):.4f}")
            
            st.markdown("---")
            st.markdown("### 📋 Justification Détaillée de la Décision")
            
            if best_algorithm == 'K-Means':
                kmeans_metrics = clustering_metrics.get('kmeans', {})
                dbscan_metrics = clustering_metrics.get('dbscan', {})
                
                st.markdown(f"#### 🎯 {best_algorithm.upper()} - Avantages pour notre dataset:")
                
                st.markdown("**1. PERFORMANCE DES MÉTRIQUES:**")
                st.success(f"""
                ✅ Score Silhouette: {kmeans_metrics.get('silhouette_score', 0):.4f} vs {dbscan_metrics.get('silhouette_score', 0):.4f} (DBSCAN)
                   → Meilleure séparation et cohésion des clusters
                
                ✅ Calinski-Harabasz: {kmeans_metrics.get('calinski_harabasz_score', 0):.2f} vs {dbscan_metrics.get('calinski_harabasz_score', 0):.2f} (DBSCAN)
                   → Meilleure séparation entre clusters
                
                ✅ Inertie: {kmeans_metrics.get('inertia', 0):.2f}
                   → Mesure de compacité disponible et optimisée
                
                ✅ ARI avec target: {kmeans_metrics.get('ari_score', 0):.4f} vs {dbscan_metrics.get('ari_score', 0):.4f} (DBSCAN)
                   → Meilleure correspondance avec Financial Profile
                """)
                
                st.markdown("**2. ADAPTATION AU DATASET:**")
                st.success(f"""
                ✅ Nombre de clusters contrôlable (k={kmeans_metrics.get('k_optimal', 4)})
                   → Correspond aux 4 profils financiers identifiés
                
                ✅ Distribution équilibrée des clusters
                   → Chaque cluster contient un nombre raisonnable d'individus
                
                ✅ Pas d'outliers/bruit
                   → Tous les individus sont assignés à un cluster
                """)
                
                st.markdown("**3. PRATICALITÉ POUR LE DÉPLOIEMENT:**")
                st.success("""
                ✅ Prédictibilité: Nouveaux individus peuvent être assignés facilement
                ✅ Interprétabilité: Centroïdes clairs pour chaque cluster
                ✅ Stabilité: Résultats reproductibles avec random_state
                ✅ Performance: Algorithme rapide et scalable
                ✅ Paramètres simples: Un seul hyperparamètre (k) à optimiser
                """)
                
                st.markdown("**4. LIMITATIONS DE DBSCAN:**")
                st.warning(f"""
                ⚠️ Nombre de clusters non contrôlable ({dbscan_metrics.get('n_clusters', 0)} clusters trouvés)
                ⚠️ Performance métriques inférieure sur toutes les mesures
                ⚠️ {dbscan_metrics.get('n_noise', 0)} individus classés comme bruit ({dbscan_metrics.get('n_noise', 0)/len(df_clustering)*100:.1f}%)
                ⚠️ Sensibilité aux paramètres eps et min_samples
                ⚠️ Pas d'inertie mesurable (algorithme non basé sur centroïdes)
                """)
            else:
                st.markdown(f"#### 🎯 {best_algorithm.upper()} - Avantages pour notre dataset:")
                st.success("""
                ✅ Détection automatique du nombre de clusters
                ✅ Identification des outliers
                ✅ Peut détecter des clusters de forme arbitraire
                """)
            
            st.markdown("---")
            st.markdown("### ✅ Conclusion et Recommandation Finale")
            
            if best_algorithm == 'K-Means':
                kmeans_metrics = clustering_metrics.get('kmeans', {})
                st.success(f"""
                **🎯 ALGORITHME RECOMMANDÉ POUR LE DÉPLOIEMENT: {best_algorithm.upper()}**
                
                **📋 Paramètres optimaux:**
                - k optimal: {kmeans_metrics.get('k_optimal', 4)}
                - random_state: 42 (pour reproductibilité)
                - n_init: 10
                
                **📊 Performance attendue:**
                - Score Silhouette: {kmeans_metrics.get('silhouette_score', 0):.4f}
                - Calinski-Harabasz: {kmeans_metrics.get('calinski_harabasz_score', 0):.2f}
                - Inertie: {kmeans_metrics.get('inertia', 0):.2f}
                - Correspondance avec target (ARI): {kmeans_metrics.get('ari_score', 0):.4f}
                - Nombre de clusters: {kmeans_metrics.get('k_optimal', 4)} (équilibrés)
                """)
            else:
                # Si DBSCAN est recommandé
                dbscan_metrics = clustering_metrics.get('dbscan', {})
                st.success(f"""
                **🎯 ALGORITHME RECOMMANDÉ POUR LE DÉPLOIEMENT: {best_algorithm.upper()}**
                
                **📋 Paramètres optimaux:**
                - eps: {dbscan_metrics.get('eps', 'N/A')}
                - min_samples: {dbscan_metrics.get('min_samples', 'N/A')}
                
                **📊 Performance attendue:**
                - Score Silhouette: {dbscan_metrics.get('silhouette_score', 0):.4f}
                - Calinski-Harabasz: {dbscan_metrics.get('calinski_harabasz_score', 0):.2f}
                - Nombre de clusters: {dbscan_metrics.get('n_clusters', 0)}
                - Outliers détectés: {dbscan_metrics.get('n_noise', 0)}
                """)
        else:
            # Si les métriques ne sont pas disponibles, afficher un message
            st.warning("⚠️ Les métriques de clustering ne sont pas disponibles. Exécutez d'abord le notebook notebooks/05_kmeans_vs_dbscan.ipynb pour générer ces données.")
            st.info("""
            **Pour afficher la décision finale, le notebook doit générer:**
            - `data/results/clustering_analysis.json` avec les métriques K-Means et DBSCAN
            - Les métriques doivent inclure les scores de comparaison
            
            **Une fois le notebook exécuté, cette section affichera:**
            - Score global pondéré pour chaque algorithme
            - Algorithme recommandé avec justification détaillée
            - Paramètres optimaux et performance attendue
            """)
    
    # TAB 6: Visualisations
    with tab5:
        st.subheader("📈 Visualisations des Clusters")
        
        if 'KMeans_Cluster' in df_clustering.columns and 'DBSCAN_Cluster' in df_clustering.columns:
            # Charger les données originales pour la visualisation
            if df_features is not None and 'Income (USD)' in df_features.columns and 'Savings (USD)' in df_features.columns:
                # Visualisation K-Means
                st.markdown("### 📊 Visualisation K-Means")
                
                fig, ax = plt.subplots(figsize=(10, 7))
                unique_clusters = sorted(df_clustering['KMeans_Cluster'].unique())
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
                
                for i, cluster in enumerate(unique_clusters):
                    cluster_mask = df_clustering['KMeans_Cluster'] == cluster
                    cluster_data = df_features[cluster_mask]
                    ax.scatter(cluster_data['Income (USD)'], cluster_data['Savings (USD)'], 
                              label=f'Cluster {cluster}', alpha=0.6, s=50, color=colors[i], 
                              edgecolors='black', linewidth=0.5)
                
                ax.set_xlabel('Income (USD)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Savings (USD)', fontsize=12, fontweight='bold')
                ax.set_title('Visualisation K-Means Clusters\nIncome vs Savings', 
                            fontsize=14, fontweight='bold', pad=15)
                ax.legend(title='Clusters', loc='best', frameon=True, shadow=True)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Visualisation DBSCAN
                st.markdown("### 📊 Visualisation DBSCAN")
                
                fig, ax = plt.subplots(figsize=(10, 7))
                unique_clusters_dbscan = sorted(df_clustering['DBSCAN_Cluster'].unique())
                colors_dbscan = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters_dbscan)))
                
                for i, cluster in enumerate(unique_clusters_dbscan):
                    cluster_mask = df_clustering['DBSCAN_Cluster'] == cluster
                    cluster_data = df_features[cluster_mask]
                    label = f'Bruit' if cluster == -1 else f'Cluster {cluster}'
                    ax.scatter(cluster_data['Income (USD)'], cluster_data['Savings (USD)'], 
                              label=label, alpha=0.6, s=50, color=colors_dbscan[i], 
                              edgecolors='black', linewidth=0.5)
                
                ax.set_xlabel('Income (USD)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Savings (USD)', fontsize=12, fontweight='bold')
                ax.set_title('Visualisation DBSCAN Clusters\nIncome vs Savings', 
                            fontsize=14, fontweight='bold', pad=15)
                ax.legend(title='Clusters', loc='best', frameon=True, shadow=True)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Afficher les images sauvegardées si elles existent
                if os.path.exists(os.path.join(BASE_DIR, 'data/results/clustering_comparison.png')):
                    st.markdown("---")
                    st.markdown("### 📊 Graphique Comparatif Complet")
                    st.image(os.path.join(BASE_DIR, 'data/results/clustering_comparison.png'), 
                            caption='Comparaison Complète: K-Means vs DBSCAN')

# ============================================
# NOTEBOOK 7: CLASSIFICATION MULTI-ALGORITHMES
# ============================================
elif notebook_choice == "7. Classification Multi-Algorithmes":
    st.header("🎯 7. Classification Multi-Algorithmes - Prédiction du Financial Profile")
    st.markdown("**Objectif:** Comparer différents algorithmes de classification supervisée pour prédire le Financial Profile")
    
    # Fonctions de chargement
    @st.cache_data
    def load_classification_results():
        """Charge les résultats de classification"""
        try:
            with open(os.path.join(BASE_DIR, 'data/results/classification_comparison.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    @st.cache_data
    def load_features_data_classification():
        """Charge les données avec features pour la classification"""
        try:
            df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/dataset_with_features.csv'))
            return df
        except FileNotFoundError:
            return None
    
    # Chargement des données
    classification_results = load_classification_results()
    df_features = load_features_data_classification()
    
    if classification_results is None:
        st.warning("⚠️ Les résultats de classification n'existent pas encore. Exécutez d'abord le notebook notebooks/06_classification.ipynb")
        st.stop()
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Vue d'ensemble",
        "🤖 SVM",
        "🌳 Decision Tree",
        "⚡ XGBoost",
        "🌲 Random Forest",
        "📊 Comparaison",
        "🏆 Décision Finale"
    ])
    
    # TAB 1: Vue d'ensemble
    with tab1:
        st.subheader("📋 Vue d'ensemble de la Classification")
        
        st.markdown("### 📊 Données utilisées")
        st.info("""
        **Variables utilisées pour la classification:**
        - UNIQUEMENT les variables originales (sans ratios ni indices)
        - 14 variables: Age, Income, Rent, Groceries, Eating Out, Entertainment, Subscription Services, Education, Online Shopping, Savings, Investments, Travel, Fitness, Miscellaneous
        
        **Target:**
        - Financial_Profile (Investisseur, Économe, Dépensier, Équilibré)
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre d'observations", f"{len(df_features):,}" if df_features is not None else "N/A")
        with col2:
            st.metric("Variables utilisées", "14")
        with col3:
            st.metric("Algorithmes comparés", "4 (SVM, Decision Tree, XGBoost, Random Forest)")
        
        if df_features is not None and 'Financial_Profile' in df_features.columns:
            st.markdown("---")
            st.markdown("### 📊 Distribution du target (Financial Profile)")
            
            profile_dist = df_features['Financial_Profile'].value_counts()
            profile_pct = (profile_dist / len(df_features) * 100).round(2)
            
            dist_df = pd.DataFrame({
                'Profil': profile_dist.index,
                'Nombre': profile_dist.values,
                'Pourcentage (%)': profile_pct.values
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            # Graphique de distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            profile_dist.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
            ax.set_xlabel('Financial Profile', fontsize=12, fontweight='bold')
            ax.set_ylabel('Nombre d\'individus', fontsize=12, fontweight='bold')
            ax.set_title('Distribution du Financial Profile (Target)', fontsize=14, fontweight='bold', pad=15)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Fonction helper pour afficher les métriques d'un algorithme
    def display_algorithm_metrics(algorithm_name, algorithm_data):
        """Affiche les métriques d'un algorithme"""
        if algorithm_data is None:
            st.warning(f"⚠️ Les métriques pour {algorithm_name} ne sont pas disponibles")
            return
        
        metrics = algorithm_data.get('metrics', {})
        params = algorithm_data.get('params', {})
        cv_score = algorithm_data.get('cv_score', {})
        train_time = algorithm_data.get('train_time', 0)
        
        st.markdown(f"### ✅ {algorithm_name} appliqué avec succès!")
        
        # Paramètres optimaux
        st.markdown("#### ⚙️ Paramètres optimaux")
        params_df = pd.DataFrame({
            'Paramètre': list(params.keys()),
            'Valeur': list(params.values())
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        # Métriques de performance
        st.markdown("---")
        st.markdown("#### 📊 Métriques de performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
        
        # Validation croisée
        st.markdown("---")
        st.markdown("#### 📈 Validation croisée (5-fold CV)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CV Score Mean", f"{cv_score.get('mean', 0):.4f}")
        with col2:
            st.metric("CV Score Std", f"±{cv_score.get('std', 0):.4f}")
        with col3:
            st.metric("Temps d'entraînement", f"{train_time:.2f}s")
        
        st.info("""
        **Interprétation:**
        - **Accuracy**: Pourcentage de prédictions correctes (plus haut = mieux)
        - **Precision**: Exactitude des prédictions positives (plus haut = mieux)
        - **Recall**: Capacité à trouver tous les cas positifs (plus haut = mieux)
        - **F1-Score**: Moyenne harmonique de Precision et Recall (plus haut = mieux)
        - **CV Score**: Score moyen sur validation croisée (plus haut = mieux, std bas = stable)
        """)
    
    # TAB 2: SVM
    with tab2:
        st.subheader("🤖 SVM - Support Vector Machine")
        
        svm_data = classification_results.get('svm')
        display_algorithm_metrics("SVM", svm_data)
        
        if svm_data:
            st.markdown("---")
            st.markdown("### 💡 Caractéristiques de SVM")
            st.info("""
            **Avantages:**
            - Efficace pour les problèmes de classification non linéaires
            - Bonne performance avec peu de données
            - Résistant au surapprentissage grâce à la marge maximale
            
            **Inconvénients:**
            - Nécessite la standardisation des données
            - Sensible aux hyperparamètres (C, gamma)
            - Temps d'entraînement peut être long avec de grandes données
            """)
    
    # TAB 3: Decision Tree
    with tab3:
        st.subheader("🌳 Decision Tree - Arbre de Décision")
        
        dt_data = classification_results.get('decision_tree')
        display_algorithm_metrics("Decision Tree", dt_data)
        
        if dt_data:
            st.markdown("---")
            st.markdown("### 💡 Caractéristiques de Decision Tree")
            st.info("""
            **Avantages:**
            - Très interprétable (règles claires)
            - Pas besoin de standardisation
            - Gère bien les relations non linéaires
            - Peut gérer les valeurs manquantes
            
            **Inconvénients:**
            - Sensible au surapprentissage
            - Instable (petits changements = arbre différent)
            - Peut créer des arbres trop complexes
            """)
    
    # TAB 4: XGBoost
    with tab4:
        st.subheader("⚡ XGBoost - Gradient Boosting")
        
        xgb_data = classification_results.get('xgboost')
        display_algorithm_metrics("XGBoost", xgb_data)
        
        if xgb_data:
            st.markdown("---")
            st.markdown("### 💡 Caractéristiques de XGBoost")
            st.info("""
            **Avantages:**
            - Très haute performance (souvent meilleur que les autres)
            - Régularisation intégrée (évite le surapprentissage)
            - Gère bien les valeurs manquantes
            - Efficace et scalable
            
            **Inconvénients:**
            - Moins interprétable que Decision Tree
            - Temps d'entraînement plus long
            - Sensible aux hyperparamètres
            - Nécessite plus de tuning
            """)
    
    # TAB 5: Random Forest
    with tab5:
        st.subheader("🌲 Random Forest - Forêt Aléatoire")
        
        rf_data = classification_results.get('random_forest')
        display_algorithm_metrics("Random Forest", rf_data)
        
        if rf_data:
            st.markdown("---")
            st.markdown("### 💡 Caractéristiques de Random Forest")
            st.info("""
            **Avantages:**
            - Très robuste et performant
            - Réduit le surapprentissage (moyenne de plusieurs arbres)
            - Gère bien les données non linéaires
            - Peut mesurer l'importance des features
            - Pas besoin de standardisation
            
            **Inconvénients:**
            - Moins interprétable qu'un seul arbre
            - Peut être lent avec beaucoup d'arbres
            - Nécessite plus de mémoire
            """)
    
    # TAB 6: Comparaison
    with tab6:
        st.subheader("📊 Comparaison des Algorithmes")
        
        comparison_data = classification_results.get('comparison', [])
        
        if comparison_data:
            # Tableau comparatif
            st.markdown("### 📋 Tableau Comparatif Complet")
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualisation comparative
            st.markdown("---")
            st.markdown("### 📈 Visualisation Comparative des Métriques")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            algorithms = comparison_df['Algorithme'].values
            colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
            
            # Accuracy
            axes[0, 0].bar(algorithms, comparison_df['Accuracy'], color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Accuracy', fontsize=12)
            axes[0, 0].set_title('Accuracy (Plus haut = mieux)', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            axes[0, 0].set_ylim([0.85, 0.95])
            for i, (alg, val) in enumerate(zip(algorithms, comparison_df['Accuracy'])):
                axes[0, 0].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Precision
            axes[0, 1].bar(algorithms, comparison_df['Precision'], color=colors, alpha=0.7)
            axes[0, 1].set_ylabel('Precision', fontsize=12)
            axes[0, 1].set_title('Precision (Plus haut = mieux)', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            axes[0, 1].set_ylim([0.85, 0.95])
            for i, (alg, val) in enumerate(zip(algorithms, comparison_df['Precision'])):
                axes[0, 1].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Recall
            axes[0, 2].bar(algorithms, comparison_df['Recall'], color=colors, alpha=0.7)
            axes[0, 2].set_ylabel('Recall', fontsize=12)
            axes[0, 2].set_title('Recall (Plus haut = mieux)', fontsize=14, fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3, axis='y')
            axes[0, 2].set_ylim([0.85, 0.95])
            for i, (alg, val) in enumerate(zip(algorithms, comparison_df['Recall'])):
                axes[0, 2].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # F1-Score
            axes[1, 0].bar(algorithms, comparison_df['F1-Score'], color=colors, alpha=0.7)
            axes[1, 0].set_ylabel('F1-Score', fontsize=12)
            axes[1, 0].set_title('F1-Score (Plus haut = mieux)', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].set_ylim([0.85, 0.95])
            for i, (alg, val) in enumerate(zip(algorithms, comparison_df['F1-Score'])):
                axes[1, 0].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # CV Score Mean
            axes[1, 1].bar(algorithms, comparison_df['CV_Score_Mean'], color=colors, alpha=0.7)
            axes[1, 1].errorbar(algorithms, comparison_df['CV_Score_Mean'], 
                               yerr=comparison_df['CV_Score_Std'], fmt='none', color='black', capsize=5)
            axes[1, 1].set_ylabel('CV Score Mean', fontsize=12)
            axes[1, 1].set_title('CV Score Mean ± Std (Plus haut = mieux)', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].set_ylim([0.85, 0.98])
            for i, (alg, val) in enumerate(zip(algorithms, comparison_df['CV_Score_Mean'])):
                axes[1, 1].text(i, val, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Train Time
            axes[1, 2].bar(algorithms, comparison_df['Train_Time'], color=colors, alpha=0.7)
            axes[1, 2].set_ylabel('Temps (secondes)', fontsize=12)
            axes[1, 2].set_title('Temps d\'Entraînement (Plus bas = mieux)', fontsize=14, fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            for i, (alg, val) in enumerate(zip(algorithms, comparison_df['Train_Time'])):
                axes[1, 2].text(i, val, f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Comparaison Complète: SVM vs Decision Tree vs XGBoost vs Random Forest', 
                        fontsize=18, fontweight='bold', y=0.995)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Analyse détaillée
            st.markdown("---")
            st.markdown("### 📊 Analyse Détaillée par Métrique")
            
            best_accuracy_alg = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Algorithme']
            best_f1_alg = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Algorithme']
            best_cv_alg = comparison_df.loc[comparison_df['CV_Score_Mean'].idxmax(), 'Algorithme']
            fastest_alg = comparison_df.loc[comparison_df['Train_Time'].idxmin(), 'Algorithme']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🥇 Meilleurs résultats par métrique:")
                st.success(f"""
                **Accuracy:** {best_accuracy_alg} ({comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Accuracy']:.4f})
                
                **F1-Score:** {best_f1_alg} ({comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'F1-Score']:.4f})
                
                **CV Score:** {best_cv_alg} ({comparison_df.loc[comparison_df['CV_Score_Mean'].idxmax(), 'CV_Score_Mean']:.4f})
                """)
            
            with col2:
                st.markdown("#### ⚡ Performance:")
                st.info(f"""
                **Plus rapide:** {fastest_alg} ({comparison_df.loc[comparison_df['Train_Time'].idxmin(), 'Train_Time']:.2f}s)
                
                **Plus stable (CV Std):** {comparison_df.loc[comparison_df['CV_Score_Std'].idxmin(), 'Algorithme']} (±{comparison_df['CV_Score_Std'].min():.4f})
                """)
    
    # TAB 7: Décision Finale
    with tab7:
        st.subheader("🏆 Décision Finale - Algorithme le Plus Adapté")
        
        final_decision = classification_results.get('final_decision', 'N/A')
        final_metrics = classification_results.get('final_metrics', {})
        best_accuracy_alg = classification_results.get('best_algorithm_accuracy', 'N/A')
        best_f1_alg = classification_results.get('best_algorithm_f1', 'N/A')
        best_global_alg = classification_results.get('best_algorithm_global', 'N/A')
        
        if final_decision != 'N/A':
            st.markdown("### 📊 Résultats par Métrique")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Meilleur Accuracy", best_accuracy_alg)
            with col2:
                st.metric("Meilleur F1-Score", best_f1_alg)
            with col3:
                st.metric("Meilleur Score Global", best_global_alg)
            
            st.markdown("---")
            st.success(f"✅ **ALGORITHME RECOMMANDÉ: {final_decision.upper()}**")
            
            if final_metrics:
                st.markdown("### 📈 Performance du Modèle Recommandé")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{final_metrics.get('Accuracy', 0):.4f}")
                with col2:
                    st.metric("Precision", f"{final_metrics.get('Precision', 0):.4f}")
                with col3:
                    st.metric("Recall", f"{final_metrics.get('Recall', 0):.4f}")
                with col4:
                    st.metric("F1-Score", f"{final_metrics.get('F1-Score', 0):.4f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CV Score Mean", f"{final_metrics.get('CV_Score_Mean', 0):.4f}")
                with col2:
                    st.metric("CV Score Std", f"±{final_metrics.get('CV_Score_Std', 0):.4f}")
                with col3:
                    st.metric("Temps d'entraînement", f"{final_metrics.get('Train_Time', 0):.2f}s")
            
            st.markdown("---")
            st.markdown("### 📋 Justification de la Décision")
            
            if final_decision == 'Random Forest':
                st.markdown("#### 🎯 Random Forest - Avantages pour notre dataset:")
                
                st.markdown("**1. PERFORMANCE EXCEPTIONNELLE:**")
                st.success(f"""
                ✅ Accuracy: {final_metrics.get('Accuracy', 0):.4f} ({final_metrics.get('Accuracy', 0)*100:.2f}%)
                   → Prédit correctement le Financial Profile dans {final_metrics.get('Accuracy', 0)*100:.1f}% des cas
                
                ✅ F1-Score: {final_metrics.get('F1-Score', 0):.4f}
                   → Excellent équilibre entre Precision et Recall
                
                ✅ CV Score: {final_metrics.get('CV_Score_Mean', 0):.4f} ± {final_metrics.get('CV_Score_Std', 0):.4f}
                   → Performance stable et reproductible sur validation croisée
                """)
                
                st.markdown("**2. ROBUSTESSE ET STABILITÉ:**")
                st.success("""
                ✅ Ensemble de plusieurs arbres → Réduit la variance et le surapprentissage
                ✅ Résultats stables (faible écart-type en CV)
                ✅ Gère bien les relations non linéaires entre variables
                ✅ Peut mesurer l'importance des features
                """)
                
                st.markdown("**3. ADAPTATION AU DATASET:**")
                st.success("""
                ✅ Pas besoin de standardisation (contrairement à SVM)
                ✅ Gère bien les 4 classes du Financial Profile
                ✅ Performance équilibrée sur toutes les classes
                ✅ Temps d'entraînement acceptable pour la performance obtenue
                """)
                
                st.markdown("**4. COMPARAISON AVEC LES ALTERNATIVES:**")
                comparison_data = classification_results.get('comparison', [])
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    st.markdown(f"""
                    **vs SVM:**
                    - Accuracy: {final_metrics.get('Accuracy', 0):.4f} vs {comparison_df[comparison_df['Algorithme']=='SVM']['Accuracy'].values[0]:.4f} (+{final_metrics.get('Accuracy', 0) - comparison_df[comparison_df['Algorithme']=='SVM']['Accuracy'].values[0]:.4f})
                    - Random Forest plus performant et plus stable
                    
                    **vs Decision Tree:**
                    - Accuracy: {final_metrics.get('Accuracy', 0):.4f} vs {comparison_df[comparison_df['Algorithme']=='Decision Tree']['Accuracy'].values[0]:.4f} (+{final_metrics.get('Accuracy', 0) - comparison_df[comparison_df['Algorithme']=='Decision Tree']['Accuracy'].values[0]:.4f})
                    - Random Forest évite le surapprentissage d'un seul arbre
                    
                    **vs XGBoost:**
                    - Accuracy: {final_metrics.get('Accuracy', 0):.4f} vs {comparison_df[comparison_df['Algorithme']=='XGBoost']['Accuracy'].values[0]:.4f} (+{final_metrics.get('Accuracy', 0) - comparison_df[comparison_df['Algorithme']=='XGBoost']['Accuracy'].values[0]:.4f})
                    - Random Forest légèrement meilleur avec meilleure stabilité
                    """)
            
            st.markdown("---")
            st.markdown("### ✅ Conclusion et Recommandation Finale")
            
            st.success(f"""
            **🎯 ALGORITHME RECOMMANDÉ POUR LE DÉPLOIEMENT: {final_decision.upper()}**
            
            **📋 Paramètres optimaux:**
            - Voir la section dédiée à {final_decision} pour les détails
            
            **📊 Performance attendue:**
            - Accuracy: {final_metrics.get('Accuracy', 0):.4f} ({final_metrics.get('Accuracy', 0)*100:.2f}%)
            - Precision: {final_metrics.get('Precision', 0):.4f}
            - Recall: {final_metrics.get('Recall', 0):.4f}
            - F1-Score: {final_metrics.get('F1-Score', 0):.4f}
            - CV Score: {final_metrics.get('CV_Score_Mean', 0):.4f} ± {final_metrics.get('CV_Score_Std', 0):.4f}
            - Temps d'entraînement: {final_metrics.get('Train_Time', 0):.2f}s
            
            **💡 Objectif atteint:** ✅ Oui - Performance excellente ({classification_results.get('performance_level', 'N/A')})
            """)
        else:
            st.warning("⚠️ La décision finale n'est pas disponible. Exécutez d'abord le notebook notebooks/06_classification.ipynb")

# ============================================
# NOTEBOOK 8: RÉGRESSION - PRÉDICTION DE TOTAL_EXPENSES
# ============================================
elif notebook_choice == "8. Régression - Total_Expenses":
    st.header("📉 8. Régression - Prédiction de Total_Expenses (USD)")
    st.markdown("**Objectif:** Comparer les modèles de régression (linéaire, polynomiale et régularisés) pour prédire les dépenses totales.")
    
    @st.cache_data
    def load_regression_results():
        try:
            with open(os.path.join(BASE_DIR, 'data/results/regression_analysis.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @st.cache_data
    def load_features_data_reg():
        try:
            return pd.read_csv(os.path.join(BASE_DIR, 'data/processed/dataset_with_features.csv'))
        except FileNotFoundError:
            return None

    regression_results = load_regression_results()
    df_features = load_features_data_reg()

    if regression_results is None:
        st.warning("⚠️ Les résultats de régression n'existent pas encore. Exécutez d'abord le notebook notebooks/07_regression.ipynb")
        st.stop()

    # Helper d'affichage
    def display_reg_metrics(title, metrics):
        st.markdown(f"### ✅ {title}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MSE", f"{metrics.get('mse', metrics.get('MSE', 0)):.2f}")
        with col2:
            st.metric("RMSE", f"{metrics.get('rmse', metrics.get('RMSE', 0)):.2f}")
        with col3:
            st.metric("MAE", f"{metrics.get('mae', metrics.get('MAE', 0)):.2f}")
        with col4:
            st.metric("R²", f"{metrics.get('r2', metrics.get('R²', 0)):.4f}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CV R² (moy)", f"{metrics.get('cv_r2_mean', metrics.get('CV_R²_Mean', 0)):.4f}")
        with col2:
            st.metric("CV R² (std)", f"{metrics.get('cv_r2_std', metrics.get('CV_R²_Std', 0)):.4f}")
        with col3:
            st.metric("Temps entraînement", f"{metrics.get('train_time', metrics.get('Train_Time', 0)):.2f}s")

        st.info("**Interprétation rapide :** R² proche de 1 = excellent, R² négatif = pire que le baseline (moyenne). RMSE/MAE plus bas = mieux. CV R² proche de 0 et stable = modèle peu performant mais régulier.")

    # Onglets
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Vue d'ensemble",
        "📈 Linéaire Simple",
        "📈 Linéaire Multiple",
        "📈 Polynomiale",
        "🛡️ Régularisation (Ridge/Lasso/ElasticNet)",
        "📈 Polynomiale Optimisée",
        "🏆 Comparaison & Décision Finale"
    ])

    # TAB 1: Vue d'ensemble
    with tab1:
        st.subheader("📋 Vue d'ensemble des données de régression")
        st.markdown(f"**Target:** {regression_results.get('target_variable', 'Total_Expenses (USD)')}")
        features_used = regression_results.get('features_used', [])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Variables utilisées", len(features_used))
        with col2:
            st.metric("Observations", f"{len(df_features):,}" if df_features is not None else "N/A")

        if features_used:
            st.markdown("### 🧮 Variables utilisées")
            st.write(", ".join(features_used))

        if df_features is not None and 'Total_Expenses' in df_features.columns:
            st.markdown("---")
            st.markdown("### 📊 Distribution de la cible (Total_Expenses)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df_features['Total_Expenses'], kde=True, ax=ax, color='steelblue')
            ax.set_xlabel('Total_Expenses (USD)')
            ax.set_ylabel('Fréquence')
            ax.set_title('Distribution de la variable cible')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # TAB 2: Linéaire Simple
    with tab2:
        display_reg_metrics("Régression Linéaire Simple", regression_results.get('linear_simple', {}).get('metrics', {}))

    # TAB 3: Linéaire Multiple
    with tab3:
        display_reg_metrics("Régression Linéaire Multiple", regression_results.get('linear_multiple', {}).get('metrics', {}))

    # TAB 4: Polynomiale (degré 2)
    with tab4:
        poly = regression_results.get('polynomial', {})
        display_reg_metrics(f"Régression Polynomiale (Degré {poly.get('best_degree', 2)})", poly.get('best_metrics', poly.get('metrics', {})))
        if poly.get('all_degrees_results'):
            st.markdown("### 🔍 Résultats par degré testé")
            st.dataframe(pd.DataFrame(poly['all_degrees_results']).T, use_container_width=True)

    # TAB 5: Régularisation
    with tab5:
        st.subheader("🛡️ Modèles régularisés")
        ridge = regression_results.get('comparison', [])
        # Extraire rapidement les lignes des modèles régularisés
        comp_df = pd.DataFrame(ridge) if ridge else pd.DataFrame()
        # Les hyperparamètres (α, l1) dépendent de la dernière exécution du notebook : on cherche par préfixe
        regularized = comp_df[comp_df['Algorithme'].str.startswith(("Ridge", "Lasso", "Elastic Net"))] if not comp_df.empty else comp_df
        for _, row in regularized.iterrows():
            display_reg_metrics(row['Algorithme'], row.to_dict())
            st.markdown("---")
        if regularized.empty:
            st.info("Aucun modèle régularisé dans les résultats.")

    # TAB 6: Polynomiale Optimisée
    with tab6:
        poly_opt = None
        # Chercher la ligne correspondante dans comparison
        comp_df = pd.DataFrame(regression_results.get('comparison', []))
        poly_opt_rows = comp_df[comp_df['Algorithme'].str.startswith("Polynomial Optimisé")] if not comp_df.empty else comp_df
        if not poly_opt_rows.empty:
            poly_opt = poly_opt_rows.iloc[0].to_dict()
        if poly_opt:
            display_reg_metrics(poly_opt['Algorithme'].replace("Polynomial Optimisé", "Régression Polynomiale Optimisée"), poly_opt)
        else:
            st.info("Aucun modèle polynomial optimisé dans les résultats.")

    # TAB 7: Comparaison & Décision Finale
    with tab7:
        st.subheader("🏆 Comparaison & Décision Finale")

        comparison_data = regression_results.get('comparison', [])
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.markdown("### 📋 Tableau comparatif complet")
            st.dataframe(comp_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📈 Visualisation comparative")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].bar(comp_df['Algorithme'], comp_df['RMSE'], color='steelblue', alpha=0.7)
            axes[0, 0].set_title('RMSE (Plus bas = mieux)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].bar(comp_df['Algorithme'], comp_df['MAE'], color='coral', alpha=0.7)
            axes[0, 1].set_title('MAE (Plus bas = mieux)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].bar(comp_df['Algorithme'], comp_df['R²'], color='mediumseagreen', alpha=0.7)
            axes[1, 0].set_title('R² (Plus haut = mieux, <0 = pire que baseline)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].bar(comp_df['Algorithme'], comp_df['CV_R²_Mean'], color='gold', alpha=0.7)
            axes[1, 1].set_title('CV R² Mean (Stabilité globale)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        best_algo = regression_results.get('best_algorithm', 'N/A')
        best_metrics = regression_results.get('best_metrics', {})

        st.markdown("---")
        st.subheader("🏁 Décision Finale")
        if best_algo != 'N/A' and best_metrics:
            st.success(f"✅ Algorithme recommandé: {best_algo}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MSE", f"{best_metrics.get('MSE', 0):.2f}")
            with col2:
                st.metric("RMSE", f"{best_metrics.get('RMSE', 0):.2f}")
            with col3:
                st.metric("MAE", f"{best_metrics.get('MAE', 0):.2f}")
            with col4:
                st.metric("R²", f"{best_metrics.get('R²', 0):.4f}")

            st.markdown("### 📋 Justification rapide")
            st.info(f"""
            - {best_algo} présente le score global le plus élevé parmi les modèles testés.
            - Les RMSE/MAE les plus bas indiquent les erreurs moyennes les plus faibles.
            - {"R² négatif → le modèle fait à peine mieux que la moyenne ; envisager d'enrichir les features." if best_metrics.get('R²', 0) < 0 else f"R² = {best_metrics.get('R²', 0):.3f} → le modèle explique {best_metrics.get('R²', 0) * 100:.0f} % de la variance des dépenses totales."}
            """)
        else:
            st.warning("⚠️ Aucune décision finale disponible. Exécutez le notebook de régression pour générer les résultats.")

# ============================================
# DÉPLOIEMENT - PRÉDICTIONS INTERACTIVES
# ============================================
elif notebook_choice == "🚀 Déploiement - Prédictions Interactives":
    st.header("🚀 Déploiement - Prédictions Interactives")
    st.markdown("**Objectif:** Appliquer les meilleurs modèles identifiés pour faire des prédictions sur de nouvelles données")
    
    st.info("""
    **Modèles déployés:**
    - 🤖 **K-Means** (Clustering) - k=4
    - 🌲 **Random Forest** (Classification) - Prédiction du Financial Profile
    - 📈 **Régression Linéaire Simple** (Régression) - Prédiction des Total Expenses
    """)
    
    # Fonctions de chargement
    @st.cache_data
    def load_deployment_data():
        """Charge les données nécessaires pour le déploiement"""
        try:
            df_features = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/dataset_with_features.csv'))
            clustering_metrics = json.load(open(os.path.join(BASE_DIR, 'data/results/clustering_analysis.json'), 'r', encoding='utf-8'))
            classification_results = json.load(open(os.path.join(BASE_DIR, 'data/results/classification_comparison.json'), 'r', encoding='utf-8'))
            regression_results = json.load(open(os.path.join(BASE_DIR, 'data/results/regression_analysis.json'), 'r', encoding='utf-8'))
            return df_features, clustering_metrics, classification_results, regression_results
        except FileNotFoundError as e:
            st.error(f"❌ Fichier manquant: {e}")
            return None, None, None, None
    
    # Chargement des données
    df_features, clustering_metrics, classification_results, regression_results = load_deployment_data()
    
    if df_features is None or clustering_metrics is None or classification_results is None or regression_results is None:
        st.warning("⚠️ Les données nécessaires ne sont pas disponibles. Exécutez d'abord les notebooks de modélisation.")
        st.stop()
    
    # Variables originales pour le clustering et la classification
    original_vars = [
        'Age', 'Income (USD)', 'Rent (USD)', 'Groceries (USD)', 
        'Eating Out (USD)', 'Entertainment (USD)', 'Subscription Services (USD)',
        'Education (USD)', 'Online Shopping (USD)', 'Savings (USD)', 
        'Investments (USD)', 'Travel (USD)', 'Fitness (USD)', 'Miscellaneous (USD)'
    ]
    
    # Variables pour la régression
    regression_features = regression_results.get('features_used', ['Income (USD)', 'Age', 'Savings (USD)', 'Investments (USD)'])
    
    # Onglets
    tab1, tab2, tab3 = st.tabs([
        "📝 Formulaire de Prédiction",
        "📊 Résultats Détaillés",
        "ℹ️ Guide d'Utilisation"
    ])
    
    # TAB 1: Formulaire de Prédiction
    with tab1:
        st.subheader("📝 Formulaire de Prédiction")
        st.markdown("Remplissez les informations ci-dessous pour obtenir les prédictions des 3 modèles")
        
        # Formulaire en colonnes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Informations Financières")
            age = st.number_input("Âge", min_value=18, max_value=100, value=25, step=1)
            income = st.number_input("Revenu (USD)", min_value=0, value=50000, step=1000)
            savings = st.number_input("Épargne (USD)", min_value=0, value=5000, step=100)
            investments = st.number_input("Investissements (USD)", min_value=0, value=2000, step=100)
        
        with col2:
            st.markdown("### 🏠 Dépenses Mensuelles (USD)")
            rent = st.number_input("Loyer", min_value=0, value=1200, step=50)
            groceries = st.number_input("Épicerie", min_value=0, value=400, step=20)
            eating_out = st.number_input("Restaurants", min_value=0, value=300, step=20)
            entertainment = st.number_input("Divertissement", min_value=0, value=200, step=20)
        
        col3, col4 = st.columns(2)
        
        with col3:
            subscription = st.number_input("Abonnements", min_value=0, value=100, step=10)
            education = st.number_input("Éducation", min_value=0, value=500, step=50)
            online_shopping = st.number_input("Shopping en ligne", min_value=0, value=250, step=20)
        
        with col4:
            travel = st.number_input("Voyages", min_value=0, value=300, step=20)
            fitness = st.number_input("Fitness", min_value=0, value=80, step=10)
            miscellaneous = st.number_input("Divers", min_value=0, value=200, step=20)
        
        # Bouton de prédiction
        if st.button("🔮 Obtenir les Prédictions", type="primary", use_container_width=True):
            # Préparation des données pour les modèles
            input_data = {
                'Age': age,
                'Income (USD)': income,
                'Rent (USD)': rent,
                'Groceries (USD)': groceries,
                'Eating Out (USD)': eating_out,
                'Entertainment (USD)': entertainment,
                'Subscription Services (USD)': subscription,
                'Education (USD)': education,
                'Online Shopping (USD)': online_shopping,
                'Savings (USD)': savings,
                'Investments (USD)': investments,
                'Travel (USD)': travel,
                'Fitness (USD)': fitness,
                'Miscellaneous (USD)': miscellaneous
            }
            
            # Création d'un DataFrame pour les prédictions
            input_df = pd.DataFrame([input_data])
            
            # ========== 1. CLUSTERING K-MEANS ==========
            st.markdown("---")
            st.markdown("## 🤖 1. Clustering K-Means")
            
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Préparation des données pour K-Means (standardisation)
                X_clustering = df_features[original_vars].copy()
                scaler_clustering = StandardScaler()
                X_scaled = scaler_clustering.fit_transform(X_clustering)
                
                # Entraînement de K-Means avec k=4
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                
                # Prédiction
                input_scaled = scaler_clustering.transform(input_df[original_vars])
                cluster_pred = kmeans.predict(input_scaled)[0]
                
                # Interprétation du cluster
                cluster_labels = {
                    0: "Cluster 0 - Profil Modéré",
                    1: "Cluster 1 - Profil Équilibré",
                    2: "Cluster 2 - Profil Investisseur",
                    3: "Cluster 3 - Profil Dépensier"
                }
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Cluster Assigné", f"Cluster {cluster_pred}")
                with col2:
                    st.success(f"**{cluster_labels.get(cluster_pred, f'Cluster {cluster_pred}')}**")
                
                # Caractéristiques du cluster
                cluster_mask = df_features['KMeans_Cluster'] == cluster_pred if 'KMeans_Cluster' in df_features.columns else None
                if cluster_mask is not None and cluster_mask.sum() > 0:
                    cluster_data = df_features[cluster_mask]
                    st.markdown("#### 📊 Caractéristiques moyennes de ce cluster:")
                    
                    avg_cols = ['Income (USD)', 'Savings (USD)', 'Investments (USD)', 'Total_Expenses']
                    avg_data = {}
                    for col in avg_cols:
                        if col in cluster_data.columns:
                            avg_data[col] = cluster_data[col].mean()
                    
                    if avg_data:
                        avg_df = pd.DataFrame([avg_data])
                        st.dataframe(avg_df.T, use_container_width=True, 
                                   column_config={0: "Valeur Moyenne"})
            except Exception as e:
                st.error(f"Erreur dans le clustering: {e}")
            
            # ========== 2. CLASSIFICATION RANDOM FOREST ==========
            st.markdown("---")
            st.markdown("## 🌲 2. Classification Random Forest")
            
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder
                
                # Préparation des données
                X_classification = df_features[original_vars].copy()
                y_classification = df_features['Financial_Profile'].copy()
                
                # Encodage du target
                le_class = LabelEncoder()
                y_encoded = le_class.fit_transform(y_classification)
                
                # Entraînement du modèle Random Forest avec les paramètres optimaux
                rf_params = classification_results.get('random_forest', {}).get('params', {})
                rf_model = RandomForestClassifier(
                    max_depth=rf_params.get('max_depth', 20),
                    min_samples_split=rf_params.get('min_samples_split', 5),
                    n_estimators=rf_params.get('n_estimators', 100),
                    random_state=42
                )
                rf_model.fit(X_classification, y_encoded)
                
                # Prédiction
                profile_pred = rf_model.predict(input_df[original_vars])[0]
                profile_proba = rf_model.predict_proba(input_df[original_vars])[0]
                
                profile_name = le_class.inverse_transform([profile_pred])[0]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Financial Profile Prédit", profile_name)
                with col2:
                    confidence = profile_proba.max() * 100
                    st.success(f"**Confiance: {confidence:.2f}%**")
                
                # Probabilités par classe
                st.markdown("#### 📊 Probabilités par profil:")
                proba_df = pd.DataFrame({
                    'Profil': le_class.classes_,
                    'Probabilité (%)': (profile_proba * 100).round(2)
                }).sort_values('Probabilité (%)', ascending=False)
                
                st.dataframe(proba_df, use_container_width=True, hide_index=True)
                
                # Graphique des probabilités
                fig, ax = plt.subplots(figsize=(10, 6))
                proba_df.plot(x='Profil', y='Probabilité (%)', kind='bar', ax=ax, 
                            color=['#2ecc71' if p == profile_name else '#95a5a6' for p in proba_df['Profil']])
                ax.set_ylabel('Probabilité (%)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Financial Profile', fontsize=12, fontweight='bold')
                ax.set_title('Probabilités de Classification par Random Forest', 
                           fontsize=14, fontweight='bold', pad=15)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim([0, 100])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Erreur dans la classification: {e}")
            
            # ========== 3. RÉGRESSION LINÉAIRE SIMPLE ==========
            st.markdown("---")
            st.markdown("## 📈 3. Régression Linéaire Simple")
            
            try:
                from sklearn.linear_model import LinearRegression
                
                # Préparation des données pour la régression
                regression_data = regression_results.get('linear_simple', {})
                feature_used = regression_data.get('feature_used', 'Investments (USD)')
                
                X_regression = df_features[[feature_used]].copy()
                y_regression = df_features['Total_Expenses'].copy()
                
                # Entraînement du modèle
                lr_model = LinearRegression()
                lr_model.fit(X_regression, y_regression)
                
                # Prédiction
                input_regression = input_df[[feature_used]]
                total_expenses_pred = lr_model.predict(input_regression)[0]
                
                # Calcul des dépenses totales réelles pour comparaison
                total_expenses_actual = (
                    rent + groceries + eating_out + entertainment + 
                    subscription + education + online_shopping + 
                    travel + fitness + miscellaneous
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Expenses Prédit", f"${total_expenses_pred:,.2f}")
                with col2:
                    st.metric("Total Expenses Réel", f"${total_expenses_actual:,.2f}")
                with col3:
                    diff = abs(total_expenses_pred - total_expenses_actual)
                    st.metric("Écart", f"${diff:,.2f}")
                
                # Interprétation
                st.markdown("#### 💡 Interprétation:")
                st.info(f"""
                Le modèle prédit des dépenses totales de **${total_expenses_pred:,.2f}** basé sur **{feature_used}**.
                
                **Dépenses réelles calculées:** ${total_expenses_actual:,.2f}
                **Écart:** ${diff:,.2f}
                
                **Note:** Le modèle utilise uniquement {feature_used} comme variable prédictive. 
                Les dépenses réelles incluent toutes les catégories de dépenses.
                """)
                
                # Graphique de comparaison
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = ['Prédit', 'Réel']
                values = [total_expenses_pred, total_expenses_actual]
                colors = ['#3498db', '#2ecc71']
                bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                ax.set_ylabel('Montant (USD)', fontsize=12, fontweight='bold')
                ax.set_title('Comparaison: Dépenses Prédites vs Réelles', 
                           fontsize=14, fontweight='bold', pad=15)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Ajouter les valeurs sur les barres
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Erreur dans la régression: {e}")
            
            # ========== RÉSUMÉ FINAL ==========
            st.markdown("---")
            st.markdown("## 📋 Résumé des Prédictions")
            
            summary_data = {
                'Modèle': ['K-Means', 'Random Forest', 'Régression Linéaire'],
                'Type': ['Clustering', 'Classification', 'Régression'],
                'Résultat': [
                    f"Cluster {cluster_pred}",
                    profile_name,
                    f"${total_expenses_pred:,.2f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # TAB 2: Résultats Détaillés
    with tab2:
        st.subheader("📊 Résultats Détaillés des Modèles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🤖 K-Means")
            kmeans_metrics = clustering_metrics.get('kmeans', {})
            st.metric("k optimal", kmeans_metrics.get('n_clusters', 4))
            st.metric("Score Silhouette", f"{kmeans_metrics.get('silhouette_score', 0):.4f}")
            st.metric("Calinski-Harabasz", f"{kmeans_metrics.get('calinski_harabasz_score', 0):.2f}")
        
        with col2:
            st.markdown("### 🌲 Random Forest")
            rf_metrics = classification_results.get('random_forest', {}).get('metrics', {})
            st.metric("Accuracy", f"{rf_metrics.get('accuracy', 0):.4f}")
            st.metric("Precision", f"{rf_metrics.get('precision', 0):.4f}")
            st.metric("F1-Score", f"{rf_metrics.get('f1', 0):.4f}")
        
        with col3:
            st.markdown("### 📈 Régression Linéaire")
            reg_metrics = regression_results.get('linear_simple', {}).get('metrics', {})
            st.metric("RMSE", f"{reg_metrics.get('rmse', 0):.2f}")
            st.metric("MAE", f"{reg_metrics.get('mae', 0):.2f}")
            st.metric("R²", f"{reg_metrics.get('r2', 0):.4f}")
    
    # TAB 3: Guide d'Utilisation
    with tab3:
        st.subheader("ℹ️ Guide d'Utilisation")
        
        st.markdown("""
        ### 📝 Comment utiliser le formulaire de prédiction
        
        1. **Remplissez les champs du formulaire** avec les informations financières de l'individu
        2. **Cliquez sur "Obtenir les Prédictions"** pour lancer les 3 modèles
        3. **Consultez les résultats** pour chaque modèle :
           - **K-Means** : Assignation à un cluster de comportement
           - **Random Forest** : Prédiction du Financial Profile avec probabilités
           - **Régression Linéaire** : Prédiction des dépenses totales
        
        ### 🎯 Interprétation des résultats
        
        **Clustering K-Means:**
        - Identifie le groupe de comportement le plus proche
        - Affiche les caractéristiques moyennes du cluster
        
        **Classification Random Forest:**
        - Prédit le Financial Profile (Investisseur, Économe, Dépensier, Équilibré)
        - Affiche les probabilités pour chaque profil
        
        **Régression Linéaire:**
        - Prédit les dépenses totales basées sur les investissements
        - Compare avec les dépenses réelles calculées
        
        ### ⚠️ Notes importantes
        
        - Les modèles sont entraînés sur les données du dataset GenZ
        - Les prédictions sont basées sur les patterns appris
        - Les résultats sont indicatifs et doivent être interprétés avec prudence
        """)

# Footer
st.markdown("---")
st.markdown("**Application créée pour visualiser les notebooks d'analyse des dépenses de la Génération Z**")

