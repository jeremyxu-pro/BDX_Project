import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import requests
import zipfile
import io
import plotly.express as px

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, learning_curve, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib
import os

# Ajouter un style CSS pour personnaliser les couleurs
def add_custom_styles():
    st.markdown(
        """
        <style>
        /* Couleur pour la barre latérale */
        section[data-testid="stSidebar"] {
            background-color: #A8E6A3; /* Vert clair */
            color: black; /* Texte noir */
        }
        /* Alignement des éléments */
        .sidebar-content {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        /* Couleur des titres principaux */
        h1, h2, h3 {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Appliquer le style personnalisé
add_custom_styles()


 
# URLs des logos hébergés sur GitHub
logo_isup_url = "https://raw.githubusercontent.com/jeremyxu-pro/BDX_Project/main/DataViz/Logo-ISUP.jpg"
logo_gov_url = "https://raw.githubusercontent.com/jeremyxu-pro/BDX_Project/main/DataViz/gov.png"
 
 
try:
    # Charger le logo gov.br en haut, centré
    st.sidebar.markdown("<div style='text-align: center;'><img src='{}' width='100'></div>".format(logo_gov_url), unsafe_allow_html=True)

    # Ajouter le texte sous le logo gov.br
    st.sidebar.markdown("<h2 style='text-align: center; color: #004C29;'>Projet de Data Visualisation</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: center; color: #004C29;'>Groupe BDX</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: #004C29;'>BAENA Miguel</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: #004C29;'>DAKPOGAN Paul</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: #004C29;'>XU Jérémy</p>", unsafe_allow_html=True)

    # Ajouter le logo ISUP après les noms, centré
    st.sidebar.markdown("<div style='text-align: center;'><img src='{}' width='100'></div>".format(logo_isup_url), unsafe_allow_html=True)

except Exception as e:
    st.sidebar.error(f"Une erreur est survenue avec les logos : {e}")



# Charger les données
# Lien brut (raw)
zip_url = "https://raw.githubusercontent.com/jeremyxu-pro/BDX_Project/main/DataViz/datavis_long_lat_with_grupo.zip"
# Télécharger et décompresser
response = requests.get(zip_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
 
# Lire un fichier CSV spécifique à l'intérieur du ZIP
with zip_file.open('datavis_long_lat_with_grupo.csv') as file:
    data = pd.read_csv(file)
 
try:
    

    # Vérifiez si les colonnes nécessaires existent
    required_columns = ['Region_Name', 'ANO_MODELO_Class', 'COBERTURA', 'INDENIZ', 'AGE_Group', 'COD_MODELO', 'SEXO']
    if not all(col in data.columns for col in required_columns):
        st.error("Certaines colonnes nécessaires sont manquantes dans vos données.")
    else:
        # Configuration des onglets
        tabs = st.tabs(["Introduction", "Composition du Portefeuille", "Analyses", "Cartographie", "Outils de Tarification", "Interpretations Modèle"])

        # **Onglet 1 : Introduction**
        with tabs[0]:
            st.title("Introduction")
            st.markdown("""
            Bienvenue dans ce projet de **Data Visualisation** développé par le groupe **BDX**. 
            Ce tableau de bord interactif est conçu pour analyser et explorer des données d'assurance au Brésil. 
            Voici un aperçu des différents onglets disponibles dans cette application :

            ### **1. Composition du Portefeuille**
            - Analyse des caractéristiques principales des polices d'assurance, notamment :
                - La répartition des clients par tranche d'âge et sexe.
                - La répartition des types de couverture.
                - Le nombre d'éléments dans chaque groupe de véhicules.

            ### **2. Analyses**
            - Exploration visuelle des données à travers :
                - L'indemnité totale par type de véhicule ou par groupe.
                - Les relations entre les causes de sinistres et les indemnités (boxplot).
                - La distribution des indemnités par type de couverture.

            ### **3. Cartographie**
            - Visualisation dynamique des données géographiques :
                - Carte des nombres de sinistres par région.
                - Carte des coûts d'indemnités par région.

            ### **4. Outils de Tarification**
            - Implémentation de deux modèles **XGBoost** pour prédire la fréquence et la sévérité respectivement.
                        
            ### **5. Interprétations Modèles**
            - Cet onglet fournit une analyse des modèles prédictifs en utilisant des outils tels que :
                - Les graphiques **SHAP** (SHapley Additive exPlanations) pour interpréter les contributions des variables au modèle.
        
            Ce tableau de bord est une démonstration de l'application des outils de **data science**, **modélisation** et **visualisation** dans le domaine des assurances. 
            Prenez le temps d'explorer chaque onglet pour une meilleure compréhension des données et des résultats. 🎉
            """)

        # **Onglet 2 : Composition du Portefeuille**
        with tabs[1]:
            st.title("Composition du Portefeuille")

            col1, col2 = st.columns(2)

            # Répartition par tranche d'âge
            with col1:
                st.header("Répartition par tranche d'âge")
                age_distribution = data['AGE_Group'].value_counts().reset_index()
                age_distribution.columns = ['Tranche d\'âge', 'Nombre']
                age_fig = px.bar(
                    age_distribution, 
                    x='Tranche d\'âge', 
                    y='Nombre', 
                    title="Répartition par tranche d'âge",color_discrete_sequence=["#6EAA6B"]
                )
                st.plotly_chart(age_fig, use_container_width=True)

            # Répartition par sexe
            with col2:
                st.header("Répartition par sexe")
                gender_distribution = data['SEXO'].value_counts().reset_index()
                gender_distribution.columns = ['Sexe', 'Nombre']
                gender_fig = px.pie(
                    gender_distribution, 
                    names='Sexe', 
                    values='Nombre', 
                    title="Répartition par sexe", color='Sexe',  # Définir les couleurs spécifiques
                    color_discrete_map={
                'M': '#A8E6A3',  # Vert clair
                'F': '#228B22'   # Vert plus foncé
                    }
                )
                st.plotly_chart(gender_fig, use_container_width=True)

            # Répartition par type de couverture
            st.header("Répartition des types de couverture")
            coverage_mapping = {
                1: "Couverture complète",
                2: "Couverture incendie et vol",
                3: "Couverture incendie",
                4: "Indemnisation intégrale",
                5: "Couverture collision et incendie",
                9: "Autres"
            }
            data['Coverage_Label'] = data['COBERTURA'].map(coverage_mapping)
            coverage_fig = px.pie(data, names='Coverage_Label', title="Types de couverture", hole=0.4,color_discrete_sequence=["#6EAA6B"])
             # Ajouter des options de mise en forme
            coverage_fig.update_traces(textinfo='percent+label')  # Afficher pourcentage + label
            coverage_fig.update_layout(
                legend_title="Types de couverture",  # Titre pour la légende
                height=500,  # Hauteur du graphique
                width=700    # Largeur du graphique
            )
            st.plotly_chart(coverage_fig, use_container_width=True)
            # Ajouter un mapping des causes de sinistres
            causa_mapping = {
                1: "Vol/Rapt",
                2: "Vol",
                3: "Rapt",
                4: "Collision partielle",
                5: "Collision avec indemnisation intégrale",
                6: "Incendie",
                7: "Assistance 24 heures",
                9: "Autres"
            }

            # Mapper les causes de sinistres
            data['CAUSA_MAPPED'] = data['CAUSA'].map(causa_mapping)

            # Calculer la distribution des causes de sinistres
            cause_distribution = data['CAUSA_MAPPED'].value_counts().reset_index()
            cause_distribution.columns = ['Cause de sinistre', 'Nombre']

            # Créer un graphique circulaire
            cause_pie_chart = px.pie(
                cause_distribution,
                names='Cause de sinistre',
                values='Nombre',
                title="Répartition des causes de sinistre",
                hole=0.4,color_discrete_sequence=["#6EAA6B"]
            )

            # Ajouter des options de mise en forme
            cause_pie_chart.update_traces(textinfo='percent+label')
            cause_pie_chart.update_layout(
                legend_title="Cause de sinistre",
                height=500,
                width=700
            )

            # Afficher le graphique dans Streamlit
            st.header("Répartition des causes de sinistre")
            st.plotly_chart(cause_pie_chart, use_container_width=True)

            # **Nombre d'occurrences par groupe (GRUPO)**
            st.header("Modèles de véhicules")
            group_counts = data['GRUPO'].value_counts().reset_index()
            group_counts.columns = ['GRUPO', 'Nombre']
            group_counts = group_counts.sort_values(by='Nombre', ascending=False)

            # Limiter l'affichage initial aux 10 premiers groupes
            top_10_groups = group_counts.head(10)
            remaining_groups = group_counts.iloc[10:]  # Les groupes restants

            # Créer le graphique pour les 10 premiers groupes
            top_10_fig = px.bar(
                top_10_groups,
                x='Nombre',
                y='GRUPO',
                orientation='h',
                title="Top 10 des modèles de véhicules assurés",
                labels={'Nombre': 'Nombre de véhicules', 'GRUPO': 'Groupe'},color_discrete_sequence=["#6EAA6B"]
            )

            top_10_fig.update_layout(
                height=400,
                xaxis=dict(title="Nombre d'Éléments"),
                yaxis=dict(title="Groupe", automargin=True, categoryorder='total ascending')
            )

            # Afficher le graphique des 10 premiers groupes
            st.plotly_chart(top_10_fig, use_container_width=True)

           
        # **Onglet 3 : Analyse**
        with tabs[2]:
            st.title("Analyse des indemnités")

            # Répartition des indemnités par tranche d'âge
            st.header("Indemnité moyenne par tranche d'âge")
            age_indemnity = data.groupby('AGE_Group')['INDENIZ'].mean().reset_index()
            age_fig = px.bar(
                age_indemnity,
                x='AGE_Group',
                y='INDENIZ',
                title="Indemnité moyenne par tranche d'âge",
                labels={'INDENIZ': 'Indemnité moyenne (R$)', 'AGE_Group': 'Tranche d\'âge'},color_discrete_sequence=["#6EAA6B"]
            )
            st.plotly_chart(age_fig, use_container_width=True)

            # Indemnité totale par modèle de véhicule
            st.header("Indemnité totale par modèle de véhicule")

            # Grouper les données par groupe et calculer l'indemnité totale
            group_indemnity = data.groupby('GRUPO')['INDENIZ'].sum().reset_index()

            # Trier les groupes par indemnité totale décroissante
            group_indemnity = group_indemnity.sort_values(by='INDENIZ', ascending=False)

            # Ne conserver que les 10 premiers groupes
            top_10_indemnity = group_indemnity.head(10)

            # Créer un graphique horizontal pour le top 10
            group_indemnity_fig = px.bar(
                top_10_indemnity,
                x='INDENIZ',
                y='GRUPO',
                orientation='h',
                title="Top 10 des véhicules par indemnité totale",
                labels={'INDENIZ': 'Indemnité Totale (R$)', 'GRUPO': 'Véhicules'},color_discrete_sequence=["#8CCB87"]
            )

            # Configurer l'apparence du graphique
            group_indemnity_fig.update_layout(
                height=600,
                xaxis=dict(title="Indemnité totale (R$)"),
                yaxis=dict(title="Groupe", automargin=True, categoryorder='total ascending')
            )

            # Afficher le graphique
            st.plotly_chart(group_indemnity_fig, use_container_width=True)
            st.header("Indemnité par cause de sinistre")
    
           # Mapping des valeurs de CAUSA
            causa_mapping = {
                1: "Vol/Rapt",
                2: "Vol",
                3: "Rapt",
                4: "Collision partielle",
                5: "Collision avec indemnisation intégrale",
                6: "Incendie",
                7: "Assistance 24 heures",
                9: "Autres"
            }

            # Ajouter une colonne mappée dans le DataFrame
            data['CAUSA_MAPPED'] = data['CAUSA'].map(causa_mapping)

            # Ajouter un boxplot dans le troisième onglet
            with tabs[2]:
                
                
                
                # Créer un boxplot avec Plotly Express
                causa_boxplot_mapped = px.box(
                    data,
                    x='CAUSA_MAPPED',
                    y='INDENIZ',
                    title="Relation entre la cause et l'indemnité du sinistre",
                    labels={'CAUSA_MAPPED': 'Cause de sinistre', 'INDENIZ': 'Indemnité (R$)'},color_discrete_sequence=["#6EAA6B"]
                )
                
                # Configurer le graphique
                causa_boxplot_mapped.update_layout(
                    xaxis=dict(title="Cause de sinistre"),
                    yaxis=dict(title="Indemnité (R$)"),
                    height=500,
                    width=800
                )
                
                # Afficher le graphique dans Streamlit
                st.plotly_chart(causa_boxplot_mapped, use_container_width=True)
                # Ajouter un mapping des types de couverture
                coverage_mapping = {
                    1: "Couverture complète",
                    2: "Couverture incendie et vol",
                    3: "Couverture incendie",
                    4: "Indemnisation intégrale",
                    5: "Couverture collision et incendie",
                    9: "Autres"
                }

                # Mapper les types de couvertures
                data['COBERTURA_MAPPED'] = data['COBERTURA'].map(coverage_mapping)

                # Créer un boxplot des indemnités par type de couverture
                coverage_boxplot = px.box(
                    data,
                    x='COBERTURA_MAPPED',
                    y='INDENIZ',
                    title="Indemnité par type de couverture",
                    labels={'COBERTURA_MAPPED': 'Type de couverture', 'INDENIZ': 'Indemnité (R$)'},color_discrete_sequence=["#6EAA6B"]
                )

                # Mettre à jour le style du graphique
                coverage_boxplot.update_layout(
                    xaxis_title="Type de couverture",
                    yaxis_title="Indemnité (R$)",
                    height=600,
                    width=800
                )

                # Afficher le graphique dans Streamlit
                st.header("Indemnité par type de couverture")
                
                st.plotly_chart(coverage_boxplot, use_container_width=True)



        # **Onglet 4 : Cartographie**
        with tabs[3]:
            st.header("Visualisation du coût des sinistres par état")
            
            
        
            carto_data_path = "https://raw.githubusercontent.com/jeremyxu-pro/BDX_Project/main/DataViz/Aggregated_Claims_Data_by_Region.csv"
            carto_data = pd.read_csv(carto_data_path)
 
            # Créer une carte Folium
            m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)

            # Définir l'échelle de couleur
            color_scale = cm.LinearColormap(
                colors=['#003366', '#008000', '#FFD700'],
                vmin=carto_data['Total_Claims'].min(),
                vmax=carto_data['Total_Claims'].max(),
                caption='Montant des sinitres (R$)'
            )
            # Ajouter l'échelle de couleurs à la carte
            color_scale.add_to(m)
            # Ajouter des régions sur la carte
            scale_factor = 1e6
            max_radius = 30
            for _, row in carto_data.iterrows():
                folium.CircleMarker(
                    location=(row['Latitude'], row['Longitude']),
                    radius=min(max(row['Total_Claims'] / scale_factor, 5), max_radius),
                    color=color_scale(row['Total_Claims']),
                    fill=True,
                    fill_color=color_scale(row['Total_Claims']),
                    fill_opacity=0.8,
                    tooltip=(
                        f"<strong>{row['Region_Name']}</strong><br>"
                        f"Total Claims: R$ {round(row['Total_Claims']):,}<br>"
                        f"Average Claims: R$ {round(row['Average_Claims']):,}"
                    )
                ).add_to(m)

            # Afficher la carte dans Streamlit
            st_folium(m, width=700, height=500)


            file_path = 'https://raw.githubusercontent.com/jeremyxu-pro/BDX_Project/main/DataViz/Cleaned_Merged_Claims_Data.csv'
            data = pd.read_csv(file_path, sep=';')
            # Titre pour la nouvelle carte
            st.header("Visualisation du nombre de sinistre par état")

            # Créer une carte Folium
            m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)  # Centré sur le Brésil

            # Définir une échelle de couleur basée sur "number_of_claims_updated"
            color_scale = cm.LinearColormap(
                colors=['#003366', '#008000', '#FFD700'],  # Palette gov.br : bleu, vert, jaune
                vmin=data['number_of_claims_updated'].min(),
                vmax=data['number_of_claims_updated'].max(),
                caption='Nombre de sinistre'
            )

            # Ajouter l'échelle de couleurs à la carte
            m.add_child(color_scale)

            # Ajouter les régions sur la carte
            scale_factor = 1e4  # Facteur d'échelle pour la taille des cercles
            max_radius = 50  # Rayon maximum des cercles
            for _, row in data.iterrows():
                radius = max(row['number_of_claims_updated'] / scale_factor, 5)  # Calculer le rayon
                folium.CircleMarker(
                    location=(row['Latitude'], row['Longitude']),
                    radius=min(radius, max_radius),  # Limiter le rayon
                    color=color_scale(row['number_of_claims_updated']),
                    fill=True,
                    fill_color=color_scale(row['number_of_claims_updated']),
                    fill_opacity=0.8,
                    tooltip=(
                        f"<strong>{row['Region_Name']}</strong><br>"
                        f"Nombre de sinistre : {round(row['number_of_claims_updated']):,}"
                    )
                ).add_to(m)

            # Afficher la carte dans Streamlit
            st_folium(m, width=700, height=500)


        # **Onglet 6 : Tarification**
        with tabs[4]:
            st.markdown("""
            Cet onglet fournit un outil réalisant une tarification simplifiée selon une logique de fréquence-sévérité avec deux modèles XGBOOST.
            """)

            # Fonction pour télécharger un fichier depuis une URL
            def download_model(url, filename):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, 'wb') as file:
                        file.write(response.content)
                    return filename
                else:
                    st.error(f"Erreur lors du téléchargement du modèle : {url}")
                    return None

            # Ajouter un style CSS pour personnaliser les boutons et les hypothèses
            st.markdown(
                """
                <style>
                /* Style pour le bouton prédire */
                .stButton > button {
                    background-color: #FF4B4B;
                    color: white;
                    font-size: 18px;
                    padding: 10px 20px;
                    border-radius: 8px;
                    border: none;
                    cursor: pointer;
                    display: block;
                    margin: 20px auto; /* Centrer horizontalement */
                }
                .stButton > button:hover {
                    background-color: #D03434;
                }

                /* Style pour la boîte des hypothèses */
                .hypotheses-box {
                    background-color: #f0f8ff; /* Bleu clair */
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    border: 1px solid #dcdcdc;
                }

                /* Style pour les boîtes de sélection */
                .stSelectbox {
                    background-color: #f5f5f5 !important; /* Gris clair */
                    color: #333333 !important; /* Texte gris foncé */
                    border: 1px solid #dcdcdc !important; /* Bordure grise */
                    padding: 10px !important;
                    border-radius: 8px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Fonction principale pour la tarification
            def run_tarification():
                st.title("Outil de tarification - Fréquence & Sévérité")

                # Téléchargement des modèles
                model_frequency_url = "https://raw.githubusercontent.com/jeremyxu-pro/BDX_Project/main/DataViz/model_pipeline_frequency.pkl"
                model_severity_url = "https://raw.githubusercontent.com/jeremyxu-pro/BDX_Project/main/DataViz/model_pipeline_severity.pkl"

                frequency_model_file = download_model(model_frequency_url, "model_pipeline_frequency.pkl")
                severity_model_file = download_model(model_severity_url, "model_pipeline_severity.pkl")

                if not frequency_model_file or not severity_model_file:
                    return

                try:
                    # Charger les modèles depuis les fichiers téléchargés
                    model_frequency = joblib.load(frequency_model_file)
                    model_severity = joblib.load(severity_model_file)
                except Exception as e:
                    st.error(f"Erreur lors du chargement des modèles : {e}")
                    return

                # Récupérer les catégories pour les colonnes
                preprocessor = model_frequency.named_steps['preprocessor']
                encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                categories = encoder.categories_
                cat_cols = preprocessor.transformers_[0][1].feature_names_in_

                # Entrées utilisateur
                st.subheader("Sélectionnez les caractéristiques de tarification :")
                inputs = {}
                col1, col2 = st.columns(2)
                for i, col in enumerate(cat_cols):
                    label_french = {
                        "DrivAge": "Tranche d'âge du conducteur :",
                        "HadRob": "A-t-il subi au moins un vol ?",
                        "HadPartColl": "A-t-il subi au moins une collision partielle ?",
                        "HadTotColl": "A-t-il subi au moins une collision totale ?",
                        "HadFire": "A-t-il subi au moins un incendie ?",
                        "HadOther": "A-t-il subi d'autres dommages non répertoriés ?",
                        "VehYear_Class": "Année du modèle de véhicule :",
                        "Region": "Région :",
                        "VehGroup_bin": "Catégorie de véhicule :",
                    }.get(col, col)

                    with col1 if i % 2 == 0 else col2:
                        inputs[col] = st.selectbox(f"{label_french}", options=categories[i])

                # Bouton de prédiction
                if st.button("Prédire"):
                    try:
                        # Préparer les données d'entrée
                        new_observation = pd.DataFrame([inputs])

                        # Prédictions
                        freq_prediction = model_frequency.predict(new_observation)[0]
                        sev_prediction = model_severity.predict(new_observation)[0]

                        # Calcul de la prime pure
                        prime_pure = freq_prediction * sev_prediction

                        # Afficher les résultats
                        st.success(f"Montant de la prime pure annuelle prédite : {prime_pure:,.2f} R$")
                    except Exception as e:
                        st.error(f"Une erreur est survenue pendant la prédiction : {e}")

            # Appel de la fonction dans l'onglet Tarification
            run_tarification()

            # **Onglet 5 : Interpretation Modèle**
        with tabs[5]:
            st.title("Interpretation Modèle")
            st.markdown("""
            Cet onglet affiche les graphiques SHAP (SHapley Additive exPlanations) permettant d'interpréter les contributions des variables dans nos deux modèles. Nous mettons en avant uniquement les 10 variables ayant les contributions les plus significatives. Par ailleurs, il est important de noter que l'analyse est réalisée à l'échelle des classes, afin d'éviter toute compensation qui pourrait survenir en passant à une échelle plus agrégée, soit celle des variables individuelles.
            """)

            # Charger les modèles
            model_frequency_file = "model_pipeline_frequency.pkl"
            model_severity_file = "model_pipeline_severity.pkl"

            try:
                # Charger les modèles
                model_frequency = joblib.load(model_frequency_file)
                model_severity = joblib.load(model_severity_file)

                # Extraire le préprocesseur et le modèle final
                preprocessor_frequency = model_frequency.named_steps['preprocessor']
                final_model_frequency = model_frequency.named_steps['model']

                preprocessor_severity = model_severity.named_steps['preprocessor']
                final_model_severity = model_severity.named_steps['model']

                # Générer des données synthétiques basées sur les colonnes attendues
                def generate_synthetic_data(preprocessor, n_samples=100):
                    """Générer des données synthétiques compatibles avec le préprocesseur."""
                    feature_names = preprocessor.transformers_[0][1].get_feature_names_out()
                    num_features = len(feature_names)
                    data = np.random.rand(n_samples, num_features)
                    return pd.DataFrame(data, columns=feature_names)

                # Générer des données synthétiques pour fréquence et sévérité
                X_frequency_synthetic = generate_synthetic_data(preprocessor_frequency)
                X_severity_synthetic = generate_synthetic_data(preprocessor_severity)

                # Calcul des valeurs SHAP pour la fréquence
                explainer_frequency = shap.TreeExplainer(final_model_frequency)
                shap_values_frequency = explainer_frequency.shap_values(X_frequency_synthetic)

                # Calcul des valeurs SHAP pour la sévérité
                explainer_severity = shap.TreeExplainer(final_model_severity)
                shap_values_severity = explainer_severity.shap_values(X_severity_synthetic)

                # Affichage des graphiques SHAP
                st.subheader("Graphiques SHAP")

                # Ligne supérieure : Graphiques d'importance
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Importance des variables - Fréquence")
                    plt.figure(figsize=(10, 5))
                    shap.summary_plot(
                        shap_values_frequency,
                        X_frequency_synthetic,
                        plot_type="bar",
                        max_display=10,
                        show=False
                    )
                    st.pyplot(plt.gcf())

                with col2:
                    st.subheader("Importance des variables - Sévérité")
                    plt.figure(figsize=(10, 5))
                    shap.summary_plot(
                        shap_values_severity,
                        X_severity_synthetic,
                        plot_type="bar",
                        max_display=10,
                        show=False
                    )
                    st.pyplot(plt.gcf())

                # Ligne inférieure : Résumés SHAP
                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Résumé SHAP - Fréquence")
                    plt.figure(figsize=(10, 5))
                    shap.summary_plot(
                        shap_values_frequency,
                        X_frequency_synthetic,
                        max_display=10,
                        show=False
                    )
                    st.pyplot(plt.gcf())

                with col4:
                    st.subheader("Résumé SHAP - Sévérité")
                    plt.figure(figsize=(10, 5))
                    shap.summary_plot(
                        shap_values_severity,
                        X_severity_synthetic,
                        max_display=10,
                        show=False
                    )
                    st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"Erreur lors de la génération des graphiques SHAP : {e}")
            
            # Section pour l'analyse de dépendance SHAP dynamique
            st.subheader("Analyse de Dépendance SHAP - Variables Catégoriques")

            # Charger les modèles
            @st.cache_resource
            def load_models():
                model_frequency_path = "model_pipeline_frequency.pkl"
                model_severity_path = "model_pipeline_severity.pkl"
                model_frequency = joblib.load(model_frequency_path)
                model_severity = joblib.load(model_severity_path)
                return model_frequency, model_severity

            # Charger les modèles
            model_frequency, model_severity = load_models()

            # Extraire les préprocesseurs et modèles finaux
            preprocessor_frequency = model_frequency.named_steps['preprocessor']
            final_model_frequency = model_frequency.named_steps['model']

            preprocessor_severity = model_severity.named_steps['preprocessor']
            final_model_severity = model_severity.named_steps['model']

            # Générer des données synthétiques pour la démonstration
            def generate_synthetic_data(preprocessor, n_samples=100):
                """Génère des données synthétiques compatibles avec le préprocesseur."""
                feature_names = preprocessor.transformers_[0][1].get_feature_names_out()
                num_features = len(feature_names)
                data = np.random.rand(n_samples, num_features)
                return pd.DataFrame(data, columns=feature_names)

            X_frequency_synthetic = generate_synthetic_data(preprocessor_frequency)
            X_severity_synthetic = generate_synthetic_data(preprocessor_severity)

            # Calculer les valeurs SHAP pour les deux modèles
            explainer_frequency = shap.TreeExplainer(final_model_frequency)
            shap_values_frequency = explainer_frequency.shap_values(X_frequency_synthetic)

            explainer_severity = shap.TreeExplainer(final_model_severity)
            shap_values_severity = explainer_severity.shap_values(X_severity_synthetic)

            # Interface utilisateur : choix dynamique des variables
            st.subheader("Choisissez une variable principale et une variable d'interaction :")

            # Choix du modèle
            model_choice = st.radio(
                "Choisissez le modèle pour l'analyse :",
                options=["Fréquence", "Sévérité"]
            )

            # Déterminer les données et valeurs SHAP en fonction du modèle choisi
            if model_choice == "Fréquence":
                shap_values = shap_values_frequency
                X_sample = X_frequency_synthetic
                feature_names = X_frequency_synthetic.columns
            else:
                shap_values = shap_values_severity
                X_sample = X_severity_synthetic
                feature_names = X_severity_synthetic.columns

            # Sélection des variables
            main_feature = st.selectbox("Caractéristique principale", feature_names)
            interaction_feature = st.selectbox("Caractéristique d'interaction", feature_names)

            # Bouton pour afficher le graphique de dépendance
            if st.button("Afficher le graphique de dépendance"):
                try:
                    # Affichage du graphique de dépendance SHAP avec correction des axes
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        main_feature,          # Variable principale (axe X)
                        shap_values=shap_values,           # Valeurs SHAP
                        features=X_sample,                 # Données utilisées pour SHAP
                        interaction_index=interaction_feature  # Variable d'interaction (colorimétrie)
                    )
                    st.pyplot(plt.gcf())


                    st.write(
                        f"Le graphique ci-dessus montre la dépendance SHAP pour `{main_feature}` "
                        f"avec l'interaction `{interaction_feature}` basée sur le modèle **{model_choice}**."
                    )
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la génération du graphique : {e}")





        
except FileNotFoundError:
    st.error(f"Fichier introuvable : {file_path}. Vérifiez le chemin.")
except Exception as e:
    st.error(f"Une erreur est survenue : {e}")
