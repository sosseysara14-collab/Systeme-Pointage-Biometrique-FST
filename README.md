# Système de Pointage Biométrique par Reconnaissance Faciale (FST)

Ce dépôt contient le prototype fonctionnel d'un système de gestion de présence en temps réel combinant l'extraction de signatures biométriques via **FaceNet** et une interface utilisateur moderne sous **CustomTkinter**.

##  Structure du Projet
* `main.py` : Application principale avec interface graphique, capture OpenCV et matching en temps réel.
* `extraire.py` : Script de traitement en amont exécuté pour l'extraction des embeddings.
* `employes_features.pkl` : Base de données sérialisée contenant les signatures des 40 profils de test (Dataset Olivetti).

##  Instructions pour l'Exécution
1. Cloner le dépôt ou télécharger le fichier ZIP.
2. Installer les dépendances requises :
   ```bash
   pip install customtkinter opencv-python deepface tensorflow
