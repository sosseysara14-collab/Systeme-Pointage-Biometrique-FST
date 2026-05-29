import pickle

# 1. Chargement du fichier pkl
nom_fichier = "employes_features.pkl"

try:
    with open(nom_fichier, "rb") as f:
        data = pickle.load(f)
    
    # 2. Affichage des profils trouvés
    print("=" * 50)
    print(f"📊 NOMBRE DE PROFILS TROUVÉS : {len(data)}")
    print("=" * 50)
    
    for i, (nom_employe, infos) in enumerate(data.items(), 1):
        # On récupère le poste et le statut s'ils existent dans le dictionnaire
        poste = infos.get("poste", "Non spécifié")
        statut = infos.get("statut", "Non spécifié")
        
        print(f"{i:02d}) Nom : {nom_employe} | Poste : {poste} | Statut : {statut}")
        
except FileNotFoundError:
    print(f"❌ Erreur : Le fichier '{nom_fichier}' n'est pas dans le même dossier.")
except Exception as e:
    print(f"❌ Impossible de lire le fichier : {e}")