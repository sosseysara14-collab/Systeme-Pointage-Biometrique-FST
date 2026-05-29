import os
import pickle
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from deepface import DeepFace
from scipy.spatial.distance import cosine

# --- CONFIGURATION DE L'INTERFACE ---
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")

class AppPointage(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Système de Pointage Biométrique - FST")
        self.geometry("900x600")
        self.resizable(False, False)

        # Seuil de sécurité strict (0.42) pour éviter les faux profils
        self.SEUIL_COSINUS = 0.42
        self.charger_base_donnees()

        self.video_capture = cv2.VideoCapture(0)
        self.compteur_frames = 0  
        
        self.derniere_prediction = "En attente..."
        self.dernier_poste = "-"
        self.dernier_statut = "-"
        self.couleur_ui = "#333333"

        # --- DESIGN LAYOUT ---
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.sidebar = ctk.CTkFrame(self, corner_radius=15)
        self.sidebar.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.sidebar, text="STATUT POINTAGE", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(pady=30)

        self.status_card = ctk.CTkFrame(self.sidebar, fg_color=self.couleur_ui, width=220, height=100, corner_radius=10)
        self.status_card.pack(pady=10)
        self.status_card.pack_propagate(False)

        self.nom_label = ctk.CTkLabel(self.status_card, text=self.derniere_prediction, font=ctk.CTkFont(size=16, weight="bold"))
        self.nom_label.pack(expand=True)

        # Les labels pour afficher le Poste et le Statut en dessous
        self.info_label = ctk.CTkLabel(self.sidebar, text="Poste: -\nStatut: -", font=ctk.CTkFont(size=14), justify="left")
        self.info_label.pack(pady=20)

        self.btn_quitter = ctk.CTkButton(self.sidebar, text="Quitter", fg_color="#C0392B", hover_color="#922B21", command=self.quitter_application)
        self.btn_quitter.pack(side="bottom", pady=30)

        self.update_frame()

    def charger_base_donnees(self):
        try:
            with open("employes_features.pkl", "rb") as f:
                self.embeddings_db = pickle.load(f)
            print(f"✅ Base de données chargée ({len(self.embeddings_db)} profils).")
        except Exception as e:
            print(f"❌ Erreur pkl : {e}")
            self.embeddings_db = {}

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.compteur_frames += 1

            # Analyse toutes les 30 images
            if self.compteur_frames % 30 == 0:
                try:
                    faces = DeepFace.represent(img_path=frame, model_name="Facenet", 
                                               detector_backend="opencv", enforce_detection=False)
                    
                    if faces and "embedding" in faces[0]:
                        current_embedding = faces[0]["embedding"]
                        nom_identifie = "Inconnu"
                        poste_info = "-"
                        statut_info = "-"
                        meilleure_distance = 1.0

                        # Recherche dans la base de données
                        for emp_name, data in self.embeddings_db.items():
                            dist = cosine(current_embedding, data["embedding"])
                            if dist < meilleure_distance:
                                meilleure_distance = dist
                                if dist < self.SEUIL_COSINUS:
                                    nom_identifie = emp_name
                                    # Extraction des données dynamiques du dictionnaire s'il y en a
                                    poste_info = data.get("poste", "Ingénieur")
                                    statut_info = data.get("statut", "Présent")

                        if nom_identifie != "Inconnu":
                            self.derniere_prediction = nom_identifie
                            self.dernier_poste = poste_info
                            self.dernier_statut = statut_info
                            self.couleur_ui = "#1E8449"  # Vert
                        else:
                            self.derniere_prediction = "INCONNU"
                            self.dernier_poste = "-"
                            self.dernier_statut = f"Inconnu (Dist: {meilleure_distance:.2f})"
                            self.couleur_ui = "#922B21"  # Rouge
                    else:
                        self.derniere_prediction = "En attente..."
                        self.dernier_poste = "-"
                        self.dernier_statut = "-"
                        self.couleur_ui = "#333333"

                except Exception as e:
                    pass

            # --- MISE À JOUR DE L'INTERFACE ---
            self.status_card.configure(fg_color=self.couleur_ui)
            self.nom_label.configure(text=self.derniere_prediction)
            
            # Affichage dynamique des informations de la personne
            self.info_label.configure(text=f"Poste: {self.dernier_poste}\nStatut: {self.dernier_statut}")

            # Carré de guidage blanc
            h, w, _ = frame.shape
            cv2.rectangle(frame, (w//2 - 110, h//2 - 140), (w//2 + 110, h//2 + 140), (255, 255, 255), 1)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(580, 435))
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

        self.after(15, self.update_frame)

    def quitter_application(self):
        self.video_capture.release()
        self.destroy()

if __name__ == "__main__":
    app = AppPointage()
    app.mainloop()