# Ping Pong AI Analyst — Roboflow 

Ping Pong AI Analyst est un outil de vision par ordinateur conçu pour détecter et suivre la balle de tennis de table, analyser ses trajectoires et classifier les frappes des joueurs. Ce dépôt repose sur deux axes principaux : la détection de balle via un modèle **YOLOv11 (Roboflow)** et la détection de frappes via **YOLOP11n-pose**.

---

## Fonctionnalités clés

* **Détection de balle par modèle entraîné** : Utilise un modèle YOLO fine-tuné pour détecter la balle sur des fichiers vidéo ou en flux webcam temps réel.
* **Analyse de trajectoire** : Reconstruit la trajectoire de la balle frame par frame en conservant un historique glissant des positions détectées.
* **Classification des frappes** : Détecte les coups (Coup Droit / Revers) en combinant la détection de l'impact balle-raquette (intersection des bounding boxes) et l'analyse de la pose du joueur via YOLO Pose.
* **Évaluation & Métriques** : Compare les détections aux annotations manuelles (CSV) pour calculer Précision, Rappel, Vrais Positifs, Faux Positifs et Faux Négatifs.
---
## Détail des scripts principaux

### `ball_detection_and_trajectory/`

**`results_trajectory.py`** — Trajectoire sur vidéo  
Lance la détection YOLO sur chaque frame d'un fichier `.mp4`. Conserve un historique glissant des `N` dernières positions détectées (`deque`) et les relie par des lignes jaunes pour visualiser la trajectoire de la balle.

**`results_trajectory_live.py`** — Trajectoire en live  
Même logique que `results_trajectory.py`, mais appliquée au flux d'une webcam. Applique un effet miroir horizontal pour un rendu naturel. Si la balle n'est pas détectée sur une frame, un `None` est inséré pour éviter de relier des positions temporellement trop distantes.

---

### `strokes_detection/`

**`pose_detection.py`** — Évaluation sur vidéo  
Utilise deux modèles YOLO en parallèle : un modèle objet (`best-2.pt`) pour détecter la balle (classe 0) et la raquette (classe 1), et un modèle pose (`yolo11x-pose.pt`) pour analyser les keypoints du joueur. Lorsque les bounding boxes de la balle et de la raquette se chevauchent (impact détecté), le script compare la position du poignet droit (keypoint 10) par rapport au coude droit (keypoint 8) pour classifier le coup en `COUP DROIT` ou `REVERS`. Chaque détection est ensuite comparée aux annotations du CSV (`Start`, `End`, `Name`) pour calculer les métriques finales.

**`pose_detection_live.py`** — Arbitrage en temps réel  
Reprend la même logique d'intersection bounding box + classification par keypoints, mais sur le flux webcam. Aucune métrique n'est calculée : le type de coup détecté est simplement affiché à l'écran pendant 1,5 seconde après chaque impact. Utilise le modèle pose `yolo11n-pose.pt` (Nano) pour maximiser les performances en temps réel.

---

## Données & Modèles

Les fichiers suivants doivent être ajoutés manuellement (exclus du dépôt via `.gitignore`) :

* **Vidéos** : à placer dans `roboflow/dataset_labelise/<nom_du_dataset>/`
* **Annotations** : fichiers `.csv` avec les colonnes `Start`, `End`, `Name` dans les mêmes dossiers
* **Poids des modèles** : à placer dans `roboflow/models/`
  * `best-2.pt` — modèle YOLO fine-tuné pour la détection balle + raquette
  * `yolo11n-pose.pt` — modèle YOLO Pose Nano (pour le live)
  * `yolo11x-pose.pt` — modèle YOLO Pose XL (pour l'évaluation offline)

---

## Prérequis & Installation

Toutes les dépendances Python nécessaires sont listées dans `roboflow/requirement.txt`. Pour les installer, exécute la commande suivante depuis la racine du projet :

```bash
pip install -r roboflow/requirement.txt
```

Les bibliothèques principales utilisées sont :

* `ultralytics` — chargement et inférence des modèles YOLO (détection objet + pose)
* `opencv-python` — capture vidéo, traitement d'images et affichage
* `numpy` — calculs vectoriels
* `pandas` — lecture des fichiers d'annotations CSV et calcul des métriques