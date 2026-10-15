# Guide d'installation

Ce guide explique comment installer et lancer le projet sur votre propre ordinateur, étape par étape, même sans expérience préalable.

Deux méthodes sont possibles. Choisissez-en une seule :

| Méthode | Pour qui | Ce qu'il faut installer |
|---|---|---|
| [A. Docker](#méthode-a--docker) | Le plus simple : aucune installation Python | Git + Docker Desktop |
| [B. Python](#méthode-b--python) | Pour modifier le code ou exécuter les notebooks | Git + Python 3.10, 3.11 ou 3.12 |

Durée : 5 à 15 minutes la première fois, selon votre connexion (environ 200 Mo de bibliothèques à télécharger). Les lancements suivants prennent quelques secondes.

---

## Étape 1 : récupérer le projet

### Avec Git (recommandé)

1. Installer Git : https://git-scm.com/downloads (garder les options par défaut).
2. Ouvrir un terminal :
   - **Windows** : menu Démarrer, taper `PowerShell`, puis Entrée
   - **macOS** : Applications > Utilitaires > Terminal
   - **Linux** : Ctrl + Alt + T
3. Copier le projet puis entrer dans son dossier :

```bash
git clone https://github.com/jamoussiamine/genz-spending-analysis.git
cd genz-spending-analysis
```

### Sans Git

Sur la page GitHub du projet : bouton vert **Code** > **Download ZIP**. Décompresser le fichier, puis ouvrir un terminal dans le dossier obtenu.

---

## Méthode A : Docker

### 1. Installer Docker Desktop

- Télécharger : https://www.docker.com/products/docker-desktop/
- Installer, puis **lancer Docker Desktop** et attendre que l'icône indique « Engine running ».
- Sous Windows, accepter l'installation de WSL 2 si elle est proposée, puis redémarrer le PC.

Vérifier dans le terminal :

```bash
docker --version
```

### 2. Construire et lancer l'application

Dans le dossier du projet :

```bash
docker compose up --build
```

La première fois, Docker télécharge Python et les bibliothèques (plusieurs minutes). Quand le message `You can now view your Streamlit app in your browser` apparaît, ouvrir :

**http://localhost:8501**

### 3. Arrêter / relancer

- Arrêter : `Ctrl + C` dans le terminal, puis `docker compose down`
- Relancer plus tard (quelques secondes) : `docker compose up`
- Après une modification du code : `docker compose up --build`

---

## Méthode B : Python

### 1. Installer Python

Télécharger **Python 3.11** (recommandé) : https://www.python.org/downloads/release/python-3119/

- **Windows** : pendant l'installation, **cocher « Add python.exe to PATH »** avant de cliquer sur Install.
- **macOS / Linux** : suivre l'installateur ou utiliser le gestionnaire de paquets du système.

> Python 3.13 ou plus récent ne fonctionne pas : les versions de scikit-learn utilisées par le projet n'existent pas pour ces versions.

Vérifier dans le terminal :

```bash
python --version     # Windows
python3 --version    # macOS / Linux
```

Le résultat doit être `Python 3.10.x`, `3.11.x` ou `3.12.x`.

### 2. Lancer avec le script automatique

- **Windows** : double-cliquer sur `run.bat` dans le dossier du projet
- **macOS / Linux** : dans le terminal, `bash run.sh`

Au premier lancement, le script :
1. crée un environnement Python isolé dans le dossier `.venv` (le reste de votre ordinateur n'est pas modifié),
2. installe les bibliothèques avec les versions exactes du projet,
3. ouvre l'application dans votre navigateur.

L'adresse s'affiche dans le terminal, en général **http://localhost:8501**.

Pour arrêter : fermer la fenêtre du terminal, ou `Ctrl + C`.

### 3. Ou installer à la main

```bash
python -m venv .venv
```

Activer l'environnement :

| Système | Commande |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Le nom `(.venv)` doit apparaître au début de la ligne du terminal. Puis :

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 4. Exécuter les notebooks (facultatif)

L'application fonctionne sans cette étape : les résultats sont déjà fournis dans `data/`. Pour refaire les analyses :

```bash
pip install -r requirements-notebooks.txt
jupyter notebook notebooks/
```

Exécuter les notebooks dans l'ordre, de `01` à `07`. Chacun écrit ses résultats dans `data/processed/` et `data/results/`, que l'application lit.

---

## Vérifier que tout fonctionne

Dans l'application, le menu de gauche propose 9 pages. Chacune doit s'afficher sans message d'erreur rouge :

1. Déploiement - Prédictions interactives
2. Data Understanding
3. Data Preparation
4. Feature Engineering
5. ACP
6. Clustering K-Means
7. Clustering K-Means vs DBSCAN
8. Classification
9. Régression

---

## Problèmes fréquents

| Problème | Solution |
|---|---|
| `Port 8501 is already in use` ou la page affiche une autre application | Un autre programme utilise le port. Docker : remplacer `"8501:8501"` par `"8502:8501"` dans `docker-compose.yml`, puis ouvrir http://localhost:8502. Python : `streamlit run streamlit_app.py --server.port 8502` |
| `python` n'est pas reconnu (Windows) | Python n'est pas dans le PATH. Réinstaller en cochant « Add python.exe to PATH », ou utiliser `py -3.11` à la place de `python` |
| `Activate.ps1 ne peut pas être chargé` (PowerShell) | Exécuter une fois `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou utiliser `run.bat` |
| `No matching distribution found for scikit-learn==1.4.2` | Version de Python trop récente (3.13 ou plus). Installer Python 3.11 |
| `Read timed out` ou `THESE PACKAGES DO NOT MATCH THE HASHES` pendant l'installation | Connexion instable : un fichier est arrivé incomplet. Relancer simplement la même commande ; les fichiers déjà téléchargés ne sont pas retéléchargés |
| `Cannot connect to the Docker daemon` / `failed to connect to the docker API` | Docker Desktop n'est pas lancé. L'ouvrir et attendre « Engine running » |
| `ModuleNotFoundError: No module named ...` | Les bibliothèques ne sont pas installées dans l'environnement actif. Vérifier que `(.venv)` est affiché, puis relancer `pip install -r requirements.txt` |
| L'application s'ouvre avec un ancien Anaconda et plante | Ne pas lancer depuis Anaconda : utiliser `run.bat` / `run.sh` ou l'environnement `.venv`, qui contient les bonnes versions |

Pour un autre problème, ouvrir une *issue* sur GitHub en copiant le message d'erreur complet.
