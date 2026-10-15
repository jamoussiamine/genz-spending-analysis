#!/usr/bin/env bash
# Lance l'application sous macOS / Linux : crée l'environnement au premier lancement.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "[1/2] Création de l'environnement virtuel..."
    for py in python3.11 python3.12 python3.10 python3; do
        if command -v "$py" >/dev/null 2>&1; then "$py" -m venv .venv && break; fi
    done
fi

echo "[2/2] Vérification des dépendances..."
.venv/bin/python -m pip install --disable-pip-version-check --timeout 120 --retries 5 -r requirements.txt

.venv/bin/python -m streamlit run streamlit_app.py
