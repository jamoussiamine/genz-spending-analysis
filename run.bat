@echo off
REM Lance l'application sous Windows : cree l'environnement au premier lancement.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creation de l'environnement virtuel...
    py -3.11 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Erreur : installez Python 3.10, 3.11 ou 3.12 depuis https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

echo [2/2] Verification des dependances...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --timeout 120 --retries 5 -r requirements.txt
if errorlevel 1 (
    echo Erreur pendant l'installation. Verifiez la connexion internet puis relancez run.bat.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
