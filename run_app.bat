
---

#  Windows AutoRun – `run_app.bat`

```bat
@echo off
echo Starting Insurance Cost Prediction App...

call venv\Scripts\activate

streamlit run src\streamlit_app.py

pause
