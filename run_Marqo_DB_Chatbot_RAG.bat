@echo off
cd /d "%~dp0"
call venv\Scripts\activate


echo Running the project...
python Marqo_DB_Chatbot_RAG.py

echo Press any key to exit...
pause >nul