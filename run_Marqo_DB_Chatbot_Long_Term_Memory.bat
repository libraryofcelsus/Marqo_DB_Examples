@echo off
cd /d "%~dp0"
call venv\Scripts\activate


echo Running the project...
python Marqo_DB_Chatbot_Long_Term_Memory.py

echo Press any key to exit...
pause >nul