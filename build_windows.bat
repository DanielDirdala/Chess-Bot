@echo off
python -m pip install -r requirements.txt
python -m PyInstaller --onefile --windowed learning_chess_bot_resizable_gui.py
pause
