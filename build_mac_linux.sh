#!/bin/bash
python3 -m pip install -r requirements.txt
python3 -m PyInstaller --onefile --windowed learning_chess_bot_resizable_gui.py
