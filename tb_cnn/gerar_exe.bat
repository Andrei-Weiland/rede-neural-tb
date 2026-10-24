@echo off
REM Gera TriagemTB.exe (pasta dist\TriagemTB\) com PyInstaller.
REM Requer o .venv do projeto ja criado e modelo em modelos\modelo_final.pt

cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat

python -m pip install -U pyinstaller
if errorlevel 1 exit /b 1

if not exist "modelos\modelo_final.pt" (
  echo ERRO: falta modelos\modelo_final.pt
  exit /b 1
)

pyinstaller --noconfirm --clean --windowed --onedir ^
  --name TriagemTB ^
  --paths . ^
  --collect-all torch ^
  --collect-all torchvision ^
  --collect-all torchxrayvision ^
  --hidden-import pydicom.pixels ^
  --hidden-import PIL._tkinter_finder ^
  --add-data "modelos\modelo_final.pt;modelos" ^
  app.py

if errorlevel 1 exit /b 1

echo.
echo Pronto: dist\TriagemTB\TriagemTB.exe
echo Copie a pasta inteira dist\TriagemTB para o outro PC.
echo Na primeira execucao a PSPNet pode baixar pesos (precisa de internet).