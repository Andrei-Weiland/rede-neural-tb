# Gera TriagemTB.exe (pasta dist\TriagemTB\)
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv nao encontrado: $py" }
if (-not (Test-Path "modelos\modelo_final.pt")) { throw "Falta modelos\modelo_final.pt" }

& $py -m pip install -U pyinstaller
& $py -m PyInstaller --noconfirm --clean --windowed --onedir `
  --name TriagemTB `
  --paths . `
  --collect-all torch `
  --collect-all torchvision `
  --collect-all torchxrayvision `
  --hidden-import pydicom.pixels `
  --hidden-import PIL._tkinter_finder `
  --add-data "modelos\modelo_final.pt;modelos" `
  app.py

Write-Host "`nPronto: dist\TriagemTB\TriagemTB.exe"
Write-Host "Distribua a pasta inteira dist\TriagemTB (nao so o .exe)."
