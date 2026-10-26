# Gera TriagemTB.exe (pasta dist\TriagemTB\)
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv nao encontrado: $py" }
if (-not (Test-Path "modelos\modelo_final.pt")) { throw "Falta modelos\modelo_final.pt" }
if (-not (Test-Path "modelos\pspnet_chestxray_best_model_4.pth")) {
  throw "Falta modelos\pspnet_chestxray_best_model_4.pth (copie de ~/.torchxrayvision/models_data/)"
}

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
  --add-data "modelos\pspnet_chestxray_best_model_4.pth;modelos" `
  app.py

Write-Host "`nPronto: dist\TriagemTB\TriagemTB.exe"
Write-Host "Distribua a pasta inteira dist\TriagemTB (nao so o .exe)."
Write-Host "Pesos da PSPNet ja vao no pacote (sem download na 1a analise)."