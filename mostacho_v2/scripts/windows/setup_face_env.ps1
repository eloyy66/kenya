# Crea entorno virtual para Face Service en Windows.
param(
  [string]$VenvPath = "$env:USERPROFILE\venvs\mostacho_face"
)

# Calcula ruta del repo y proyecto.
$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$MostachoRoot = Join-Path $RepoRoot "mostacho"

# Crea entorno virtual.
py -3.10 -m venv $VenvPath
# Activa entorno.
& "$VenvPath\Scripts\Activate.ps1"
# Actualiza herramientas base.
python -m pip install --upgrade pip setuptools wheel
# Instala dependencias de face.
pip install -r "$MostachoRoot\requirements\face_service.txt"
# Instala paquete local editable.
pip install -e $MostachoRoot

# Mensaje final.
Write-Host "Entorno face listo en: $VenvPath"
