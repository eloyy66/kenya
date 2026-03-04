# Crea entorno virtual para TF Service en Windows.
param(
  [string]$VenvPath = "$env:USERPROFILE\venvs\mostacho_tf"
)

# Calcula rutas base.
$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$MostachoRoot = Join-Path $RepoRoot "mostacho"

# Crea entorno virtual.
py -3.10 -m venv $VenvPath
# Activa entorno.
& "$VenvPath\Scripts\Activate.ps1"
# Actualiza herramientas base.
python -m pip install --upgrade pip setuptools wheel
# Instala dependencias TF.
pip install -r "$MostachoRoot\requirements\tf_service.txt"
# Instala paquete local editable.
pip install -e $MostachoRoot

# Mensaje final.
Write-Host "Entorno tf listo en: $VenvPath"
