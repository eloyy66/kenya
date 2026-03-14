# Levanta servicios face y tf en dos ventanas de PowerShell.
param(
  [string]$FaceVenv = "$env:USERPROFILE\venvs\mostacho_face",
  [string]$TfVenv = "$env:USERPROFILE\venvs\mostacho_tf"
)

# Comando para iniciar face service.
$FaceCmd = "& '$FaceVenv\Scripts\Activate.ps1'; uvicorn mostacho.services.face_api:app --host 127.0.0.1 --port 8001"
# Comando para iniciar tf service.
$TfCmd = "& '$TfVenv\Scripts\Activate.ps1'; uvicorn mostacho.services.tf_api:app --host 127.0.0.1 --port 8002"

# Abre nueva ventana para Face Service.
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FaceCmd
# Abre nueva ventana para TF Service.
Start-Process powershell -ArgumentList "-NoExit", "-Command", $TfCmd

# Mensaje final.
Write-Host "Servicios iniciados en nuevas ventanas PowerShell."
