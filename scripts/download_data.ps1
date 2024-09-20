param(
    [string]$task
)

if ($task -eq "is2re") {
    Write-Host "Downloading IS2RE 10k"
    $dataDir = "data\is2re"
    $trainDir = Join-Path -Path $dataDir -ChildPath "train"
    $valDir = Join-Path -Path $dataDir -ChildPath "val"

    New-Item -Path $trainDir -ItemType Directory -Force | Out-Null
    New-Item -Path $valDir -ItemType Directory -Force | Out-Null

    Push-Location -Path $trainDir

    $trainFileUrl = "https://drive.google.com/uc?id=19b7kOXBiHkhr_gzo-0iITIDN012puN-1"
    $trainFilePath = Join-Path -Path $trainDir -ChildPath "is2re_10k.lmdb"
    if (-not (Test-Path -Path $trainFilePath)) {
        Invoke-WebRequest -Uri $trainFileUrl -OutFile "is2re_10k.lmdb"
    }

    Set-Location -Path (Join-Path -Path "..\..\.." -ChildPath $valDir)

    $valIdUrl = "https://drive.google.com/uc?id=1ALTdSZuoc1KRmuf5KEDWFz6yciqv2zJq"
    $valIdPath = Join-Path -Path $valDir -ChildPath "val_id.lmdb"
    if (-not (Test-Path -Path $valIdPath)) {
        Invoke-WebRequest -Uri $valIdUrl -OutFile "val_id.lmdb"
    }

    $valOodBothUrl = "https://drive.google.com/uc?id=1r4yL1fRNhdFcOf7r7EwhbvOHBPhfM88F"
    $valOodBothPath = Join-Path -Path $valDir -ChildPath "val_ood_both.lmdb"
    if (-not (Test-Path -Path $valOodBothPath)) {
        Invoke-WebRequest -Uri $valOodBothUrl -OutFile "val_ood_both.lmdb"
    }

    Pop-Location

} elseif ($task -eq "dense") {
    Write-Host "Downloading OC20 Dense"
    $denseDir = "data\dense"

    New-Item -Path $denseDir -ItemType Directory -Force | Out-Null

} else {
    Write-Host "Usage: download_data.ps1 -task [is2re|oc20]"
}
