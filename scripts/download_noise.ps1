param(
    [ValidateSet("SPIB", "QUT", "ALL")]
    [string[]]$Datasets = @("ALL")
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$noiseRoot = Join-Path $repoRoot "data/noise"
$downloadsRoot = Join-Path $noiseRoot "downloads"
$spibRoot = Join-Path $noiseRoot "SPIB"
$qutExtractRoot = Join-Path $noiseRoot "QUT-NOISE"
$qutDownloadsRoot = Join-Path $downloadsRoot "QUT-NOISE"

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile
    )
    if (Test-Path -LiteralPath $OutFile) {
        Write-Host "Skipping existing file $OutFile"
        return
    }
    $partialFile = "$OutFile.part"
    if (Test-Path -LiteralPath $partialFile) {
        Remove-Item -LiteralPath $partialFile -Force
    }
    Write-Host "Downloading $Url"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partialFile
    Move-Item -LiteralPath $partialFile -Destination $OutFile -Force
}

function Expand-ZipWithRetry {
    param(
        [string]$Url,
        [string]$ArchivePath,
        [string]$Destination
    )

    try {
        Write-Host "Extracting $ArchivePath"
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $Destination -Force
    }
    catch {
        Write-Warning "Archive appears incomplete or corrupt: $ArchivePath"
        if (Test-Path -LiteralPath $ArchivePath) {
            Remove-Item -LiteralPath $ArchivePath -Force
        }
        Download-File -Url $Url -OutFile $ArchivePath
        Write-Host "Re-extracting $ArchivePath"
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $Destination -Force
    }
}

$selection = @($Datasets)
if ($selection -contains "ALL") {
    $selection = @("SPIB", "QUT")
}

if ($selection -contains "SPIB") {
    Ensure-Dir $spibRoot

    $spibFiles = @(
        "white.mat",
        "pink.mat",
        "babble.mat",
        "factory1.mat",
        "factory2.mat",
        "buccaneer1.mat",
        "buccaneer2.mat",
        "f16.mat",
        "destroyerengine.mat",
        "destroyerops.mat",
        "leopard.mat",
        "m109.mat",
        "machinegun.mat",
        "volvo.mat",
        "hfchannel.mat"
    )

    foreach ($file in $spibFiles) {
        $url = "http://spib.linse.ufsc.br/data/noise/$file"
        $outFile = Join-Path $spibRoot $file
        Download-File -Url $url -OutFile $outFile
    }
}

if ($selection -contains "QUT") {
    Ensure-Dir $qutDownloadsRoot
    Ensure-Dir $qutExtractRoot

    $qutArchives = @(
        "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/8342a090-89e7-4402-961e-1851da11e1aa/download/qutnoise.zip",
        "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip",
        "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip",
        "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip",
        "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip",
        "https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip"
    )

    foreach ($url in $qutArchives) {
        $fileName = Split-Path $url -Leaf
        $archivePath = Join-Path $qutDownloadsRoot $fileName
        Download-File -Url $url -OutFile $archivePath
        Expand-ZipWithRetry -Url $url -ArchivePath $archivePath -Destination $qutExtractRoot
    }
}

Write-Host "Noise dataset setup completed."
