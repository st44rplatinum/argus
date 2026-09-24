# Download the TotalSegmentator CT dataset, resumably.
#
# 23.58 GB, CC BY 4.0, 1228 CTs with 117 structures each. This is the dataset
# P5, P6 and P8 all need, and the only one on the roadmap that is both large
# and ungated.
#
# curl -C - resumes from wherever it stopped, so a dropped connection costs
# minutes rather than the whole night. Safe to re-run.
#
# Note the absence of `2>&1` and of ErrorActionPreference = Stop. In
# PowerShell 5.1, redirecting a native command's stderr wraps every line in an
# ErrorRecord, and curl writes its progress meter to stderr — together those
# killed this script on the first line of output. --silent removes the meter;
# --show-error keeps real failures.

$url  = 'https://zenodo.org/api/records/10047292/files/Totalsegmentator_dataset_v201.zip/content'
$dest = 'E:\radiology viewer\data\raw\Totalsegmentator_dataset_v201.zip'
$log  = 'E:\radiology viewer\data\raw\download.log'
$expected = 23581218285

New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null

function Note($message) {
    "[$(Get-Date -Format s)] $message" | Out-File $log -Append -Encoding utf8
}

Note "starting, target $dest"

& curl.exe --silent --show-error --location --continue-at - `
    --retry 20 --retry-delay 15 --retry-all-errors `
    --output $dest $url

$size = (Get-Item $dest -ErrorAction SilentlyContinue).Length
Note "curl exited $LASTEXITCODE, $([math]::Round($size/1GB,2)) GB on disk"

if ($size -eq $expected) {
    Note "complete - size matches the published figure"
} else {
    Note "INCOMPLETE - re-run this script to resume from $size bytes"
}
