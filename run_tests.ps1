param([string]$PythonExe = "python")
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    & $PythonExe -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Mechanism tests failed" }
    & $PythonExe -m dds_mamba smoke --out runs/smoke --device cpu
    if ($LASTEXITCODE -ne 0) { throw "End-to-end smoke test failed" }
} finally {
    Pop-Location
}
