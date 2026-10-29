param([string]$RunDirectory='report/weekend_distillation/weekend_kd_20261025')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$runPath = (Resolve-Path -LiteralPath (Join-Path $repo $RunDirectory)).Path
$resultsPath = Join-Path $runPath 'results.json'
$resultData = Get-Content -LiteralPath $resultsPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($resultData.status -ne 'completed' -or $resultData.jobs.Count -ne 7) { throw 'All seven training jobs must complete first.' }
$runtime = 'C:/Users/mmm/.cache/codex-runtimes/codex-primary-runtime/dependencies'
$docPython = Join-Path $runtime 'python/python.exe'
$docNode = Join-Path $runtime 'node/bin/node.exe'
$docSkill = 'C:/Users/mmm/.codex/plugins/cache/openai-primary-runtime/documents/26.1024.11018/skills/documents'
$title = 'TransMixer 周末融合与知识蒸馏实验全面报告'
$docx = Join-Path $runPath ($title+'.docx')
$pdf = Join-Path $runPath ($title+'.pdf')
Push-Location $repo
try {
    & python -X utf8 scripts/summarize_weekend_distillation.py --run $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Saved-results audit failed.' }
    & python -X utf8 scripts/plot_weekend_results.py --run $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Chart generation failed.' }
    $markerRecord = Join-Path $runPath '文档制作标记.log'
    if (-not (Test-Path -LiteralPath $markerRecord)) {
        $markerOutput = & $docNode (Join-Path $docSkill 'container_tools/mark_artifact_operation_started.mjs') --operation-kind create --expected-output-count 1 --output-format docx 2>&1
        if ($LASTEXITCODE -ne 0) { throw 'Document operation marker failed.' }
        $markerOutput | Set-Content -LiteralPath $markerRecord -Encoding UTF8
    }
    & $docPython -X utf8 scripts/build_weekend_report.py --run $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Word report build failed.' }
    $canonicalOutput = Join-Path $runPath 'canonical_render'
    & $docPython -X utf8 (Join-Path $docSkill 'render_docx.py') $docx --output_dir $canonicalOutput --emit_pdf *> (Join-Path $runPath '文档渲染日志.txt')
    $canonicalPdf = Join-Path $canonicalOutput ($title+'.pdf')
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $canonicalPdf)) {
        Copy-Item -LiteralPath $canonicalPdf -Destination $pdf -Force
    } else {
        & (Join-Path $PSScriptRoot 'export_weekend_word.ps1') -DocumentPath $docx
    }
    & $docPython -X utf8 scripts/qa_weekend_report.py --run $runPath
    if ($LASTEXITCODE -ne 0) { throw 'Rendered report needs repair; inspect 页面核验.' }
    Write-Output 'Report rendered. Inspect EVERY page PNG and record the visual review before notifying the user.'
} finally {
    Pop-Location
}
