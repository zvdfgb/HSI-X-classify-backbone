param([Parameter(Mandatory=$true)][string]$DocumentPath)
$ErrorActionPreference = 'Stop'
$sourceDoc = (Resolve-Path -LiteralPath $DocumentPath).Path
$targetPdf = [IO.Path]::ChangeExtension($sourceDoc, '.pdf')
$weekendWord = $null
$weekendDoc = $null
try {
    $weekendWord = New-Object -ComObject Word.Application
    $weekendWord.Visible = $false
    $weekendWord.DisplayAlerts = 0
    $weekendDoc = $weekendWord.Documents.Open($sourceDoc, $false, $true)
    $weekendDoc.Repaginate()
    $weekendDoc.ExportAsFixedFormat($targetPdf, 17)
    Write-Output $targetPdf
} finally {
    if ($null -ne $weekendDoc) { $weekendDoc.Close(0) }
    if ($null -ne $weekendWord) { $weekendWord.Quit() }
}
