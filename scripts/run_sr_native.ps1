param(
    [Parameter(Mandatory=$true)][string]$RunTag,
    [switch]$Resume
)
$ErrorActionPreference='Stop'
if($RunTag -notmatch '^sr_native_[0-9]{8}$'){throw 'Invalid run tag'}
$taskRoot=Split-Path -Parent $PSScriptRoot
$batchRoot=Join-Path $taskRoot "report/sr_native/$RunTag"
$manifest=Get-Content -LiteralPath (Join-Path $batchRoot 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$mutex=New-Object System.Threading.Mutex($false,'Local\HSI_SR_NATIVE_QUEUE')
if(-not $mutex.WaitOne(0)){throw 'Another SR queue is active'}
$utf8=New-Object System.Text.UTF8Encoding($false)
$state=[ordered]@{state='starting';total=27;reused=2;new_completed=0;completed=2;queue_pid=$PID;training_pid=$null;current=$null;updated_at=''}
function Save-State {
    $state.updated_at=(Get-Date).ToString('o')
    $temp=Join-Path $batchRoot 'state.tmp'
    [IO.File]::WriteAllText($temp,($state | ConvertTo-Json -Depth 8),$utf8)
    Move-Item -LiteralPath $temp -Destination (Join-Path $batchRoot '运行状态.json') -Force
}
function Assert-PinnedFiles {
    foreach($collection in @($manifest.source_and_config_sha256,$manifest.data_sha256)){
        foreach($item in $collection.PSObject.Properties){
            $actual=(Get-FileHash -LiteralPath (Join-Path $taskRoot $item.Name) -Algorithm SHA256).Hash.ToLower()
            if($actual -ne $item.Value){throw "Pinned file changed: $($item.Name)"}
        }
    }
}
try {
    Set-Location -LiteralPath $taskRoot
    $env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
    $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUNBUFFERED='1'
    # Separate clean processes use the original lab randomness/attention defaults.
    Remove-Item Env:CUBLAS_WORKSPACE_CONFIG -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path (Join-Path $batchRoot 'completed') | Out-Null
    foreach($job in $manifest.jobs){
        if($job.reused){continue}
        $marker=Join-Path $batchRoot "completed/$($job.id).json"
        if(Test-Path -LiteralPath $marker){
            if(-not $Resume){throw 'Batch already has completed jobs; use -Resume to skip them'}
            $done=Get-Content -LiteralPath $marker -Raw -Encoding UTF8 | ConvertFrom-Json
            if((Get-FileHash -LiteralPath $job.checkpoint -Algorithm SHA256).Hash.ToLower() -ne $done.checkpoint_sha256 -or (Get-FileHash -LiteralPath $job.report -Algorithm SHA256).Hash.ToLower() -ne $done.report_sha256){throw 'Completed output hash changed'}
            $state.new_completed++; $state.completed++; continue
        }
        $others=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(train\.py|train_all\.py|run_batch\.py|train_transmixer_).*'})
        if($others.Count){throw 'Another native training process is active; refusing duplicate launch'}
        foreach($path in @($job.stdout,$job.stderr,$job.native_log,$job.report,$job.checkpoint)){
            if(Test-Path -LiteralPath $path){throw "Unfinished output exists; review before rerunning: $path"}
        }
        Assert-PinnedFiles
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $job.stdout) | Out-Null
        $state.state='launching';$state.current=$job;$state.training_pid=$null;Save-State
        $quotedConfig='"'+$job.config+'"'
        $process=Start-Process -FilePath $manifest.python -ArgumentList @('-X','utf8','-u','scripts/train.py','--config',$quotedConfig) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $job.stdout -RedirectStandardError $job.stderr -PassThru
        $state.state='training';$state.training_pid=$process.Id;Save-State
        $process.WaitForExit();$process.Refresh()
        if($process.ExitCode -ne 0){throw "Native training failed: $($job.id), exit=$($process.ExitCode)"}
        foreach($path in @($job.checkpoint,$job.report)){if(-not (Test-Path -LiteralPath $path)){throw "Missing native output: $path"}}
        Assert-PinnedFiles
        $done=[ordered]@{id=$job.id;finished_at=(Get-Date).ToString('o');checkpoint_sha256=(Get-FileHash -LiteralPath $job.checkpoint -Algorithm SHA256).Hash.ToLower();report_sha256=(Get-FileHash -LiteralPath $job.report -Algorithm SHA256).Hash.ToLower()}
        [IO.File]::WriteAllText($marker,($done | ConvertTo-Json),$utf8)
        & $manifest.python -X utf8 scripts/prepare_sr_native.py summarize --run-tag $RunTag
        if($LASTEXITCODE -ne 0){throw 'Native report summary failed; outputs preserved'}
        $state.new_completed++;$state.completed++;$state.training_pid=$null;Save-State
    }
    $state.state='completed';$state.current=$null;$state.training_pid=$null;Save-State
} catch {
    $state.state='failed';$state.error=$_.Exception.Message;Save-State
    throw
} finally {
    $mutex.ReleaseMutex();$mutex.Dispose()
}
