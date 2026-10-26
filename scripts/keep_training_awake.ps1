param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId,
    [Parameter(Mandatory = $true)]
    [string]$StatusPath,
    [ValidateRange(1, 24)]
    [int]$MaximumHours = 12
)

$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class TrainingAwakeNative {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@

$training = Get-Process -Id $TrainingProcessId -ErrorAction Stop
$trainingStarted = $training.StartTime.ToUniversalTime().Ticks
$started = Get-Date
$deadline = $started.AddHours($MaximumHours)

function Write-GuardStatus([string]$State, [string]$Reason) {
    [ordered]@{
        state = $State
        reason = $Reason
        guard_pid = $PID
        training_pid = $TrainingProcessId
        started_at = $started.ToString('o')
        updated_at = (Get-Date).ToString('o')
        deadline = $deadline.ToString('o')
        scope = 'Prevents idle system sleep while training lives; allows display off; cannot block shutdown/restart/lid action.'
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

try {
    while ((Get-Date) -lt $deadline) {
        $current = Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
        if ($null -eq $current -or $current.StartTime.ToUniversalTime().Ticks -ne $trainingStarted) {
            Write-GuardStatus 'finished' 'Training process exited; sleep protection released.'
            break
        }
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, without ES_DISPLAY_REQUIRED.
        $previous = [TrainingAwakeNative]::SetThreadExecutionState([uint32]2147483649)
        if ($previous -eq 0) { throw 'Windows rejected SetThreadExecutionState.' }
        Write-GuardStatus 'active' 'Windows accepted the system-awake request.'
        Start-Sleep -Seconds 30
    }
    if ((Get-Date) -ge $deadline) {
        Write-GuardStatus 'expired' 'Maximum protection duration reached.'
    }
} catch {
    Write-GuardStatus 'failed' $_.Exception.Message
    throw
} finally {
    [void][TrainingAwakeNative]::SetThreadExecutionState([uint32]2147483648)
}
