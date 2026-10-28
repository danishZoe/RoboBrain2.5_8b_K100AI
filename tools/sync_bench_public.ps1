# Pull public-benchmark artifacts (text only; images are not mirrored).
# ASCII-only comments: PowerShell 5.1 reads BOM-less files as ANSI.
$ErrorActionPreference = 'Continue'
$base = 'D:\01postStudy\07obsidian_space\workspace_v2\02_algorithm\HG\proj_v2\agent_work\adaptation'
$remote = 'dcu_1026:/root/private_data/workspace/robo'
$opts = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=25', '-o', 'ServerAliveInterval=15')

function Pull-File([string]$rpath, [string]$lpath) {
    $dir = Split-Path -Parent $lpath
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    & scp @opts "$remote/$rpath" $lpath 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output "WARN: pull failed $rpath" } else { Write-Output "OK: $rpath" }
}

$root = Join-Path $base 'samples\bench_public'

# 1) summaries / manifests / checks
foreach ($f in @('manifest/data_manifest.json', 'manifest/data_manifest.md',
                 'refspatial/summary.json', 'cvbench/summary.json',
                 'tracespatial/summary.json', 'stability/stability.json',
                 'rescore_check.json', 'summary_table.md',
                 'accuracy_regression.md', 'accuracy_regression.json',
                 'verify_3d_merge.json',
                 'cvbench_protocol/protocol_compare.json',
                 'cvbench_upscale/upscale_compare.json',
                 'cvbench_prompt/prompt_compare.json',
                 'cvbench_paired_tests.json',
                 'cvbench_answer_format_audit.json',
                 'eval_timing.md', 'eval_timing.json',
                 'cvbench/task_breakdown.json',
                 'cvbench_count_samples/manifest.json')) {
    Pull-File "outputs/bench_public/$f" (Join-Path $root ($f -replace '/', '\'))
}

# 1b) per-variant TraceSpatial chunked scoring results (2D + 3D)
foreach ($v in @('official', 'upstream')) {
    foreach ($t in @('2D', '3D')) {
        Pull-File "outputs/bench_public/tracespatial/$v/scores_3dchunk_$t.json" `
                 (Join-Path $root "tracespatial\$v\scores_3dchunk_$t.json")
    }
}

# 2) candidate run summaries
$cand = Join-Path $base 'samples\bench_public_noneager'
foreach ($f in @('refspatial/summary.json', 'cvbench/summary.json',
                 'tracespatial/summary.json', 'rescore_check.json')) {
    Pull-File "outputs/bench_public_noneager/$f" (Join-Path $cand ($f -replace '/', '\'))
}
foreach ($v in @('official', 'upstream')) {
    foreach ($t in @('2D', '3D')) {
        Pull-File "outputs/bench_public_noneager/tracespatial/$v/scores_3dchunk_$t.json" `
                 (Join-Path $cand "tracespatial\$v\scores_3dchunk_$t.json")
    }
}

# 2b) noise floor
$noise = Join-Path $base 'samples\bench_public_noisecheck'
foreach ($f in @('noise_floor.json', 'refspatial/summary.json')) {
    Pull-File "outputs/bench_public_noisecheck/$f" (Join-Path $noise ($f -replace '/', '\'))
}

# 3) logs
foreach ($d in @('logs/public_bench_baseline-eager', 'logs/public_bench_noneager-candidate',
                 'logs/ts_score_2d.log', 'logs/ts_score_3d.log',
                 'logs/ts_score_3d_96.log',
                 'logs/verify_3d_merge.log', 'logs/exp_cvbench_protocol.log',
                 'logs/exp_cvbench_upscale.log', 'logs/cvbench_upscale.log',
                 'logs/cvbench_prompt.log', 'logs/cvbench_up2_full.log',
                 'logs/watch_up2_full.log', 'logs/chain_cvbench_fast.log')) {
    & scp @opts -r "$remote/$d" (Join-Path $base 'logs') 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output "WARN: pull failed $d" } else { Write-Output "OK: $d" }
}

# 4) reports 16/17/00 are authored LOCALLY -> push to remote (never pull, would overwrite edits)
foreach ($f in @('reports/00_RUNBOOK.md', 'reports/16_public_bench_results.md',
                 'reports/17_accuracy_regression.md')) {
    $src = Join-Path $base ($f -replace '/', '\')
    if (-not (Test-Path -LiteralPath $src)) { Write-Output "WARN: local missing $f"; continue }
    & scp @opts $src "$remote/$f" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output "WARN: push failed $f" } else { Write-Output "OK(push): $f" }
}

Write-Output ""
Write-Output "===== local public-bench mirror ====="
Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
    Write-Output ("  {0,-52} {1,10} B" -f $_.Name, $_.Length)
}
