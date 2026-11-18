# Run the hip-gpt2 GPT-2 engine on your NVIDIA GPU from Windows, through Docker (no Linux install needed).
#
#   .\scripts\gpt2.ps1 "Once upon a time"                      # 64 tokens, temperature 0.8
#   .\scripts\gpt2.ps1 "The meaning of life is" -Tokens 100 -Temperature 0
#   .\scripts\gpt2.ps1 -Chat                                   # type prompts in a loop; empty line quits
#
# First run builds the Docker image, the engine and downloads GPT-2 (~500 MB); later runs start in seconds.
# GPT-2 small is a 2019 text-completion model, not a chat assistant: it continues whatever you type.
param(
    [Parameter(Position = 0)] [string]$Prompt = "Once upon a time",
    [int]$Tokens = 64,
    [double]$Temperature = 0.8,
    [int]$TopK = 40,
    [switch]$Chat
)
$repo = (Resolve-Path "$PSScriptRoot\..").Path
$mount = "${repo}:/work"
$noise = '^\s*$|^==|CUDA Version|NVIDIA C|NVIDIA Deep|icense|Container image|By pulling|https://|A copy'

docker image inspect hip-gpt2-nv *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Building the hip-gpt2-nv Docker image (one time, ~15 min)..."
    docker build -t hip-gpt2-nv -f "$repo\docker\Dockerfile.nv" "$repo\docker"
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

docker run --rm --gpus all -v $mount -v hipgpt2-build-nv:/build -v hipgpt2-venv:/venv -w /work hip-gpt2-nv `
    bash scripts/docker_gpt2_setup.sh | Where-Object { $_ -notmatch $noise }
if ($LASTEXITCODE -ne 0) { throw "setup failed (is Docker Desktop running?)" }

function Invoke-Gpt2([string]$p) {
    $a = @('--model', 'models/gpt2', '--prompt', $p, '--max-new-tokens', $Tokens,
           '--temperature', $Temperature, '--top-k', $TopK, '--seed', (Get-Random), '--stop-at-eot')
    Write-Host -NoNewline $p -ForegroundColor Cyan
    docker run --rm --gpus all -v $mount -v hipgpt2-build-nv:/build -w /work hip-gpt2-nv /build/build-nv/gpt2 @a |
        Where-Object { $_ -notmatch $noise -and $_ -notmatch '^load_ms' }
}

if ($Chat) {
    Write-Host "GPT-2 small on your GPU. It continues your text; it is not a chat assistant. Empty line quits."
    while ($true) {
        $p = Read-Host "`nprompt"
        if ([string]::IsNullOrWhiteSpace($p)) { break }
        Invoke-Gpt2 $p
    }
} else {
    Invoke-Gpt2 $Prompt
}
