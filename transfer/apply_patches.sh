#!/usr/bin/env bash
# Bootstrap the patched repos on a fresh machine (e.g. the H100).
# Run from anywhere; clones into $HOME. Re-run safe-ish (will skip existing clones).
set -e
HOME_DIR="${HOME}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../robocasa_experiments/transfer

clone() {  # $1=url $2=dir [$3=branch]
  if [ -d "$HOME_DIR/$2/.git" ]; then echo "skip $2 (exists)"; return; fi
  if [ -n "$3" ]; then git clone -b "$3" "$1" "$HOME_DIR/$2"; else git clone "$1" "$HOME_DIR/$2"; fi
}

echo "=== clone public forks ==="
clone https://github.com/ARISE-Initiative/robosuite robosuite            # use master
clone https://github.com/robocasa/robocasa robocasa
clone https://github.com/robocasa-benchmark/diffusion_policy diffusion_policy
clone https://github.com/robocasa-benchmark/robomimic robomimic          # fork w/ VisualCoreLanguageConditioned
clone https://github.com/NVlabs/mimicgen mimicgen experimental/robocasa  # optional (data-gen)

echo "=== apply patches ==="
git -C "$HOME_DIR/diffusion_policy" apply "$HERE/patches/diffusion_policy.patch" && echo "  diffusion_policy patched"
git -C "$HOME_DIR/robomimic"        apply "$HERE/patches/robomimic.patch"        && echo "  robomimic patched"
git -C "$HOME_DIR/mimicgen"         apply "$HERE/patches/mimicgen.patch" 2>/dev/null && echo "  mimicgen patched" || echo "  (mimicgen patch skipped)"

echo "=== place the standalone scripts ==="
cp "$HERE/new_files/run_dp_smoketest.py" "$HOME_DIR/diffusion_policy/" 2>/dev/null || true
mkdir -p "$HOME_DIR/Isaac-GR00T/scripts" 2>/dev/null || true
cp "$HERE/new_files/run_groot_smoketest.py" "$HOME_DIR/Isaac-GR00T/scripts/" 2>/dev/null || true
mkdir -p "$HOME_DIR/openpi/examples/robocasa" 2>/dev/null || true
cp "$HERE/new_files/run_pi0_client.py" "$HERE/new_files/record_pi0_demos.py" "$HOME_DIR/openpi/examples/robocasa/" 2>/dev/null || true

echo "=== install Claude memory (so a new session has full context) ==="
MEMDST="$HOME_DIR/.claude/projects/-home-asurite-ad-asu-edu-xinyua11-robocasa/memory"
mkdir -p "$MEMDST"
cp -n "$HERE/memory/"*.md "$MEMDST/" 2>/dev/null || true
echo "  memory at $MEMDST"

cat <<'NEXT'

=== NEXT (not automated — see H100_HANDOFF.md) ===
1. Create the py3.11 training env; pip install -e the forks (+ hydra-core omegaconf zarr
   numcodecs matplotlib egl_probe accelerate transformers). See memory dp-model-smoketest-status.
2. Re-download big artifacts on this machine:
     python -m robocasa.scripts.download_kitchen_assets
     # checkpoints from HF robocasa/robocasa365_checkpoints (huggingface_hub snapshot_download)
     # datasets: python -m robocasa.scripts.download_datasets --tasks PickPlaceCounterToSink --source human --split pretrain
3. Train: cd ~/diffusion_policy && python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/<soup>
NEXT
