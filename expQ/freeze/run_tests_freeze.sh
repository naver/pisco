#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/run_tests_freeze_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/run_tests_freeze_%j.out
#
# Campaign `freeze`: the CPU unit tests (no model, no data) run against the tree in DIR (default: the repo).
# DIR = expQ/freeze/split_stage tests the split-freeze patch on a staged copy BEFORE it is installed in pisco/.
#   expmon launch --exp freeze --stage test expQ/freeze/run_tests_freeze.sh [DIR]
source ~/.bashrc
ROOT=/beegfs/scratch/user/hdejean/pisco
DIR=$(realpath "${1:-$ROOT}")
cd "$DIR" || exit 2
echo "FACTORS kind=unit_tests tree=$(basename "$DIR")"
git -C "$ROOT" rev-parse --short HEAD
md5sum "$DIR/pisco/model.py" "$DIR/test_freeze.py"
export PYTHONPATH="$DIR"
pixi run --manifest-path "$ROOT/pixi.toml" python -c "import pisco.model as m; print('pisco.model from', m.__file__)"
RC=0
for t in test_positions.py test_mm_collators.py test_freeze.py; do
  echo "=== $t ==="; pixi run --manifest-path "$ROOT/pixi.toml" python -u "$DIR/$t" || RC=1
done
echo "=== EXIT CODE $RC ==="; exit $RC
