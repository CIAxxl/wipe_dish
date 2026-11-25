#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
rdp="$repo/third_party/reactive_diffusion_policy"
configs="$repo/configs/real_world/wipe_v2"
export PYTHONPATH="$repo:$rdp${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1

stage="${1:?Usage: run.sh <at|ldp|dp|uvip-init|cache|prepare|preflight|teacher|labels|audit|student> ...}"
shift
case "$stage" in
    at|ldp|dp)
        : "${WIPE_V2_DATA:?Set WIPE_V2_DATA to the extracted dataset directory}"
        output="${1:?Pass the output directory}"
        shift
        if [[ "$stage" == ldp ]]; then
            : "${WIPE_V2_AT:?Set WIPE_V2_AT to your selected AT checkpoint, e.g. checkpoints/best.ckpt}"
            [[ -f "$WIPE_V2_AT" ]]
        fi
        exec python -u "$rdp/train.py" --config-dir "$configs" --config-name "$stage" \
            "hydra.run.dir=$output" "$@"
        ;;
    uvip-init)
        exec python -u "$repo/scripts/real_world/wipe_v2/prepare_uvip.py" "$@"
        ;;
    audit)
        exec python -u "$repo/scripts/real_world/audit_uvip_labels.py" "$@"
        ;;
    cache|prepare|preflight|teacher|labels|student)
        exec python -u "$repo/scripts/real_world/run_uvip.py" "$stage" "$@"
        ;;
    *)
        echo "Unknown Wipe V2 stage: $stage" >&2
        exit 2
        ;;
esac
