#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Thin wrapper that locks GPU/DRAM clocks before the Python server starts
# and guarantees unlock on exit (even SIGTERM/crash).  All real logic lives
# in sol_execbench.core.bench.clock_lock.
# ---------------------------------------------------------------------------

lock_clocks() {
    if python -c "
import logging, torch
logging.basicConfig(level=logging.INFO, format='%(message)s')
from sol_execbench.core.bench.clock_lock import lock_clocks
device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
if not device:
    raise SystemExit('No CUDA device detected')
print(f'Detected GPU: {device}')
if not lock_clocks(device):
    raise SystemExit(1)
"; then
        export SOL_EXECBENCH_CLOCKS_LOCKED=1
    else
        echo "WARNING: Clock locking failed — proceeding unlocked"
        export SOL_EXECBENCH_CLOCKS_LOCKED=0
    fi
}

cleanup() {
    if [ "${SOL_EXECBENCH_CLOCKS_LOCKED}" = "1" ]; then
        python -c "
from sol_execbench.core.bench.clock_lock import unlock_clocks
print('Unlocking clocks...')
unlock_clocks()
print('Clocks unlocked')
"
    fi
}

# check if flashinfer-trace directory is mounted
if [ ! -d "${FLASHINFER_TRACE_DIR}" ]; then
    echo "ERROR: FLASHINFER_TRACE_DIR is not mounted"
    echo "       Mount the flashinfer-trace directory into the container and set the env var."
    echo "       The easiest way is to use the helper script:"
    echo ""
    echo "         ./scripts/run_docker.sh -- <command>"
    echo ""
    echo "       Or manually:"
    echo ""
    echo "         docker run \\"
    echo "           -v /path/to/flashinfer-trace:/sol-execbench/data/flashinfer-trace \\"
    echo "           -e FLASHINFER_TRACE_DIR=/sol-execbench/data/flashinfer-trace \\"
    echo "           sol-execbench:latest <command>"
    exit 1
fi

lock_clocks
trap 'cleanup' EXIT
trap 'exit' TERM INT

"$@"
