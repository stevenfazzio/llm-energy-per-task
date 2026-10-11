#!/usr/bin/env bash
# Runs ON THE POD, before anything else: has the pod terminate itself after a
# number of minutes, so a hung run or a closed laptop cannot keep it billing.
# Terminating deletes the pod's disk, so copy results off before it fires.
#
#   ssh -p PORT root@HOST 'bash -s -- 90'  < scripts/pod_guard.sh   # arm
#   ssh -p PORT root@HOST 'bash -s -- now' < scripts/pod_guard.sh   # terminate at once
set -euo pipefail

AFTER="${1:?minutes until the pod terminates itself, or 'now'}"

# RunPod gives the pod's id, and a key for it, to the container's first process
# and not to SSH shells, so they are read from there.
from_init() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p"; }
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-$(from_init RUNPOD_POD_ID)}"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(from_init RUNPOD_API_KEY)}"

# Never report a guard that could not fire.
if [ -z "$RUNPOD_POD_ID" ] || [ -z "$RUNPOD_API_KEY" ]; then
  echo "NOT ARMED: no pod id or key in the environment of PID 1" >&2
  exit 1
fi
# The pod's key cannot read pods, so there is no asking whether this one is
# visible. Asked to terminate a pod that does not exist, a key that may
# terminate is told it was not found, and one that may not is refused.
probe="$(runpodctl remove pod not-a-pod 2>&1 || true)"
if [[ "$probe" != *"not found"* ]]; then
  echo "NOT ARMED: runpodctl is missing here, or its key may not terminate pods: $probe" >&2
  exit 1
fi

if [ "$AFTER" = now ]; then
  exec runpodctl remove pod "$RUNPOD_POD_ID"
fi

# Detached from this SSH session, which would otherwise take it along on closing.
setsid bash -c "sleep $((AFTER * 60)); runpodctl remove pod '$RUNPOD_POD_ID'" \
  > /tmp/pod_guard.log 2>&1 < /dev/null &
echo "armed: pod $RUNPOD_POD_ID terminates itself at $(date -u -d "+$AFTER min" +%H:%M) UTC"
