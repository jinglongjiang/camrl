#!/bin/bash
# Wait for the GPU to be genuinely free, then run the 2x2. Watches only --
# it never signals another user's job.
#
# Three conditions, all required, on THREE consecutive checks: an empty
# compute-app list (the authoritative resource view), no train_interface.py
# process (the ECG job's own name, whose dataloader workers come and go
# between epochs and would otherwise look like a gap), and enough free
# memory. The consecutive requirement is what stops a momentary lull between
# ECG epochs from launching a 2-hour experiment.
set -u
LOG=/root/queue_2x2.log
GATEDIR=/root/workspace/nav_data/mamba/camrl/bdvl_v6/CrowdNav/runs/v2
ARMED=$GATEDIR/QUEUE_ARMED
RUNNING=$GATEDIR/QUEUE_RUNNING
NEED_FREE_MIB=12288
CLEAR_STREAK_REQUIRED=3
streak=0
echo "$(date -Is) queue armed: need empty GPU + no train_interface + >=${NEED_FREE_MIB} MiB, x${CLEAR_STREAK_REQUIRED}" >> $LOG
while true; do
  [ -f "$ARMED" ] || { echo "$(date -Is) gate file removed -- standing down" >> $LOG; exit 0; }
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  ECG=$(pgrep -fc "python train_interface.py" || true)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${APPS:-1}" -eq 0 ] && [ "${ECG:-1}" -eq 0 ] && [ "${FREE:-0}" -ge "$NEED_FREE_MIB" ]; then
    streak=$((streak+1))
    echo "$(date -Is) clear ($streak/$CLEAR_STREAK_REQUIRED) free=${FREE}MiB" >> $LOG
    if [ "$streak" -ge "$CLEAR_STREAK_REQUIRED" ]; then
      # Atomic claim: mv succeeds for exactly one waiter, so a second queue
      # process started by mistake finds no gate and stands down instead of
      # launching a duplicate experiment onto the same GPU.
      if ! mv "$ARMED" "$RUNNING" 2>/dev/null; then
        echo "$(date -Is) another process claimed the gate -- standing down" >> $LOG; exit 0
      fi
      echo "$(date -Is) claimed the gate; starting 2x2" >> $LOG
      bash /root/run_2x2.sh >> $LOG 2>&1
      rc=$?
      echo "$(date -Is) 2x2 runner exited $rc" >> $LOG
      exit $rc
    fi
  else
    [ "$streak" -ne 0 ] && echo "$(date -Is) streak reset (apps=$APPS ecg=$ECG free=${FREE}MiB)" >> $LOG
    streak=0
  fi
  sleep 60
done
