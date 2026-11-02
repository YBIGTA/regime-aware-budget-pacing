#!/usr/bin/env bash
# 실험 2·3 수집이 끝난 뒤 VM 결과를 받고, 개수를 검증하고, 상태 테이블을 만든다.
#
#   experiments/data_collection/finish_collection.sh              # 받기 + 검증 + 상태 테이블
#   experiments/data_collection/finish_collection.sh --delete-vm  # 검증 통과 시 VM까지 삭제
#
# 다시 실행해도 된다. 이미 받은 실험은 다시 받지 않는다.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
VM_SCRIPT=experiments/data_collection/gcp_vm.sh
VM_NAME="${VM_NAME:-pacing-sim}"
ZONE="${ZONE:-asia-northeast3-a}"
RUNS="${RUNS:-outputs/simulation_runs}"
STATE_OUT="${STATE_OUT:-outputs/state_table/experiments_1_to_3}"

DELETE_VM=0
[ "${1:-}" = "--delete-vm" ] && DELETE_VM=1

# 실험 폴더와 기대 run 수
EXPECTED=(
  "experiment_1_1:180"
  "experiment_2_test_seeds:675"
  "experiment_3_static_mac:729"
  "experiment_3_static_vm:486"
)
VM_EXPERIMENTS=(experiment_2_test_seeds experiment_3_static_vm)

count_traces() { find "$RUNS/$1/runs" -name trace.csv 2>/dev/null | wc -l | tr -d ' '; }
count_completed() {
  # 새로 실행한 run은 completed, 이미 있어서 건너뛴 run은 skipped로 기록된다.
  # validation_status(10번째 열)가 passed인 행만 센다.
  awk -F, 'NR > 1 && ($6 == "completed" || $6 == "skipped") && $10 == "passed"' \
    "$RUNS/$1/manifest.csv" 2>/dev/null | wc -l | tr -d ' '
}

# 1. VM 결과 받기
missing=()
for experiment in "${VM_EXPERIMENTS[@]}"; do
  [ -f "$RUNS/$experiment/manifest.csv" ] || missing+=("$experiment")
done
if [ "${#missing[@]}" -gt 0 ]; then
  status="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --format='value(status)')"
  if [ "$status" != "RUNNING" ]; then
    echo "starting VM ($status)"
    gcloud compute instances start "$VM_NAME" --zone "$ZONE"
    until gcloud compute ssh "$VM_NAME" --zone "$ZONE" --quiet --command true >/dev/null 2>&1; do
      sleep 15
    done
  fi
  if ! gcloud compute ssh "$VM_NAME" --zone "$ZONE" --quiet --command "test -f ~/rabp/DONE"; then
    echo "VM has not finished yet (no DONE). check: $VM_SCRIPT status"
    exit 1
  fi
  "$VM_SCRIPT" download "${missing[@]}"
fi

# 2. 개수 검증
ok=1
for item in "${EXPECTED[@]}"; do
  experiment="${item%%:*}"
  expected="${item##*:}"
  traces="$(count_traces "$experiment")"
  completed="$(count_completed "$experiment")"
  echo "$experiment: trace.csv=$traces manifest=$completed expected=$expected"
  if [ "$traces" -ne "$expected" ] || [ "$completed" -ne "$expected" ]; then
    ok=0
  fi
done
if [ "$ok" -ne 1 ]; then
  echo "count mismatch. state table not built, VM not deleted."
  exit 1
fi

# 3. 상태 테이블 (학습 42~45, 검증 46, 평가 47~61)
.venv/bin/python src/preprocessing/build_state_table.py \
  --experiment-dir \
    "$RUNS/experiment_1_1" \
    "$RUNS/experiment_2_test_seeds" \
    "$RUNS/experiment_3_static_mac" \
    "$RUNS/experiment_3_static_vm" \
  --output-dir "$STATE_OUT" \
  --train-seeds 42 43 44 45 \
  --validation-seeds 46 \
  --test-seeds $(seq 47 61)

# 4. VM 삭제 (선택)
if [ "$DELETE_VM" -eq 1 ]; then
  gcloud compute instances delete "$VM_NAME" --zone "$ZONE" --quiet
  echo "VM deleted"
else
  echo "VM kept. delete with: $VM_SCRIPT delete"
fi
