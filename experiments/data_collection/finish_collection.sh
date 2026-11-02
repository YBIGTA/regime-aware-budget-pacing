#!/usr/bin/env bash
# 실험 2·3 수집이 끝난 뒤 맥미니·VM 결과를 받고, 개수를 검증하고, 상태 테이블을 만든다.
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
# 맥미니: Tailscale + SSH 키(이 노트북에서만 허용)로 접속한다.
MINI_HOST="${MINI_HOST:-chanwoong@100.102.68.41}"
MINI_KEY="${MINI_KEY:-$HOME/.ssh/id_ed25519_macmini}"
MINI_DIR="${MINI_DIR:-Desktop/ybigta_conference/regime-aware-budget-pacing}"

DELETE_VM=0
[ "${1:-}" = "--delete-vm" ] && DELETE_VM=1

# 실험 폴더와 기대 run 수
EXPECTED=(
  "experiment_1_1:195"
  "experiment_2_test_seeds:675"
  "experiment_3_static_mac:729"
  "experiment_3_static_mini:210"
  "experiment_3_static_vm:114"
  "experiment_3_static_vm2:162"
)
VM_EXPERIMENTS=(experiment_2_test_seeds experiment_3_static_vm experiment_3_static_vm2)
MINI_EXPERIMENTS=(experiment_3_static_mac experiment_3_static_mini)

count_traces() { find "$RUNS/$1/runs" -name trace.csv 2>/dev/null | wc -l | tr -d ' '; }
count_completed() {
  # 새로 실행한 run은 completed, 이미 있어서 건너뛴 run은 skipped로 기록된다.
  # validation_status(10번째 열)가 passed인 행만 센다.
  awk -F, 'NR > 1 && ($6 == "completed" || $6 == "skipped") && $10 == "passed"' \
    "$RUNS/$1/manifest.csv" 2>/dev/null | wc -l | tr -d ' '
}

PARTS=(experiment_3_static_mac experiment_3_static_mini experiment_3_static_vm experiment_3_static_vm2)
MERGED=experiment_3_static
MERGED_EXPECTED=1215

expected_for() {
  for item in "${EXPECTED[@]}"; do
    [ "${item%%:*}" = "$1" ] && echo "${item##*:}"
  done
}

if [ "$(count_traces "$MERGED")" -eq "$MERGED_EXPECTED" ]; then
  echo "$MERGED already merged; skipping download"
else

# 1. 맥미니 결과 받기 (rsync라 이미 받은 파일은 건너뛴다)
for experiment in "${MINI_EXPERIMENTS[@]}"; do
  if [ "$(count_traces "$experiment")" -ne "$(expected_for "$experiment")" ]; then
    echo "fetching $experiment from Mac mini"
    mkdir -p "$RUNS/$experiment"
    rsync -az -e "ssh -i $MINI_KEY -o BatchMode=yes" \
      "$MINI_HOST:$MINI_DIR/outputs/simulation_runs/$experiment/" "$RUNS/$experiment/"
  fi
done

# 2. VM 결과 받기
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

# 3. 개수 검증
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

# 4. 실험 3 조각 폴더를 experiment_3_static 하나로 합친다
.venv/bin/python - "$RUNS" "$MERGED" "${PARTS[@]}" <<'PY'
import csv, shutil, sys
from pathlib import Path
runs, merged, parts = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
target = runs / merged
(target / "runs").mkdir(parents=True, exist_ok=False)
manifest, conditions, condition_fields = [], {}, None
for part in parts:
    part_dir = runs / part
    with (part_dir / "manifest.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_fields = reader.fieldnames
        for row in reader:
            run_dir = target / "runs" / row["run_id"]
            shutil.move(str(part_dir / "runs" / row["run_id"]), run_dir)
            row["trace_path"] = str((run_dir / "trace.csv").resolve())
            row["metadata_path"] = str((run_dir / "metadata.json").resolve())
            manifest.append(row)
    with (part_dir / "conditions.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        condition_fields = reader.fieldnames
        for row in reader:
            conditions[(row["scenario_id"], row["start_minute"])] = row
for name, fields, rows in (
    ("manifest.csv", manifest_fields, manifest),
    ("conditions.csv", condition_fields, list(conditions.values())),
):
    with (target / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
for part in parts:
    shutil.rmtree(runs / part)
print(f"merged {len(manifest)} runs into {target}")
PY

fi

# 5. 최종 개수 확인
for item in "experiment_1_1:195" "experiment_2_test_seeds:675" "$MERGED:$MERGED_EXPECTED"; do
  experiment="${item%%:*}"
  expected="${item##*:}"
  traces="$(count_traces "$experiment")"
  completed="$(count_completed "$experiment")"
  echo "$experiment: trace.csv=$traces manifest=$completed expected=$expected"
  if [ "$traces" -ne "$expected" ] || [ "$completed" -ne "$expected" ]; then
    echo "count mismatch after merge. stopping."
    exit 1
  fi
done

# 6. 상태 테이블 (학습 42~45, 검증 46, 평가 47~61)
.venv/bin/python src/preprocessing/build_state_table.py \
  --experiment-dir \
    "$RUNS/experiment_1_1" \
    "$RUNS/experiment_2_test_seeds" \
    "$RUNS/$MERGED" \
  --output-dir "$STATE_OUT" \
  --train-seeds 42 43 44 45 \
  --validation-seeds 46 \
  --test-seeds $(seq 47 61)

# 7. VM 삭제 (선택)
if [ "$DELETE_VM" -eq 1 ]; then
  gcloud compute instances delete "$VM_NAME" --zone "$ZONE" --quiet
  echo "VM deleted"
else
  echo "VM kept. delete with: $VM_SCRIPT delete"
fi
