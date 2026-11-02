#!/usr/bin/env bash
# GCP VM 한 대에서 실험 설정을 병렬 실행하고 결과를 내려받는다.
#
#   experiments/data_collection/gcp_vm.sh create     # Spot VM 생성 (비용 발생 시작)
#   experiments/data_collection/gcp_vm.sh upload     # 작업 폴더 업로드 (.venv, outputs 제외)
#   experiments/data_collection/gcp_vm.sh setup      # Python 환경 설치
#   experiments/data_collection/gcp_vm.sh run [실험 ID ...]  # 백그라운드 실행, 끝나면 VM 자동 정지
#   experiments/data_collection/gcp_vm.sh status     # 진행 상황
#   experiments/data_collection/gcp_vm.sh start      # 정지된 VM 다시 켜기 (Spot 회수 후 재개 등)
#   experiments/data_collection/gcp_vm.sh download [실험 ID ...]  # 결과를 outputs/simulation_runs로
#   experiments/data_collection/gcp_vm.sh delete     # VM과 디스크 삭제 (비용 종료)
#
# 실험 ID를 생략하면 experiment_2 ~ experiment_6 전체를 대상으로 한다.
# Spot VM이 회수되면 VM이 정지된다. start 후 run을 다시 실행하면 남은 run만 이어서 실행된다.
set -euo pipefail

VM_NAME="${VM_NAME:-pacing-sim}"
ZONE="${ZONE:-asia-northeast3-a}"
MACHINE_TYPE="${MACHINE_TYPE:-c2d-highcpu-32}"
WORKERS="${WORKERS:-32}"
DISK_SIZE="${DISK_SIZE:-100GB}"
REMOTE_DIR="rabp"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_EXPERIMENTS=(
  experiment_2_test_seeds
  experiment_3_static
  experiment_4_pi_grid
  experiment_5_transitions
  experiment_6_shocks
)

remote() {
  gcloud compute ssh "$VM_NAME" --zone "$ZONE" --quiet --command "$1"
}

experiments_or_default() {
  if [ "$#" -gt 0 ]; then
    echo "$@"
  else
    echo "${DEFAULT_EXPERIMENTS[@]}"
  fi
}

cmd="${1:-}"
shift || true

case "$cmd" in
  create)
    gcloud compute instances create "$VM_NAME" \
      --zone "$ZONE" \
      --machine-type "$MACHINE_TYPE" \
      --provisioning-model SPOT \
      --instance-termination-action STOP \
      --image-family debian-12 \
      --image-project debian-cloud \
      --boot-disk-size "$DISK_SIZE" \
      --boot-disk-type pd-balanced
    ;;

  upload)
    # 외부 Yahoo 서브모듈까지 포함한다. .git, 가상환경, 기존 출력은 올리지 않는다.
    # COPYFILE_DISABLE: macOS tar가 ._ 메타데이터 파일을 넣지 않게 한다.
    COPYFILE_DISABLE=1 tar -C "$PROJECT_ROOT" -czf - \
      --exclude .git --exclude .venv --exclude outputs --exclude __pycache__ \
      . | gcloud compute ssh "$VM_NAME" --zone "$ZONE" --quiet \
        --command "mkdir -p ~/$REMOTE_DIR && tar -xzf - -C ~/$REMOTE_DIR"
    echo "uploaded to ~/$REMOTE_DIR"
    ;;

  setup)
    remote "set -e
      sudo apt-get update -qq
      sudo apt-get install -y -qq python3-venv
      cd ~/$REMOTE_DIR
      python3 -m venv .venv
      .venv/bin/pip install -q -r requirements.txt
      .venv/bin/python --version"
    ;;

  run)
    experiments="$(experiments_or_default "$@")"
    # SSH가 끊겨도 계속 실행되도록 nohup으로 띄운다. 모두 끝나면 VM을 정지한다.
    remote "cd ~/$REMOTE_DIR && rm -f DONE && setsid nohup bash -c '
      for experiment in $experiments; do
        echo \"=== \$experiment \$(date)\"
        .venv/bin/python src/simulator/run_experiments.py \
          --config configs/\$experiment.json --workers $WORKERS
      done
      echo \"=== finished \$(date)\"
      touch DONE
      sudo shutdown -h now
    ' > run.log 2>&1 < /dev/null &
    echo started: $experiments"
    ;;

  status)
    remote "cd ~/$REMOTE_DIR
      if [ -f DONE ]; then echo 'state: DONE'; else echo 'state: running or stopped'; fi
      for dir in outputs/simulation_runs/*/; do
        [ -d \"\$dir\" ] || continue
        echo \"\$(basename \$dir): \$(ls \$dir/runs 2>/dev/null | wc -l) run folders, \$(find \$dir/runs -name trace.csv 2>/dev/null | wc -l) finished\"
      done
      echo '--- last log lines'
      tail -n 5 run.log"
    ;;

  start)
    gcloud compute instances start "$VM_NAME" --zone "$ZONE"
    ;;

  download)
    experiments="$(experiments_or_default "$@")"
    mkdir -p "$PROJECT_ROOT/outputs/simulation_runs"
    archive="$PROJECT_ROOT/outputs/simulation_runs/download.tar.gz"
    paths=""
    for experiment in $experiments; do
      paths="$paths outputs/simulation_runs/$experiment"
    done
    gcloud compute ssh "$VM_NAME" --zone "$ZONE" --quiet \
      --command "cd ~/$REMOTE_DIR && tar -czf - $paths" > "$archive"
    tar -xzf "$archive" -C "$PROJECT_ROOT"
    rm "$archive"
    echo "downloaded: $experiments"
    ;;

  delete)
    gcloud compute instances delete "$VM_NAME" --zone "$ZONE"
    ;;

  *)
    sed -n '2,15p' "$0"
    exit 1
    ;;
esac
