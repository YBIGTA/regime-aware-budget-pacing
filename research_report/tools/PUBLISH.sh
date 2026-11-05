#!/usr/bin/env bash
set -euo pipefail
project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
if ! git diff --cached --quiet; then
  echo '이미 staged인 변경이 있습니다. 다른 작업과 섞이지 않도록 중단합니다. 해당 작업을 먼저 정리한 뒤 다시 실행하세요.'
  exit 1
fi
origin_url=$(git remote get-url origin)
case "$origin_url" in
  *YBIGTA/regime-aware-budget-pacing*|*YBIGTA:regime-aware-budget-pacing*) ;;
  *) echo "예상한 YBIGTA 저장소가 아닙니다. 현재 origin: $origin_url"; exit 1 ;;
esac
if [ -z "$(git symbolic-ref --short -q HEAD)" ]; then
  echo '브랜치가 없는 detached HEAD 상태입니다. 현재 작업 브랜치로 이동한 뒤 실행하세요.'
  exit 1
fi
if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
  echo '해결되지 않은 Git 충돌이 있습니다.'; exit 1
fi
if [ -d research_report ]; then
  echo '기존 research_report가 있어 새로 덮어쓰지 않습니다. 이미 생성했다면 아래 commit/push 명령만 실행하세요.'; exit 1
fi
auctionnet_lab/.venv/bin/python pacing_report_tools/build_report.py --root "$project_root"
git add -f -- README.md research_report
if ! git diff --cached --quiet; then
  git commit -m "Document pacing experiments, raw results, and reproducibility audit"
fi
git push -u origin HEAD
echo '완료: README와 실험 증거를 현재 origin 브랜치에 푸시했습니다.'
