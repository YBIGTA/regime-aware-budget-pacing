# Ki 보정 실험 공통 실행 안내

이 브랜치는 팀의 원격 실험 브랜치 `analysis/regime-aware-pi-20261114`의 `698b49a`를 기준으로, 로컬 연구 커밋 `de5fa3f`에서 Ki 관련 실험만 옮긴 공개용 작업본이다. NSGA-II 실험 폴더와 커밋, 별도 DOCX 보고서 폴더는 추가하지 않는다. 기존 팀 코드와 CMA 모델은 기준 브랜치의 내용을 사용한다.

## 포함한 실험

| 폴더 | 내용 |
|---|---|
| [pacing_sensitivity_pi](../pacing_sensitivity_pi/README.md) | 관측 민감도를 이용한 Ki 보정의 첫 비교 |
| [pacing_sensitivity_refinement](../pacing_sensitivity_refinement/README.md) | 배율 중심 보정과 관측 신뢰도 및 상수 대조군 |
| [pacing_sensitivity_hypotheses](../pacing_sensitivity_hypotheses/README.md) | 초기화와 관측 지연 및 배율 범위의 후속 검증 |
| [pacing_deadline_pi](../pacing_deadline_pi/README.md) | 남은 시간 목표와 소진 위험 시 Ki 복원 |
| [pacing_asymmetric_pi](../pacing_asymmetric_pi/README.md) | 민감도 Ki와 비대칭 지출 목표의 결합 |

각 실험의 코드, 사전 계획, 결과 CSV·JSON과 그림을 포함한다. 최신 주 후보 G_A05는 기존 CMA 계수망, 민감도 기반 Ki 보정 G, 비대칭 50% 목표 보정을 결합한 것이다. 새 1,000일에서 CMA 대비 TE 0.45%, UR 9.41%, V 1.17% 감소를 관측했다. G 대비로는 UR 8.98% 감소와 TE 0.10%, V 0.51% 증가가 함께 나타났다. 최종 테스트 500일은 미사용이다.

## 원본과 공개 작업본의 차이

다섯 실험의 제어기, 가속 평가기, 테스트, 저장된 결과, 계획과 프로토콜은 원본 바이트 그대로다. 실행 환경을 연결하는 `bridge.py` 또는 `env.py`에서 공통 코드 경로 한 곳만 이 폴더로 바꿨다. `runtime.py`와 `fast_eval.py`는 기존 공통 경매 재생 도구를 내용 변경 없이 옮겼다. 이 두 파일에는 NSGA-II 최적화 실험이 포함되지 않는다. README의 실행 안내는 이 공개 작업본에 맞게 바꿨다.

`export_manifest.json`은 원본 커밋과 각 파일의 원본·공개 SHA256 및 변경 종류를 기록한다. 기존 `protocol.json`, `audit.json`, 실행 시각 및 성능 수치를 다시 쓰지 않았다. 원래 감사는 당시 전체 작업 트리의 소스와 제외한 선행 실험에도 의존한다. 경로를 옮긴 이 작업본에서 원래 `experiment.py all` 또는 `experiment.py audit`를 실행하면 동결 해시 검사가 거부할 수 있다. 이 명령들은 원 실험 절차의 기록으로 보존하며, 현재 공개본의 검증은 아래의 **배포 파일 검사와 읽기 전용 재실행**으로 수행한다. 이 검증을 원 실험의 사전 동결 감사를 새로 수행한 것으로 해석하지 않는다.

## 설치

저장소 루트에서 실행한다. 학습은 수행하지 않으며 `pymoo`, `cma` 패키지가 없어도 동결한 모델의 평가가 가능하다. Python 3.14.4 환경에서 검증했다. 직접 의존성 버전은 `requirements.txt`에 고정했으며 전체 전이 의존성 lock 파일은 아니다.

```powershell
py -3.14 -m venv outputs/ki-venv
.\outputs\ki-venv\Scripts\python.exe -m pip install -r research_report/pacing_ki_common/requirements.txt
```

## 파일 검사와 기존 테스트

```powershell
.\outputs\ki-venv\Scripts\python.exe research_report/pacing_ki_common/code/verify_export.py
$kiBundles = @('pacing_sensitivity_pi', 'pacing_sensitivity_refinement', 'pacing_sensitivity_hypotheses', 'pacing_deadline_pi', 'pacing_asymmetric_pi')
foreach ($kiBundle in $kiBundles) {
    .\outputs\ki-venv\Scripts\python.exe -m unittest discover -s "research_report/$kiBundle/code" -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw "검사 실패: $kiBundle" }
}
```

파일 검사는 공개 파일의 해시, 경로 변경 외 원본 소스의 일치, 원래 계획, 기준 시뮬레이터와 모델을 확인한다. 기존 테스트는 미래 정보 차단, 적분 누적 보존, 예산·승수 제한 및 독립 구현의 일치를 검사한다.

## 저장된 확인 자료의 읽기 전용 재실행

```powershell
.\outputs\ki-venv\Scripts\python.exe research_report/pacing_ki_common/code/replay.py --bundle pacing_asymmetric_pi --days 2
```

기본 두 날짜는 해당 실험의 기존 확인 자료 첫날과 마지막 날이다. 사전에 저장된 모든 확인 정책을 가속 평가기와 원본 Python 제어기로 각각 재생하고, 여섯 성능 지표와 경매 테이프를 저장된 결과에 대조한다. `--days`로 기존 확인 자료에서 등간격으로 고르는 날짜 수를 지정할 수 있으며 최종 테스트는 접근하지 않는다. 비교 결과는 Git 추적에서 제외된 `outputs/ki-export-checks`에만 쓴다. 원래 실험 결과나 후보 선택은 변경하지 않는다. 전체 확인 자료 재실행에는 실험별 300·400·800·1,000·1,000을 지정한다.

공개 전 검증 결과는 `validation.json`에 기록한다. 이 파일의 검사는 배포를 위한 재현 확인이며 새로운 성능 가설 검증이나 최종 테스트 평가가 아니다.
