# Parent 정규화 단일 변수 대조

이 workflow는 VGGFace2 Parent의 weight_decay만 0.0001에서 0으로 바꿔 출력 성분 축소와 인식 성능의 관계를 조사합니다. UCFace 저자의 미공개 설정을 복원했다거나 성능 재현에 성공했다는 뜻이 아닙니다.

## 실행

`02a_parent_regularization_control.ipynb`의 첫 설정 셀에서 EXECUTE, STOP_AFTER_EPOCH, RESUME_CHECKPOINT, KEEP_RAW_RESULTS를 지정합니다. 기본은 계획 확인(EXECUTE=False), 총 40 epoch 일정의 첫 2 epoch에서 저장 후 정지입니다. STOP_AFTER_EPOCH는 총 학습 epoch를 변경하지 않습니다.

새 대조는 seed 8972로 처음부터 시작합니다. 이전 Parent의 최종/복구 체크포인트는 재개 대상이 아닙니다. 계속할 때는 새 대조 폴더의 명시적 체크포인트와 STOP_AFTER_EPOCH=None을 사용합니다. epoch 2 모델은 completed=False이며 정식 Parent export 및 후속 학습에 사용할 수 없습니다. kernel Interrupt는 기존 launch_campaign의 안전 중단을 요청합니다. 별도 background=True 실행은 커널 종료와 독립적으로 계속되며 출력한 PID/stop_file로 상태·중단을 관리합니다. 동일 run을 중복 실행하지 않습니다.

기존 전처리 이미지·목록은 변경하지 않는 전제로 검증 receipt를 재사용합니다. 학습 이미지 재전처리나 00·01 전체 재실행은 필요하지 않습니다. 모델/로그/요약은 새 외부 출력 경로에 저장하며 기존 모델은 비교 기준으로 보존합니다.

## 결과 읽기

권장 순서: control.json/plan.json → summaries의 명시적 summary.json → 별도 고정 VGGFace2 test 진단 결과 → 40 epoch Parent 완료 상태 → SurvFace MagFace Origin 결과.

summary.json의 paired_checkpoint_health는 동일 epoch의 기존/신규 Parent를 비교합니다. active_gamma는 abs(BN scale)>0.01, active_variance는 running_var>0.001인 성분 수입니다. 이는 진단 기준이며 정확한 유효 차원·인식률·정상 판정 기준이 아닙니다. optimizer_steps와 epoch가 같은지 확인합니다. 일부 epoch만 있으면 partial 결과이며 40 epoch 완료와 구분합니다.

VGGFace2 학습 identity와 분리된 공식 test 고정 표본에서 인식 성능과 표현 집중을 함께 확인합니다. 작은 표본의 결과는 해당 표본에 한정됩니다. SurvFace 최종 test를 조건/중간 epoch 선택에 사용하지 않습니다. 새 Parent를 쓰는 후속 MagFace는 기존 full-FR split(5,319명/220,888장), resize_only, seed 및 weight_decay=0.0001을 유지합니다. Parent 효과와 fine-tune 정규화 효과를 동시에 변경하지 않습니다.

고정 진단은 seed 8972의 SHA-256 순서로 200명×10장을 선택합니다. 인물당 2장은 gallery, 8장은 probe이며 동일 인물 gallery 점수는 최대값으로 합칩니다. genuine score와 동점인 다른 인물은 앞선 순위로 처리합니다. Rank-1/5/20의 성공 수와 probe 분모(1,600)를 같이 남깁니다. 서로 다른 인물의 평균 cosine과 상위 출력 성분의 제곱 비중은 표현 집중의 진단값입니다. paired 차이의 95% 구간은 같은 200명에 대한 인물 단위 bootstrap(2,000회)이므로 독립 학습 seed의 불확실성이나 전체 데이터셋의 성능 보장이 아닙니다.

RUN_FIXED_VALIDATION=True이면 EXECUTE=True 실행이 반환된 후 저장된 epoch 1/2/40 중 존재하는 쌍을 평가합니다. `validation/evaluations/<내용 해시>/summary.json`이 경량 결과, `identity_counts.json`이 인물별 집계입니다. KEEP_RAW_RESULTS=True일 때만 같은 디렉터리의 `raw_outputs.npz`에 임베딩·probe별 rank를 남깁니다. 고정 표본 이미지는 재현 입력으로 보존하며 이 옵션으로 삭제하지 않습니다. 비교 JSON은 `validation/comparisons/`에 새 결과로 저장합니다. 별도 agent 실행의 `watch_parent_pilot.py`는 지정한 학습 PID 종료와 epoch 2 저장을 검증한 후 같은 평가를 수행하며 학습을 자동 재개하지 않습니다.

ronbun 비교는 압축 전 cosine의 TPIR@20@FPIR입니다. 정답 identity의 Top-20 포함과 정답 score의 threshold 통과를 함께 요구합니다. 목표 FPIR 열과 실제 FPIR, 성공/오류 수와 분모를 함께 읽습니다. test ROC의 benchmark operating point와 별도 calibration으로 고정한 threshold의 test 결과는 다릅니다. 본 workflow의 Table I 비교를 Global-safe 보정 결과로 대신하지 않습니다. 같은 cohort의 조건 비교는 paired이며 single seed를 5회 독립 평균으로 표현하지 않습니다.

## 보존·재개·요약

KEEP_RAW_RESULTS=True가 기본이며 epoch별 checkpoint를 보존해 시점 추적을 가능하게 합니다. False이면 Trainer가 이전 주기 checkpoint를 정리하므로 과거 시점의 재분석에는 재계산이 필요할 수 있습니다. 최신 재개 checkpoint, 필수 요약/설정/출처는 보존합니다. 입력 원본·공유 전처리·기존 Parent/완료 run은 삭제 대상이 아닙니다. 모든 checkpoint는 Git 제외 대상이며 외부 출력 경로에 둡니다.

보존 여부와 무관하게 summaries 아래에 작은 JSON과 이 안내를 생성합니다. 요약의 각 행에 checkpoint hash, 설정 hash, epoch, step, weight_decay와 집계값을 남깁니다. bundle.json은 파일 크기와 토큰 추정 기준, 업로드 순서를 기록합니다. 요약만으로 개별 이미지의 오분류나 임베딩 분포를 다시 계산할 수는 없습니다. 그런 분석에는 별도 진단 원본 또는 checkpoint/입력에서 재계산이 필요합니다.

## 승인된 후속 단계의 자동 실행

초기 대조를 검토해 40 epoch 계속 학습을 결정한 뒤에는 `modules.parent_followup.prepare_plan`으로 명시적 Parent launch, 기존 full-FR MagFace checkpoint/설정 사본, ronbun 설정·기준 report를 고정할 수 있습니다. `scripts/watch_parent_followup.py --plan <출력된 JSON>`은 그 계획만 실행합니다. 기본 `keep_raw_results=True`이며 계획 작성 시 변경할 수 있습니다. 이 실행은 노트북 커널과 별개이므로 실행 중 03/04를 중복 실행하지 않습니다.

순서는 Parent 프로세스 종료 및 40 epoch 완료 검증 → 같은 고정 표본에서 최종 Parent 비교 → SurvFace MagFace 1개 학습 → export → ronbun 입력 추출·Origin 평가입니다. 새 Parent를 연결한 실효 fine-tune 설정과 기존 완료 모델을 정확히 비교하고, Parent hash 이외의 학습·데이터·전처리 차이는 차단합니다. 실행 중 고정한 코드·protocol·설정이 달라지면 다음 단계로 넘어가지 않습니다. 오류는 `followup_execution/failure.json`과 실행 로그에 남기며 학습 batch를 건너뛰거나 재개 checkpoint를 임의 선택하지 않습니다.

`followup_execution/status.json`은 현재 단계입니다. `final_parent_report.json`은 Parent 진단만 완료됐다는 뜻이고, 전체 후속 완료에는 `followup_execution/summary.json`과 그 hash를 담은 `COMPLETED`가 모두 필요합니다. `origin_comparison.csv`는 FPIR 20/10/5/1%의 기존·신규 Origin TPIR@20(%), 차이(%p), 실제 FPIR, 평가 분모, 고득점 비등록자 수와 UCFace Table I 수치를 담습니다. 기존 ronbun runner가 생성하는 부가 보정 열은 이 대조의 성능 근거로 사용하지 않습니다. 비교에는 `origin_test_roc_tpir_at_20`을 사용하며 `global_safe` 행의 보정 후 `tpir`과 혼동하지 않습니다.

`HOW_TO_READ.md`, `summary.json`, `origin_comparison.csv`, `chatgpt_bundle.json`은 작은 분석용 묶음입니다. 보존 옵션을 끄면 새 fine-tune의 과거 주기 checkpoint와 진단 raw outputs 및 ronbun 상세 결과 보존이 제한되지만, 입력·재개 cache·기존 완료 결과는 삭제하지 않습니다. 모델·임베딩·상세 점수는 외부 F 드라이브에 두고 Git에 추가하지 않습니다. 단일 seed와 논문의 5회 평균 차이, 고정 gallery/공유 cohort의 의존성, test ROC와 별도 보정 규칙의 차이를 유지해서 해석합니다.
