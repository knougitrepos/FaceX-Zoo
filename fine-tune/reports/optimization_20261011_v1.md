# 공통 부모·공식 후속 학습 split 및 입력 파이프라인 최적화

- 작성: 2026-10-11 00:00 KST
- 저장소/브랜치: F:/FaceX-Zoo / add-adaface, HEAD 0d9be3f76a5afbd43e7775d22d0768e3702f54c7 + 미커밋 변경.
- 상태: 구현·제한된 검증 완료. 정식 부모·15개 후속 모델·FIQA 평가 완료가 아니다.
- 출처 및 작은 정량 요약: `optimization_20261011_v1.json`. 기존 변경·완료 결과·입력 파일 보존.

## 논문 대응

UCFace IV-A(PDF 6쪽), 보충자료 II-A(PDF 1쪽)/Table IX(PDF 2쪽)/augmentation(PDF 3쪽)을 대조했다. 사전학습은 VGGFace2, 후속은 대상 공식 training set이다. SurvFace 5,319명/220,888장, TinyFace 2,570명/7,804장, SCFace 50명/800장이다. 현재 SurvFace만 실행 가능한 manifest가 준비됐으며 나머지는 명시적 manifest가 필요하다.

공통 VGGFace2 MobileFaceNet 부모 1개에서 독립적으로 분기한다. 이는 사용자 승인 설계이며 논문이 동일 부모 파일을 모든 head/반복에 공유했다고 명시한 것은 아니다. 부모 ArcFace loss·40 epoch 및 VGGFace2 95:5 이미지 분할은 구현 선택이다. 후속 batch 64·Adam·lr .001·40 epoch·10 epoch마다 .1 decay·1024D·dropout .6·저해상도 head 파라미터는 명시 조건을 검사한다. 112×112/flip/affine는 보충자료에 명시되어 있다. BGR·픽셀 정규화·augmentation 수치 범위·dropout 위치·weight decay 등의 미공개 세부는 정확 재현을 주장하지 않는다.

SurvFace native 평가 신원 5,319개와 training 신원의 교집합은 없다. distractor 28,476장은 신원을 복원할 수 없어 공식 분할에 의존한다. 과거 ronbun calibration 44,683장이 공식 training_set에 포함되어 이번 FR 학습에 들어가므로 새 모델에서 과거 calibration을 독립 cohort로 자동 재사용하지 않는다. C:/ronbun 입력·코드는 수정하지 않았다.

## 구현

- process_runner/run_campaign: 노트북 제어와 Windows spawn 학습 분리, 중복 캠페인 lock, 안전 stop 요청, 로그 보존.
- input_pipeline/dataset: path/label memmap, 반복 경로 검사 감소, worker thread 제한, prefetch/persistent/pin memory. 데이터 bytes·batch·학습 파라미터 유지.
- worker 개수에 독립적인 seed/epoch/sample 입력 난수. 이전 RNG의 부분 checkpoint는 strict config 검증에서 차단한다.
- Trainer: epoch/batch/처리량/ETA progress.json, 약 10분 및 stop 시 optimizer/RNG/batch 위치 recovery, 마지막 cache 유지.
- notebook_workflow/finetune_protocol: 부모 1개·대상별 15개, training/evaluation 출처 hash/수량/신원 확인, 학습·export의 부모와 설정 검증.
- 00 첫 셀에서 NUM_WORKERS=2, PREFETCH_FACTOR=2, TORCH_THREADS=2, RECOVERY_MINUTES=10. 새 namespace shared_parent_v2. 모든 학습·실행 안내는 fine-tune 안에 유지.

## 처리량 진단

실제 GTX 1080 Ti, VGGFace2, MobileFaceNet 1024D/ArcFace 8,631 classes/batch64/Adam, 10 batch warmup. 가중치 저장 없는 제한된 업데이트다.

| 구간 | worker | images/s | 최대 PyTorch GPU allocation MB |
| --- | ---: | ---: | ---: |
| 100 batch, 0→2→4 | 0 | 216.87 | 2565.13 |
| 100 batch, 0→2→4 | 2 | 564.39 | 2566.16 |
| 100 batch, 0→2→4 | 4 | 563.00 | 2565.74 |
| 300 batch, 2→0 | 2 | 472.78 | 2565.13 |
| 300 batch, 2→0 | 0 | 545.38 | 2566.16 |

첫 비교의 worker2/0 비율은 약 2.60이나 전체 epoch 배속으로 해석하지 않는다. 반복 파일 캐시·짧은 sample·두 번째 구간의 다른 CPU 테스트 동시 실행 영향이 있어 worker0도 높은 처리량을 보였다. worker2는 첫 비교에서 4와 차이가 작아 RAM·프로세스 수를 고려한 기본값이다. 최적 worker/전체 epoch 처리량·GPU 평균 활용률은 장기 실행으로 확정하지 않았다. AdaFace/MagFace 대규모 처리량은 미측정이다.

## 검증과 한계

전체 자동 테스트 92개 통과(최종 55.49초). 부모·export 검사 보강 경로의 대상 테스트도 20개 통과했다. 소규모 CPU 입력으로 세 head의 train 중간/validation 중단 재개 가중치·history 일치, worker0/2 입력 순서·augmentation 일치, hash/분할/설정 거부를 검증했다. 다섯 노트북을 임시 설정 파일로 각각 새 커널 계획 모드 실행했고 모두 통과했다. Windows Jupyter의 selector thread fallback 경고가 한 번 있었으며 오류 없이 완료됐다.

전체 데이터 decode/정식 학습/정확한 저자 수치 재현/실제 ronbun 1024D loader/Continuous FIQA는 미실행이다. 원본 archive·checkpoint를 삭제하지 않았고 commit/staging하지 않았다.

## 사용 순서

커널 재시작 → 00 변수 확인·저장 → 01 기존 준비 확인(이미 추출됐으면 재해제하지 않음) → 02 EXECUTE=True/정식 부모 1개 → 03 공식 split 확인·15개 후속 → 04 동일 행렬 export. 재개 시 progress.json의 last_checkpoint를 해당 run ID에 명시한다. 커널 강제 종료 전에 실행 셀 interrupt로 recovery 저장·자식 종료를 확인한다.

[PyTorch 2.7 DataLoader 공식 문서](https://docs.pytorch.org/docs/2.7/data.html)는 Windows spawn, top-level worker 함수, pinned memory 및 worker별 메모리 복제 주의사항의 근거다.

01 노트북은 source 변경이 없어 기존 사용자 실행 출력·execution count를 그대로 복원했다. 변경된 00/02~04의 이전 출력은 notebook_backups에 보존했다.
