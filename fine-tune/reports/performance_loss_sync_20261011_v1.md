# 실제 Trainer 처리량 개선 및 recovery 재개 안내

작성: 2026-10-11T03:30:17.295879+09:00. HEAD `e81a154d76228ac2c462b22e94c06002f6563052`, 미커밋 변경 포함. 동일 JSON에 코드·요청·recovery hash와 실제 측정치를 기록한다.

## 결론과 근거

현재 권장 설정은 **NUM_WORKERS=4, PREFETCH_FACTOR=4, TORCH_THREADS=2, LOSS_SYNC_BATCHES=20**이다. 실제 VGGFace2/MobileFaceNet1024/ArcFace/Adam/batch64와 중단 recovery를 사용해 bounded Trainer를 측정했다. 정식 학습을 재시작하거나 새 FR checkpoint를 생성하지 않았다.

이전 측정하지 않은 후속 batch 구간에서 입력 공급 능력을 비교한 결과는 다음과 같다. 각 조건은 서로 다른 시작 batch10000/20000/30000에서 warmup30 + 측정250 step이다. 원래 recovery3102 상태에서 진단용 sample window만 달리하며 이 offset을 정식 재개로 저장하지 않는다. Windows cache를 강제로 비우지 않았으므로 엄밀한 cold-cache 또는 동일 샘플 paired 실험으로 해석하지 않는다.

| worker | 이미지/초 | 평균 GPU 사용률 | 평균 CPU 사용률 | 사용 RAM GiB | host loader 대기 초 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 319.5 | 57.7% | 40.1% | 36.0 | 35.27 |
| 4 | 548.9 | 94.0% | 54.3% | 36.6 | 0.08 |
| 8 | 534.5 | 93.8% | 56.8% | 37.9 | 0.08 |

4 worker는 약549장/초·GPU94%였고 8 worker는 약534장/초로 추가 이점이 보이지 않았다. 2 worker는 약319장/초·GPU58%, 50초 측정 중 host loader next() 대기 약35초였다. 이는 host 대기 측정이며 CUDA critical path 시간을 독립 분해한 값은 아니다. CPU/RAM 수치를 무조건 높이는 것보다 실제 입력 대기를 줄이는 설정을 선택했다.

반복한 같은 sample window는 이미 cache된 파일의 영향을 받는다. 200 step 시험에서 매 batch 동기화 모드는317장/초, 묶음 모드는547장/초였지만 모든 차이를 loss 동기화만의 효과로 귀속하지 않는다. 400 step 확인은413장/초 vs542장/초였으며 4 worker의 추가 동일-window 확인은533장/초였다. 짧은 단순 benchmark의500장/초를 정식 전체 epoch의 지속 속도로 단정하지 않는다.

## 구현과 학습 조건

- DeferredLossMeter는 detached scalar만 제한된 개수 보관하고 묶어서 CPU로 가져온 뒤 기존 Python 합산 순서와 sample 가중치를 그대로 적용한다. pending tensor가 training graph를 보관하지 않는다.
- optimizer.zero_grad/backward/step은 매 batch 수행한다. gradient accumulation, AMP, batch 확대, subset, 조기 종료를 적용하지 않는다.
- 로그/recovery/epoch 종료 전에 flush하며 validation도 같은 방식을 적용한다. 완료한 batch의 loss를 checkpoint에서 빠뜨리지 않는다.
- loss_sync_batches는 실행 설정에 기록되며 과학적 checkpoint config를 바꾸지 않는다. worker/prefetch 변경도 기존 input RNG·신원 split·sample order를 유지한다.
- 00/02/03/04 첫 셀에서 LOSS_SYNC_BATCHES를 설정한다. 00 기본 worker/prefetch는4/4다. 기존 사용자 변경/출력은 `data/notebook_backups/loss_sync_20261011_031825/`에 보존하고 01은 이번 수정에서 변경하지 않았다.

## 검증 범위

- 전체 `py -3.11 -B -m pytest fine-tune/tests -q -p no:cacheprovider`: **104 passed**. 새 4개 테스트는 scalar 합산 순서 및 세 head의 기존 동기화/묶음 모드 가중치·이력·pending loss가 있는 중단 recovery의 동일성을 확인한다.
- 별도 새 커널에서00–04 계획 실행:5개 통과. 임시 설정으로 EXECUTE=False를 사용하여 정식 학습/원본 설정 저장을 실행하지 않았다.
- 설치된 GTX1080Ti/CUDA 환경에서 원래 recovery3102 + warmup50 + 같은400 step의 2 worker/sync1, 2 worker/sync20, 8 worker/sync20, 4 worker/sync20은 **모델 state_dict hash·누적 loss sum/count·과학적 config hash가 모두 일치**했다.
- 원래 recovery SHA-256 `bfa09a2f877d5c58d1c74616c40d66f324d60afd3ebe9aba7adec55c34e32560`는 시험 전후 동일하다. CPU/GPU·RNG·optimizer 재개에 대한 제한된 검증이며 모든 장비/전체 epoch의 bitwise 동일성이나 정확도 개선을 보장하지 않는다.

## 사용자 실행 순서

1. 변경된 노트북을 다시 열고 커널을 재시작한다. 00의 NUM_WORKERS=4, PREFETCH_FACTOR=4, LOSS_SYNC_BATCHES=20을 확인하고 실행/저장한다.
2. 01은 다시 실행하지 않는다. 기존 준비된 VGGFace2 이미지/목록을 그대로 사용한다.
3. 02 첫 셀의 RESUME_CHECKPOINTS에 이미 연결한 `pretrain_arcface_seed8972`의 recovery_e1_b3102 경로를 확인한다. 이후 새 recovery를 선택할 때는 정확한 경로로 직접 갱신한다.
4. EXECUTE=True, RUN_SMOKE_TEST=False, RUN_BENCHMARK=False로02를 실행한다. 현재 source checkpoint에서1번째 epoch의3102개 완료 batch를 건너뛰고 이어간다. 원래 optimizer/scheduler/RNG/누적 loss를 복구한다.
5. 재개 로그의 실제 worker·진행률·속도를 확인한다. 완료 부모를 얻은 뒤03/04를 실행한다. 이 보고서의 진단 trial을 정식 학습 완료 결과로 사용하지 않는다.

첫 설정 셀의 과거 RESUME_CHECKPOINTS는 자동 최신 선택이 아니다. 부분 run 및 최초 notebook_settings.json은 보존하고 새 resume 설정 기록을 별도로 남긴다. 실제 실행하면 rolling recovery가 새 경로로 갱신되므로 이후 중단 때는 progress.json의 정확한 경로를 확인해야 한다.

## 시간·해석·보존

약530–550장/초가 계속 유지된다는 가정이면 2,984,795 training 이미지 ×40 epoch는 약60–63시간의 순수 학습량이다. validation·recovery/epoch 저장·파일 cache 변동을 추가해야 하며 전체 완료시간은 첫 실제 epoch를 끝낸 뒤 갱신한다. CPU/RAM 사용률 또는 VRAM 할당량 자체는 학습 속도를 보장하지 않는다.

이 작은 JSON/Markdown은 ChatGPT 분석용 집계이며 개별 sample 점수를 포함하지 않는다. 진단 request/progress는 Git 제외된 data/performance_*에, 공식 학습 결과는 기존 campaign/체크포인트 계약에 저장한다. 원본 데이터와 원래 recovery는 삭제하지 않았다. 학습 결과의 KEEP_RAW_RESULTS 정책 및 공식 평가 프로토콜은 그대로 유지한다.
