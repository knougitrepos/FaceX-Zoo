# FaceX-Zoo 노트북 실행 안내

사용자 설정은 **노트북 첫 코드 셀**에서 조절한다. YAML은 누락값을 위한 기본값이며 직접 편집할 필요가 없다.

## 설정 우선순위와 실행 순서

`YAML 기본값 < 00에서 저장한 공통 설정 < 현재 노트북 첫 셀의 변수`

1. `00_environment_and_data.ipynb`: 첫 셀에서 공통 변수를 지정하고 설정 저장 셀까지 실행한다. 공통값은 `fine-tune/data/notebook_settings.json`에 저장된다. 뒤 노트북에서 공통값을 바꾸려면 00을 다시 저장하거나 해당 노트북의 변수로 덮어쓴다. 개별 노트북 변경은 다른 단계에 자동 전파하지 않는다.
2. `01_prepare_vggface2.ipynb`: `EXECUTE=False`로 경로를 확인한 후 `True`로 실행한다. `EXTRACT_TRAIN_ARCHIVE=True`이면 로컬 train 압축파일을 raw_root/train 아래에 해제한 뒤 목록을 생성한다. 최초 실행은 오래 걸릴 수 있으며 약 10초마다 진행 상황을 표시한다. 중단 후 재실행은 gzip을 처음부터 읽으며 기존 파일의 바이트를 비교하고 누락 파일만 추가한다. 다른 내용의 기존 파일은 덮어쓰지 않는다. 압축 해제를 완료했다면 `EXTRACT_TRAIN_ARCHIVE=False`로 건너뛸 수 있다. 기존 목록과 내용이 다르면 덮어쓰지 않고 중단한다. 분할을 바꾸려면 새 train/val 목록 경로를 사용한다.
3. `02_pretrain.ipynb`: `PARENT_SEED=8972`로 VGGFace2 공통 ArcFace 부모 **1개**를 학습한다. `EXECUTE=True`, `RUN_SMOKE_TEST=False`, `RUN_BENCHMARK=False`가 정식 학습이다. `RUN_BENCHMARK=True`면 worker 처리량만 측정하고 학습 가중치를 만들지 않는다.
4. `03_finetune.ipynb`: F 드라이브의 TinyFace·SurvFace를 전체 decode/파일 SHA-256/MAT 이름·신원으로 검증하고 공식 training manifest를 만든 뒤, 동일 부모를 독립 복사하여 대상 데이터셋 × seed × ArcFace/AdaFace/MagFace/CosFace를 순차 학습한다. 새 classifier를 대상 identity 수로 초기화하며 backbone도 업데이트한다. FR 학습 split에서 FIQA fit을 따로 떼지 않는다. VGGFace2 재학습 또는 Arc→Ada→Mag 연쇄 학습이 아니다.
5. `04_export_and_validate.ipynb`: 같은 데이터셋 × head × seed 목록을 순차 export한다. 학습 당시 전처리·학습 설정·분할·출처 hash를 함께 기록하며 설정 불일치는 거부한다.

01~04의 초기값 `EXECUTE=False`는 계획 모드다. 단순 Run All로 장기 학습이나 export를 시작하지 않는다. 실제 실행하려면 해당 첫 셀에서 True로 바꾸고 순서대로 실행한다. 수정된 Python 모듈을 쓰는 다음 단계는 새 커널에서 실행한다. 실행 중인 02 커널은 재시작하지 않고 완료까지 유지한다.

## CosFace 추가 후 실행 (2026-10-13)

- 현재 02는 기존 ArcFace 공통 부모 하나를 계속 학습한다. 00·02 노트북, 저장된 `notebook_settings.json`, 실행 중 request, trainer·backbone·기존 head 계산은 이 변경의 대상이 아니다.
- 02 완료 후 **03을 별도 새 커널**에서 열고 첫 셀부터 실행한다. `HEADS`는 네 head를 명시하므로 과거 00에 저장된 세-head 목록을 덮어쓴다. 이번 추가만을 위해 00/01/02를 재실행할 필요는 없다. 기본 `EXECUTE=False`에서 40개 계획을 먼저 확인한다.
- `HEAD_PARAMETERS["cosface"].update(type="CosFace", margin=0.3, scale=64.0)`가 노트북에 노출된다. 논문 IV-A-2의 저해상도 값이며 정식 03은 다른 margin/scale을 거부한다. 클래스 수는 TinyFace 2570, SurvFace 5319로 각 profile에서 설정한다.
- 04도 새 커널에서 03과 같은 네 head·seed·데이터셋·부모·출력 root로 실행한다. 04는 재학습하지 않는다. 기존 세 head를 이미 완료했다면 두 노트북의 `HEADS=["cosface"]`로 CosFace만 추가할 수 있다. 기존 완료 모델은 보존된다.
- CosFace 계산은 기존 `head/AM_Softmax.py`의 additive cosine margin을 사용한다. run/파일명/metadata의 head는 `cosface`, `source_config.head.type`은 실제 클래스 `AM_Softmax`다. `margin`과 `scale`을 checkpoint 설정에 기록하여 다른 설정의 재개·export를 거부한다. 별도의 CosFace 부모 학습은 없다.

## 주요 사용자 변수

| 변수 | 기본/역할 | 주의점 |
| --- | --- | --- |
| `SEEDS` | `[8972,8973,8974,8975,8976]` | 실제 반복 실행한다. 한 seed만 필요하면 `[8972]`. 다음 단계도 같은 seed 목록을 사용한다. |
| `HEADS` | `["arcface", "adaface", "magface", "cosface"]` | 03/04에서 실제 반복한다. 02는 공통 ArcFace 부모만 생성한다. |
| `KEEP_RAW_RESULTS` | `True` | 현재 노트북에서 변경하면 Trainer에 전달된다. False는 이번 run의 이전 중간 periodic checkpoint를 정리한다. 입력 이미지·부모 가중치 삭제 옵션이 아니다. |
| `NUM_WORKERS` | `4` (실측 권장) | 학습은 별도 Python 프로세스에서 실행하여 Windows spawn을 지원한다. 새 후속 batch 구간에서 2 worker의 입력 대기를 확인하고 4를 선택했다. 01의 노트북 내 표본 확인 loader는 0을 유지한다. |
| `PREFETCH_FACTOR` / `PERSISTENT_WORKERS` | `4` / `True` | worker별 미리 준비할 batch 수·epoch 간 worker 유지. RAM 부족 시 worker/prefetch를 줄인다. |
| `LOSS_SYNC_BATCHES` | `20` | loss 기록용 CPU 전송을 묶어 GPU 동기화를 줄인다. optimizer는 매 batch 갱신한다. 로그·recovery·epoch 종료 전에 모두 반영하며 gradient accumulation이 아니다. `1`은 매 batch 동기화 진단 모드다. |
| `TORCH_THREADS` | `2` | 학습 프로세스 CPU 스레드. 각 worker OpenCV/PyTorch는 1로 제한한다. |
| `RECOVERY_MINUTES` | `10` | 완료된 step까지 중간 recovery checkpoint를 저장하는 간격. 중단 요청에도 저장한다. |
| `PARENT_SEED` / `PRETRAIN_EPOCHS` | `8972` / `40` | 부모 1개의 seed·일정. 사전학습 loss/일정의 세부는 논문 미공개이므로 구현 선택이다. 후속 학습 40 epoch와 구분한다. |
| `PARENT_CHECKPOINT` / `PARENT_SHA256` | `None` | 02 결과를 기본으로 사용. 다른 부모는 정확한 파일·hash·완료 출처를 지정하며 임의 파일이나 부분 학습을 허용하지 않는다. |
| `FINETUNE_DATASETS` / `FINETUNE_PROFILES` | `["tinyface", "survface"]` / 로컬 공식 원본 | `raw_root`: F:/FaceX-Zoo/tinyface 및 F:/FaceX-Zoo/QMUL-SurvFace. TinyFace 2570/7804, SurvFace 5319/220888. SCFace는 미확보 상태로 enabled=False이며 선택 시 거부한다. |
| `AUDIT_WORKERS` | `4` (03/04) | CPU 이미지 decode worker 수. 최초 검사 후 audit hash/source snapshot을 확인하여 재사용한다. 이 값은 학습 NUM_WORKERS와 별개다. |
| `DATASET` | 입력 root, train/val 목록, 크기, mean/std, 클래스 수 | 실제 loader에 전달된다. 현재 MobileFaceNet/export는 112×112만 지원하므로 다른 크기는 명시적으로 차단한다. 값이 조용히 무시되지 않는다. |
| `TRAINING` | Adam, lr=0.001, 40 epoch, batch=64 등 | 논문 대응 조건을 유지한다. 변경하면 별도 실험이며 기존 checkpoint 재개 검증에서 거부될 수 있다. |
| `BACKBONE` / `HEAD_PARAMETERS` | 차원, dropout, 네 head의 파라미터 | CosFace는 03/04 첫 셀의 `margin=0.3`, `scale=64`가 우선한다. 나머지는 00 저장값을 이어받는다. 본문 미명시 설정은 본 연구 선택이다. |
| `PATHS` | checkpoint/export/report 위치 | F 드라이브 기본. 다른 실험 조건은 새 checkpoint/export root를 사용한다. |
| `RUN_SMOKE_TEST` | `False` | 02/03에서 True면 더미 입력 forward/backward/optimizer 1 step만 실행한다. 정식 checkpoint는 생성하지 않는다. |
| `RESUME_CHECKPOINTS` | `{}` | run ID → 중단 checkpoint 경로. 예: `{"finetune_survface_adaface_seed8972": "F:/.../recovery_....pt"}`. progress.json의 last_checkpoint를 확인해서 지정한다. 자동 최신 선택은 없다. |
| `REUSE_COMPLETED` | `True` | 명시된 run의 완료·설정·출처를 검증한 뒤 재사용한다. 기존 파일을 다시 학습하거나 덮어쓰지 않는다. |
| `EXTRACT_TRAIN_ARCHIVE` | `True` (01만) | `EXECUTE=True`와 함께 설정하면 로컬 train archive를 해제한다. 원본 archive와 test archive는 보존한다. 얼굴 정렬/리사이즈 단계가 아니다. |
| `VAL_EVERY` | `20` (01만) | 20번째마다 validation으로 분리하는 기존 정책. 이미지 단위 분할이며 신원 분리 평가가 아니다. |

기본 계획은 부모 **1개**, TinyFace 20개 + SurvFace 20개 = 후속 모델 **40개**, export 40개다. SCFace는 아직 선택할 수 없다. 모든 후속 run이 같은 부모 hash를 공유한다. 논문의 5회 반복과 정확한 부모 공유 절차는 구분하며 부모 공유·seed 값은 사용자 선택이다. 변경 후 00을 다시 저장해야 기존 JSON의 SurvFace 전용/C 경로 설정을 새 F 경로로 바꿀 수 있다.

## 재개·저장·실패 처리

- 완료 run을 다시 실행하면 설정·데이터 목록 hash·부모·seed·head·stage를 검사하고 일치하는 경우에만 재사용한다.
- 미완료 run 폴더에 checkpoint가 있으면 `RESUME_CHECKPOINTS`로 명시적으로 재개한다. 실패 후 01을 다시 실행하여 공유 목록을 바꾸지 않는다.
- 첫 실패에서 후속 작업을 중단한다. `reports/campaigns/<stage>_<KST 시각>_<식별자>.json`에 완료·실패·미실행 run과 실제 설정을 기록한다. 중단 후 새 실행의 summary는 별도 파일이다.
- 원본 미보존 설정에서도 완료 모델·출처와 집계 summary는 남긴다. 삭제된 중간 checkpoint 시점으로 되돌아갈 수 없으며, 초기 학습을 다시 해야 할 수 있다.
- `notebook_settings.json`은 사용자의 변경 가능한 설정이다. run 폴더의 설정 사본과 완료 모델은 기존 결과로 보존한다. 설정을 바꾸면 이미 완료된 run을 소급 변경하지 않는다.
- 코드 변경 전 기존 사용자 노트북과 실행 출력은 `fine-tune/data/notebook_backups/`에 보존했다. 이 폴더는 Git 제외 대상이며 입력 데이터/연구 결과와 구분한다.
- 진행률은 각 run의 `progress.json`, 전체 콘솔은 `fine-tune/data/launches/<id>/console.log`에 남긴다. epoch/batch/이미지 처리량과 epoch 학습 ETA를 표시한다. validation·checkpoint 저장 시간은 학습 ETA와 별도다.
- 중단은 실행 셀 interrupt로 요청하고 recovery 저장을 확인한다. 커널 강제 종료는 별도 자식 프로세스를 종료하지 않는다. 남은 `.campaign.lock`은 기록 PID 종료를 확인한 후 수동 처리한다.
- 경로·label memmap 캐시는 원본 이미지 캐시가 아니다. worker 수 변경은 sample/epoch/index별 입력 난수를 바꾸지 않는다. 이전 input RNG 방식의 부분 checkpoint는 새 실행에 자동 재개하지 않는다.
- recovery는 마지막 완료 optimizer step을 복구한다. 정전/강제 종료 시 마지막 recovery 이후 작업은 재실행한다. loader spawn 및 recovery 정확성 시험은 제한된 입력으로 검증했으며 모든 장비의 bitwise 동일성을 보장하지 않는다.
- loss의 GPU→CPU 전송은 `LOSS_SYNC_BATCHES`로 묶되, Python scalar의 원래 순서·가중 합산을 유지한다. 로그/recovery/epoch 종료 전 flush하여 완료 batch의 기록을 빠뜨리지 않는다. 이 옵션과 worker/prefetch는 실행 설정에 기록하며 학습 hyperparameter를 변경하지 않는다.
- `KEEP_RAW_RESULTS=False`에서도 최신 recovery·최종 모델·집계·출처는 남긴다. 현재 프로세스가 만든 이전 recovery는 간격마다 최신 파일로 교체하며 이 정책은 상세 결과 보존과 구분한다.

## 환경 확인과 결과 해석

00은 CUDA 사용 가능 여부, 장치, branch, 파일 존재·크기와 메타데이터 identity 수를 표시한다. 파일 존재를 압축파일 전체 무결성 검증이라고 표시하지 않는다. `archive_integrity`와 `all_images_decode`는 실행하지 않았다면 `not_checked`다. 환경 점검은 정식 학습 완료를 의미하지 않는다.

01의 loader 확인은 일부 이미지의 존재·decode·전처리를 확인한다. 전체 이미지가 모두 정상이라는 증거는 아니다. 이후 누락/손상 이미지가 나타나면 학습은 오류로 중단된다.

캠페인 JSON의 `planned`, `completed`, `rows`, `not_started`를 함께 읽는다. `smoke`의 `completed=False`는 의도적이며 성능 평가 결과가 아니다. 각 run의 checkpoint SHA-256·config hash·부모 hash와 실제 보존 설정을 확인한다. 이 summary는 원본 점수 전체를 묶은 ZIP이 아니라 실행 상태·출처의 경량 집계다. 전체 개요는 캠페인별 JSON 하나부터 공유하고, 모델 세부 분석에는 해당 export의 `_meta.json`을 함께 사용한다. 학습별 상세 sample score는 이 workflow에서 생성하지 않는다.

04는 순수 backbone strict-load/shape를 검사한다. 실제 얼굴인식 지표, ronbun loader 일치, Continuous FIQA, TPIR/FPIR 평가는 여기서 수행하지 않는다. 메타데이터에도 미실행 상태를 남긴다.

## UCFace와의 대응 범위

MobileFaceNet 1024D, VGGFace2 사전학습, 대상 공식 training split의 후속 학습, Adam, batch 64, lr=0.001, 후속 40 epoch와 저해상도 head 설정을 논문 IV-A 및 보충자료 II-A/Table IX에 맞춘다. 112×112와 flip/affine augmentation은 보충자료에 명시되어 있다. 공통 ArcFace 부모, 사전학습 40 epoch·VGGFace2 95:5 이미지 분할, dropout 위치·전처리 세부는 구현 선택이다. 저자의 미공개 recipe까지 정확히 재현했다고 표시하지 않는다. UCFace 추가 contrastive loss는 이번 baseline 범위에 포함하지 않는다.

SurvFace의 알려진 native identity는 train/evaluation 간 겹침을 검사한다. 28,476개 `distractors_*`의 신원은 파일에서 확인되지 않으므로 공식 test 분할에 의존하는 한계를 protocol.json에 기록한다. 현재 ronbun의 공식 training_set 중 과거 FIQA calibration 44,683장도 FR 학습에 포함된다. 새 모델로 ronbun 실험을 할 때 그 calibration을 독립 cohort로 자동 재사용하지 말고 별도 분할 계약을 검토한다. 기존 C 저장소의 파일은 이번 작업에서 변경하지 않는다.

## 로컬 데이터 검증 및 Continuous FIQA fit

파생 학습 목록·MAT 출처·전체 이미지 hash·decode 검사 결과는 `fine-tune/data/protocols/local_qmul_v1/<dataset>/`에 생성한다. 이전 C 경로의 protocol은 보존한다. `.gitignore`는 원본 두 폴더와 대형 감사 CSV를 제외하며 소형 보고서는 추적 가능하다. `KEEP_RAW_RESULTS`는 입력 데이터·이 입력 검증 출처를 삭제하지 않는다.

검증은 원본을 읽기만 하고 이미지·MAT·공식 split을 수정하지 않는다. 소스 snapshot/출처 파일이 바뀌면 재사용을 거부한다. 완료 감사의 신뢰 범위는 로컬 MAT·수량·전체 decode/개별 SHA-256이며 원 저자 archive와 바이트 단위 동일성은 확인되지 않았다. 공식 홈페이지/README/UCFace 표의 수량 차이는 audit notes에 남긴다. MAT에 없는 probe를 만들어 숫자를 맞추지 않는다.

FR 학습은 논문 training split 전체를 유지한다. 이후 FIQA fit은 별도 역할이다. 현재 ronbun Continuous FIQA는 non-mated query score의 fit/safety를 각각 최소20개 요구하며, 표본 수만으로 품질 분포 coverage나 FPIR 신뢰성을 보장하지 않는다.

SurvFace에는 native identity를 확인 가능한 non-mated 2319명/93260장과 anonymous distractor 28476장이 있다. `fiqa_fit_capacity_20261011_v1.json`은 알려진 non-mated 신원을 seed8972로 development40%에 배정하고 그 안을 fit/safety로 나누는 **용량 검토용 제안**이다. 공식 gallery3000명·FR train은 유지할 수 있으나 최종 unknown test의 분모가 바뀐다. 이 제안은 아직 실험 split이나 ronbun 실행 설정에 적용하지 않았다. 독립 FIQA test와 공식 전체 UCFace 평가를 구분하거나, 공식 전체 test를 유지하려면 외부 calibration을 확보해야 한다. fit/test를 같은 공식 test 전체로 재사용하지 않는다.

TinyFace의 공식 평가는 mated probe + distractor gallery의 ranking이며 non-mated probe가 없다. 따라서 현재 threshold 방식의 FPIR calibration을 그 공식 protocol에서 바로 수행할 수 없다. probe-only 임계값은 순위를 바꾸지 않으므로 Rank-1 개선으로 효용을 평가하지 않는다. TinyFace는 FR fine-tuning/공식 ranking 검사, SurvFace는 Continuous FIQA 1:N 주 평가에 사용한다. TinyFace에서 rejection을 평가하려면 별도의 unknown-probe 파생 protocol을 명시해야 한다.

현재 ronbun SurvFace manifest는 일부 unknown identity를 파일명 단위로 표현한다. 새로운 fit/safety 분할에는 이 값 대신 source_identity_id로 복원한 native identity를 사용해야 같은 사람의 이미지가 양쪽으로 섞이지 않는다. 이 저장소에서 C:/ronbun의 분할·학습·평가 코드를 수정하거나 FIQA model을 fit하지 않는다.

이번 로컬 전체 검사 결과와 FIQA 분리안은 [데이터 검증·FIQA 용량 보고서](reports/local_qmul_integrity_and_fiqa_20261011_v1.md)부터 읽는다. 같은 이름의 JSON은 입력 수량·출처 hash·검증/미실행 상태를 담은 ChatGPT 분석용 소형 요약이며, 모델 학습 성능 결과가 아니다.

- [UCFace 공식 논문](https://ieeexplore.ieee.org/document/10595448)
- [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo)
- [AdaFace 공식 코드](https://github.com/mk-minchul/AdaFace)
- [PyTorch Windows FAQ](https://docs.pytorch.org/docs/2.7/notes/windows.html)

## 검증 명령

CosFace 추가와 기존 02 보존·CPU 재개/export 검증은 [CosFace 통합 검증 기록](reports/cosface_integration_20261013_v1.md)을 따른다. 해당 기록은 정식 CosFace 모델 학습 완료를 뜻하지 않는다.

SCFace 확보 후 별도 실행을 위한 사용자 변수와 manifest 준비 절차는 [scface_instruction.md](scface_instruction.md)를 따른다. 현재 기본 학습 대상은 TinyFace·SurvFace이며 SCFace는 준비·검증 후 명시적으로 활성화한다.

실제 Trainer의 입력 대기·loss 동기화 비교와 현재 권장 설정은 [처리량 개선 보고서](reports/performance_loss_sync_20261011_v1.md)를 따른다. 02의 기존 `RUN_BENCHMARK`는 워밍업 후 짧은 단순 루프 시험이므로 전체 epoch 속도와 구분한다. 추가 진단용 `scripts/profile_training.py --request <JSON>`는 정식 학습이 중단된 상태에서만 사용한다. CUDA·psutil·nvidia-smi가 필요하며 정식 모델을 저장하지 않는다. 반복 파일 cache와 새로운 batch 구간을 구분하여 해석한다.

```powershell
py -3.11 -B -m pytest fine-tune/tests -q -p no:cacheprovider
```

단위 테스트, 임시 소량 이미지 학습, 전체 VGGFace2 정식 학습은 구분한다. 실제 검증 결과는 해당 작업 보고서에 기록한다.
