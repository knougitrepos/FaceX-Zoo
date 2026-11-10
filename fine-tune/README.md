# FaceX-Zoo 모델 생성: 외부 데이터·모델 저장소

현재 실행 안내다. 과거 adjust1/shared_parent_v3 안내는
[기록용 안내](docs/legacy/README_before_shared_storage_20261028.md)에 보존했다.
과거 실패 조건의 fine-tune 결과는 기록용이며 새 실험에서 자동으로 선택하지 않는다.

## 저장 위치와 독립성

### Backbone 이름과 단계별 학습률

모델명 입력은 앞뒤 공백 제거 후 소문자화하여 판별한다. `resnet`, `ResNet`,
`RESNET`은 모두 같으며, 새 설정에는 기존 호환 이름 `Resnet`으로 기록한다.
MobileFaceNet도 대소문자를 구분하지 않고 새 설정에는 `MobileFaceNet`으로 기록한다.
알 수 없는 이름은 거부한다. 기존 checkpoint와 그 metadata/hash는 변경하지 않는다.

UCFace IV-A-3의 downstream baseline LR은 MobileFaceNet `0.001`, ResNet-50/100
`0.0001`이다. 기본값 생성과 정식 fine-tune 검증에 함께 적용한다. 저장된 과거
ResNet LR `0.001`은 조용히 교체하지 않으며 정식 fine-tune에서 거부한다.
00의 새 설정을 확인하여 저장한 뒤 01~04에서 같은 설정 파일을 사용한다.
ResNet-152에는 이 논문의 LR를 임의로 외삽하지 않으며 이 baseline recipe는 지원하지 않는다.

`training.lr`는 downstream 설정이고 `campaign.pretrain_lr`/노트북 `PRETRAIN_LR`은
parent의 LR이다. parent 기본 `0.001`은 이번 수정 전 로컬 조건을 유지한 것이며,
UCFace가 parent 세부 조건을 따로 명시했다고 주장하지 않는다. 실제 parent 실행 시
별도 LR가 optimizer·checkpoint·campaign 실행 기록에 적용된다. 기존에 명시한 값은 유지한다.
MobileFaceNet·ResNet-50·ResNet-100 모두 baseline batch 64를 사용한다.
00에서 공통 기본값 64를 설정하며, 정식 fine-tune 검증에서 ResNet의 소규모 batch 예외를 허용하지 않는다.
과거 batch 16 설정 파일·checkpoint는 자동 변경하지 않는다. 00에서 새 조건을 저장한 뒤
다른 batch의 checkpoint를 resume하지 않고 새 run을 시작한다.
공통 downstream 조건은 Adam, 40 epochs, dropout 0.6, 10/20/30 epoch 이후 LR ×0.1이다.
2026-10-31 GTX 1080 Ti에서 ResNet-50/100의 batch 64 합성 입력 학습 점검은 통과했지만,
전체 데이터 학습 검증은 아니며 ResNet-100은 메모리 여유가 적다.

수정 후 커널을 재시작한다. LR가 다른 checkpoint의 resume는 기존 설정 비교에서 거부한다.
학습 코드 fingerprint도 바뀌므로 과거 parent를 자동으로 승인하지 않는다.
새 namespace에서 검증된 조건으로 실행하고 기존 결과를 덮어쓰지 않는다.

- 데이터 기본 루트: `F:/face_datasets` (`FACE_DATASET_ROOT` 환경변수).
- 모델 기본 루트: `F:/models` (`FACE_MODEL_ROOT` 환경변수).
- Linux 기본값: `/data/face_datasets`, `/data/models`. 서버 마운트 위치로 환경변수를 지정한다.
- FaceX-Zoo는 학습·export를 담당한다. ronbun은 완료된 export/sidecar와 protocol 파일을 소비한다.
  상대 저장소의 Python 코드를 import하지 않으며 Git 저장소를 합치지 않는다.
- 공유 protocol index: `<DATASET_ROOT>/protocols/shared_v1/handoff.json`.
  portable v2의 파일 경로는 해당 JSON 위치 기준이다. 이전 roles/train-list 바이트와 인물 hash를 유지한다.
- 새 모델/checkpoint: `<MODEL_ROOT>/facex/checkpoints/<namespace>`.
  완료 backbone export/metadata: `<MODEL_ROOT>/facex/exports/<namespace>`.
  로그와 campaign 보고서도 같은 facex 아래 `logs`, `reports`에 저장한다.
- 설정 파일: `<DATASET_ROOT>/.state/facex/shared_v1_boundary/notebook_settings_mobilefacenet.json`.
  Resnet은 설정 파일과 namespace를 별도로 지정한다. 모든 노트북에서 같은 명시적 설정 파일을 쓴다.
  00의 RUN_NAMESPACE=None은 backbone별 기본 namespace다. 전처리/학습 조건을 바꿀 때는
  새 명시적 이름을 지정한다. 이미 생성한 처리 결과를 다른 호스트로 옮긴 경우 절대 경로와
  코드/정책 검증을 우회하지 않는다. 기존 처리 결과를 사용할 수 없으면 새 namespace에서 재생성한다.

## 실행 순서

| 순서 | 노트북 | 동작 |
| --- | --- | --- |
| 00 | environment_and_data | 외부 root·3개 학습 데이터셋·backbone·head·seed를 설정하고 저장. 환경/계획 확인 |
| 01 | prepare_vggface2 | 공식 train 목록 생성, 필요하면 압축 해제와 제공 좌표 기반 정렬 |
| 02 | pretrain | 새 VGGFace2 MobileFaceNet/Resnet 1024D 부모 학습 |
| 03 | finetune | 외부 고정 역할 검증, resize-only 준비, 각 데이터셋/head/seed 학습 |
| 04 | export_and_validate | 완료 checkpoint 검증 후 backbone와 대응 metadata export |

00은 설정을 저장한다. 01–04의 `EXECUTE=False`는 계획 모드이며 학습/전처리/export를 실행하지 않는다.
모듈 변경 뒤에는 커널을 재시작하고 첫 셀부터 순차 실행한다. 설정 우선순위는 YAML < 저장값 < 현재 셀 override다.
00에서 외부 protocol이 없으면 준비 오류를 내며 기존 내부 경로로 조용히 돌아가지 않는다.
현재 로컬 데이터의 외부 protocol은 사전 migration으로 준비됐다. 다른 서버에는 외부 데이터 tree를 같이 복사/마운트한다.

VGGFace2의 archive/metadata는 `<DATASET_ROOT>/VGGFace2/data`, `meta`에 보존한다.
`EXTRACT_TRAIN_ARCHIVE=True`일 때 이미지는 `VGGFace2/extracted/official_train_v1/train`에 생성한다.
`VAL_EVERY=None`으로 공식 training 전체를 쓰며 별도 95:5 validation을 만들지 않는다.
전처리 결과는 각 데이터셋의 `processed/<namespace>`에 저장하고 PNG로 보존한다.
`PREPROCESS_MAX_IMAGES=32`는 부분 점검이며 학습 준비 완료가 아니다. 전체 준비에는 `None`을 지정한다.
부분 출력은 별도 audits 경로에 생성하고 완료 train 결과에 섞지 않는다.

## 고정 데이터 역할

| 데이터셋 | FR 학습 | 추가 역할 / 평가 |
| --- | --- | --- |
| SurvFace | 4,255명 / 176,205장 | FR 내 PQ-development 1,255명 / 52,227장; FR 제외 fit 744명 / 33,754장, safety 320명 / 10,929장. 공식 test 유지 |
| TinyFace | 2,570명 / 7,804장 | 동일 training을 PQ development로 사용. 공식 rank 평가 |
| SCFace | 50명 / 800장 | 다른 80명 gallery 80장 / probe 1,200장. rank 평가 |
| LFW·RFW | 사용하지 않음 | ronbun의 pair 보정·평가 전용 |

SurvFace calibration gallery와 등록 probe는 FR 학습 인물에서 구성되므로 calibration 전체가 FR-unseen은 아니다.
이 분할은 연구자가 고정한 조건이며 최적 비율이나 UCFace와 같은 성능을 뜻하지 않는다.
기본 3 dataset × 4 head × 5 seed는 60개 후속 모델과 60개 export다. 부모는 별도 1개다.

## 주요 설정과 보존

- MobileFaceNet 또는 Resnet IR-50, 1024D. 기존 F:/models의 512D 사전학습 파일과 구분한다.
- SEEDS 기본 8972–8976, HEADS는 ArcFace/AdaFace/MagFace/CosFace.
- PARENT_CHECKPOINT/PARENT_SHA256는 둘 다 지정하거나 둘 다 None. 자동 최신 checkpoint 선택은 없다.
- RESUME_CHECKPOINTS는 이번 namespace의 run ID → 정확한 checkpoint 경로다.
- KEEP_RAW_RESULTS=True는 이번 run의 상세 checkpoint를 보존한다. False는 Trainer의 현재 run 보존 정책을 따르며
  입력 dataset, 공유 전처리, 부모 가중치, 과거 실패 기록을 삭제하지 않는다. 최종 모델·집계·출처는 남긴다.
- 노트북 기본 `DATA_INTEGRITY_MODE="immutable_verified_reuse"`는 최초 전체 검증 이후 완료 JSON을 재사용한다.
  사용자는 검증된 원본·전처리 이미지와 목록을 임의로 변경하지 않는다는 전제다.
  매번 report/목록/manifest/정책·검증 코드의 일치 여부는 확인하지만 개별 이미지 파일은 다시 읽지 않는다.
  이미지 내용 변경·손상을 자동으로 탐지한다는 뜻은 아니다.
- 00–04 첫 설정 셀의 `FORCE_DATA_REVERIFY=False`는 일치하는 검증 기록을 재사용한다.
  `True`는 해당 검증 호출에서 전체 이미지 hash+decode를 강제한다. 사용 후 False로 되돌린다.
  Python/설정 JSON에서는 `data_integrity.force_reverify`에 같은 bool을 지정한다.
  `content_verified_reuse`는 종전처럼 매번 이미지 hash를 확인하고 decode만 재사용하며, `full`은 매번 둘 다 수행한다.
- ArcFace/MagFace의 target-only FP32 exact-boundary safeguard와 MagFace adaptive-margin graph를 유지한다.
  비유한 loss/gradient는 중단·기록한다. 이번 경로 변경으로 수식이나 augmentation 조건을 바꾸지 않는다.

## 생성 이력·완료 판단

새 추출/후처리 폴더에 README.md와 derivation.json을 같은 기록에서 생성한다.
원본 SHA-256, 실행 코드, 처리 조건, 개수와 완료/부분 상태를 기록한다.
후처리 manifest.jsonl은 각 이미지의 원본/출력 hash와 실패 원인을 담는다.
원본의 과거 생성 이력은 추정하지 않는다. 압축 해제 성공은 얼굴 검출이나 전 이미지 decode 성공을 뜻하지 않는다.
목적지 lock으로 동시 작성을 막고 완료 표시를 확인한 후 학습에 연결한다. lock을 자동 삭제하지 않는다.
완료 산출물과 다른 조건은 새 버전을 사용한다. 이 기록은 정식 학습이나 인식 성능 결과가 아니다.

### 검증 완료 기록과 중단 학습 재개

검증 기록은 전처리 report 옆 `verification_cache/immutable_v1/<계약 SHA-256>.json`에 저장된다.
`status=completed`, `verified_images`, 검증 계약·근거, 기록 자체의 hash가 있어야 재사용한다.
기존 `image-decode-receipt-v1`도 현재 report/목록/manifest/정책/검증 코드/라이브러리와
완전히 일치하면 최초 전체 검증의 근거로 가져온다. 기존 기록과 전처리 결과는 수정하지 않는다.
강제 검사 전의 기록은 `immutable_v1/history/`에 보존한다. 검사 실패·중단 기록이 있으면
예전 성공으로 돌아가지 않으며, 원인을 해결하고 `FORCE_DATA_REVERIFY=True`로 다시 검사한다.
완료된 전처리를 `RESUME_PREPROCESSING=True`로 열어도 같은 정책을 적용한다.
미완료 전처리의 복구는 기존 이미지별 검증·복구 절차를 유지한다.

기록은 준비 데이터의 재사용 cache이며 `KEEP_RAW_RESULTS` 삭제 대상이 아니다. 외부 dataset
root에 보존하므로 Git에 원본 이미지/상세 manifest를 추가하지 않는다. 실행 checkpoint/export의
`data_integrity_verification`에는 재사용 여부, 이번 이미지 검사 수, receipt 경로/hash와 한계를 남긴다.
`REUSED (0 image files read)` 출력은 검증 단계의 이미지 파일 재읽기가 없다는 뜻이다.
목록 확인·metadata hash·DataLoader/모델 초기화와 실제 학습용 이미지 읽기에는 별도 시간이 든다.

02를 중단할 때는 Interrupt 후 안전 저장 완료와 자식 프로세스 종료를 확인한다. 재개는
`RESUME_CHECKPOINTS={"pretrain_arcface_seed8972": r"정확한 복구 checkpoint 경로"}`로 지정한다.
기존 설정 JSON을 그대로 사용해도 02의 새 검증 변수 override가 적용되므로 00/01 재실행은 필요 없다.
모듈 수정 후 커널을 재시작하고 02 첫 셀부터 실행한다. 현재 노트북의 checkpoint 경로는
이번 안전 중단 파일로 명시되어 있으므로 다른 seed/namespace의 새 실행에는 해당 값을 수정한다.

이번 검증 정책 변경 전 checkpoint는 `configs/verification_resume_compatibility.json`의
정확한 이전→현재 코드 hash 조합에 한해서 재개를 허용한다. 모델·데이터·optimizer 등 학습
설정은 완전히 같아야 하며, 새 checkpoint의 `resume_source`에 전환과 원본 hash를 기록한다.
이 허용은 일반적인 코드/학습 조건 변경 우회가 아니다. 원본 checkpoint를 재작성하지 않는다.

기존 완료 run/report는 불변이다. 기본 경로는 새 namespace이므로 과거 shared_parent 결과를 재개하지 않는다.
GPU 정식 실행 전에 CUDA와 장치를 확인한다. Linux/A100 실제 실행 검증은 별도이며 Windows 경로 검증과 구분한다.
전체 fine-tune 테스트: `py -3.11 -m pytest fine-tune/tests -q` (저장소 루트).

## 입력 검증과 중단 복구 — 2026-10-29

학습 목록 cache는 `path_index_v2`를 사용한다. 재사용할 때 원본 목록·root·schema,
NPY의 SHA-256·행 수·dtype, 각 path/label의 원본 목록 일치를 검사한다. 이미지 전체를
다시 decode하는 검사는 아니며 목록과 cache를 순차 읽는다. 기존 v1 cache는 수정하지 않는다.
손상되거나 metadata가 없는 cache는 학습 전에 거절한다. 진단용 사본을 보존하고
`paths.cache_dir`에 새 경로를 지정해 다시 생성한다. 학습 중에는 목록·cache를 변경하지 않는다.

Export는 목적지별 writer lock을 사용한다. PT와 sidecar 각각의 게시만 원자적이며,
두 파일 전체가 하나의 원자적 거래인 것은 아니다. 중단으로 PT만 남았으면 같은 checkpoint·설정으로
04를 재실행한다(`campaign.reuse_completed=True`). 가중치와 내장 출처가 일치할 때에만
sidecar를 복구하며 기존 PT 바이트는 변경하지 않는다. 완료된 쌍은 검증 후 재사용한다.
다른 가중치·출처, 손상된 metadata, metadata만 남은 상태는 자동 덮어쓰기하지 않는다.
중단 후 lock이 남았다면 기록된 host/PID의 실행 종료를 확인한 후 해당 lock만 별도로 처리한다.

기존 DataLoader도 최종 1장 batch를 학습 전에 거절했다. 이번에는 잘못된 batch_size 입력과
진단 문구를 보완했다. 정식 학습은 `drop_last=False`를 유지하며 나머지가 0 또는 2 이상인
batch_size인지 검사한다. 현재 baseline batch 64에서 SurvFace 176,205장의 나머지는 13이다.
자동으로 샘플을 버리거나 batch를 합치지 않는다. 이 검증은 전체 학습의 무오류 보장이 아니다.

학습 중 `view` 등의 예외가 발생하면 traceback을 남기고 해당 학습과 남은 campaign 실행을 중단한다.
campaign 보고서에는 실패한 run과 시작하지 않은 run을 기록한다. 오류가 난 batch를 건너뛰지 않는다.
원인을 확인한 뒤 마지막으로 정상 저장된 checkpoint를 명시하여 재개한다.
오류 발생 직전 상태가 반드시 저장되는 것은 아니므로 마지막 저장 이후 계산은 다시 필요할 수 있다.


## 전처리 재개 (2026-11-02)

01/03에서 `EXECUTE=True`, `RUN_PREPROCESSING=True`, `PREPROCESS_MAX_IMAGES=None`,
`RESUME_PREPROCESSING=True`로 실행한다. 이미 archive 추출을 완료한 01은
`EXTRACT_TRAIN_ARCHIVE=False`를 사용한다. 커널을 재시작하고 설정 셀부터 실행한다.
CPU 정렬/리사이징 정책과 학습 데이터 분할은 변경하지 않는다.

- 새 출력: 재개 계약을 먼저 저장하고 전체 전처리를 시작한다.
- 중단 출력: 입력 목록 hash, 경로, label, 전처리 정책과 복구 코드 hash를 확인한다.
  시작 계약이 없던 구버전의 이미지는 원본을 다시 읽어 동일 변환의 픽셀과 대조한다.
  정상 PNG는 다시 쓰지 않는다. 이 최초 검증은 빠른 건너뛰기가 아니며 시간이 걸린다.
- 새 journal의 정상 행은 원본/출력 SHA-256, decode, 크기를 확인해 재사용한다.
  source bytes/분할/정책이 달라지면 중단한다. 임의로 계약 파일을 지우지 않는다.
- 잘린 마지막 JSONL 행은 보존하고 재처리한다. 중간의 손상된 JSON/순서 불일치는 중단한다.
  기록 없이 남은 PNG는 현재 변환과 픽셀 대조한다. 손상·다른 PNG는 recovery 시도 폴더에
  원본을 보존한 후 atomic replace한다. 학습 목록은 검증된 행으로 재구성한다.
- OS 잠금은 프로세스 종료 시 해제된다. 기존 artifact lock은 같은 호스트의 PID가
  없을 때만 별도 stale 파일로 보존하고 회수한다(psutil 필요). 살아 있는 PID나 다른
  호스트의 lock은 자동 제거하지 않는다. 노트북 중단 후 자식 프로세스가 살아 있으면 기다린다.
- 완료 결과는 수정하지 않고 전체 검증만 수행한다. report 게시 후 marker 직전의 중단은
  전체 검증 후 COMPLETED만 복구한다. 검증/변환은 CPU 작업이며 GPU 학습 완료와 무관하다.

출력의 `resume_contract.json`은 재개 계약, `recovery/<시도>/manifest.jsonl`과
`train_list.txt`는 매 시도의 상세 기록이다. 1,000행마다 flush/fsync하며 중단 직전의
미저장 구간은 다음 실행에서 픽셀 검증 또는 재처리한다. `previous-*`는 게시 전 기록,
`replaced/`는 교체된 이미지다. 자동 삭제하지 않으며 입력/전처리 cache이므로 학습의
`KEEP_RAW_RESULTS=False`로 삭제하지 않는다. 출력은 기존 외부 dataset root를 따르며
Git에 커밋하지 않는다. metadata 기록은 매 시도마다 늘어나므로 가용 공간을 확인한다.

`report.json`의 `total/processed/prepared/failed`는 전체 범위/처리/성공/실패 수,
`reused_images`는 PNG를 재작성하지 않은 수, `regenerated_images`는 새로 쓴 수다.
`completed=true`, `failed=0`, `COMPLETED`와 학습 전 검증 통과를 함께 확인한다.
부분 report는 학습에 사용할 수 없다. 학습/FPIR/TPIR 성능 결과를 뜻하지 않는다.
01은 변환 뒤 학습 입력 검증도 수행하므로 이미지 저장이 끝나도 셀이 즉시 끝나지 않을 수 있다.

전처리 자식 프로세스의 stdout/stderr는 노트북과 launch ticket의 `console.log`에 함께 기록한다.
실패 시 해당 로그의 원본 traceback을 확인한다. 커널 재시작으로 변경된 process_runner를 로드한다.
노트북에서 interrupt하면 자식 프로세스를 종료하고 산출물을 보존하며, 다음 실행에서 재개 검증한다.
