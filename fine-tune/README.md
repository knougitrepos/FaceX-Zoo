# FaceX-Zoo 모델 생성: 외부 데이터·모델 저장소

현재 실행 안내다. 과거 adjust1/shared_parent_v3 안내는
[기록용 안내](docs/legacy/README_before_shared_storage_20261028.md)에 보존했다.
과거 실패 조건의 fine-tune 결과는 기록용이며 새 실험에서 자동으로 선택하지 않는다.

## 저장 위치와 독립성

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
- data_integrity=content_verified_reuse는 현재 이미지 바이트 hash를 확인한 뒤 같은 계약의 decode 검사만 재사용한다.
- ArcFace/MagFace의 target-only FP32 exact-boundary safeguard와 MagFace adaptive-margin graph를 유지한다.
  비유한 loss/gradient는 중단·기록한다. 이번 경로 변경으로 수식이나 augmentation 조건을 바꾸지 않는다.

## 생성 이력·완료 판단

새 추출/후처리 폴더에 README.md와 derivation.json을 같은 기록에서 생성한다.
원본 SHA-256, 실행 코드, 처리 조건, 개수와 완료/부분 상태를 기록한다.
후처리 manifest.jsonl은 각 이미지의 원본/출력 hash와 실패 원인을 담는다.
원본의 과거 생성 이력은 추정하지 않는다. 압축 해제 성공은 얼굴 검출이나 전 이미지 decode 성공을 뜻하지 않는다.
목적지 lock으로 동시 작성을 막고 완료 표시를 확인한 후 학습에 연결한다. lock을 자동 삭제하지 않는다.
완료 산출물과 다른 조건은 새 버전을 사용한다. 이 기록은 정식 학습이나 인식 성능 결과가 아니다.

기존 완료 run/report는 불변이다. 기본 경로는 새 namespace이므로 과거 shared_parent 결과를 재개하지 않는다.
GPU 정식 실행 전에 CUDA와 장치를 확인한다. Linux/A100 실제 실행 검증은 별도이며 Windows 경로 검증과 구분한다.
전체 fine-tune 테스트: `py -3.11 -m pytest fine-tune/tests -q` (저장소 루트).
