# FaceX-Zoo 노트북 실행 안내

사용자 설정은 **노트북 첫 코드 셀**에서 조절한다. YAML은 누락값을 위한 기본값이며 직접 편집할 필요가 없다.

## 설정 우선순위와 실행 순서

`YAML 기본값 < 00에서 저장한 공통 설정 < 현재 노트북 첫 셀의 변수`

1. `00_environment_and_data.ipynb`: 첫 셀에서 공통 변수를 지정하고 설정 저장 셀까지 실행한다. 공통값은 `fine-tune/data/notebook_settings.json`에 저장된다. 뒤 노트북에서 공통값을 바꾸려면 00을 다시 저장하거나 해당 노트북의 변수로 덮어쓴다. 개별 노트북 변경은 다른 단계에 자동 전파하지 않는다.
2. `01_prepare_vggface2.ipynb`: `EXECUTE=False`로 경로를 확인한 후 `True`로 목록을 생성한다. 실제 이미지를 먼저 압축 해제해야 한다. 기존 목록과 내용이 다르면 덮어쓰지 않고 중단한다. 분할을 바꾸려면 새 train/val 목록 경로를 사용한다.
3. `02_pretrain.ipynb`: seed별 ArcFace 부모를 순차 학습한다. `EXECUTE=True`, `RUN_SMOKE_TEST=False`가 정식 학습이다.
4. `03_finetune.ipynb`: 각 seed의 같은 부모에서 ArcFace → AdaFace → MagFace를 순차 학습한다. 여러 모델을 동시에 GPU에 올리지 않는다.
5. `04_export_and_validate.ipynb`: 같은 head × seed 목록을 순차 export한다. 학습 당시 전처리·학습 설정·출처 hash를 함께 기록한다.

01~04의 초기값 `EXECUTE=False`는 계획 모드다. 단순 Run All로 장기 학습이나 export를 시작하지 않는다. 실제 실행하려면 해당 첫 셀에서 True로 바꾸고 순서대로 실행한다. Python 모듈 수정 후에는 반드시 커널을 재시작한다.

## 주요 사용자 변수

| 변수 | 기본/역할 | 주의점 |
| --- | --- | --- |
| `SEEDS` | `[8972,8973,8974,8975,8976]` | 실제 반복 실행한다. 한 seed만 필요하면 `[8972]`. 다음 단계도 같은 seed 목록을 사용한다. |
| `HEADS` | 세 head 모두 | 03/04에서 실제 반복한다. 02는 공통 ArcFace 부모만 생성한다. |
| `KEEP_RAW_RESULTS` | `True` | 현재 노트북에서 변경하면 Trainer에 전달된다. False는 이번 run의 이전 중간 periodic checkpoint를 정리한다. 입력 이미지·부모 가중치 삭제 옵션이 아니다. |
| `NUM_WORKERS` | `0` | Windows/Jupyter 경로는 0만 허용한다. 1 이상도 subprocess를 사용하므로 '2 미만이면 안전'으로 가정하지 않는다. CLI 다중 worker 최적화는 이번 범위 밖이다. |
| `DATASET` | 입력 root, train/val 목록, 크기, mean/std, 클래스 수 | 실제 loader에 전달된다. 현재 MobileFaceNet/export는 112×112만 지원하므로 다른 크기는 명시적으로 차단한다. 값이 조용히 무시되지 않는다. |
| `TRAINING` | Adam, lr=0.001, 40 epoch, batch=64 등 | 논문 대응 조건을 유지한다. 변경하면 별도 실험이며 기존 checkpoint 재개 검증에서 거부될 수 있다. |
| `BACKBONE` / `HEAD_PARAMETERS` | 차원, dropout, 세 head의 파라미터 | 00 첫 셀에 실제 값이 보인다. 본문 미명시 설정은 본 연구 선택이다. |
| `PATHS` | checkpoint/export/report 위치 | F 드라이브 기본. 다른 실험 조건은 새 checkpoint/export root를 사용한다. |
| `RUN_SMOKE_TEST` | `False` | 02/03에서 True면 더미 입력 forward/backward/optimizer 1 step만 실행한다. 정식 checkpoint는 생성하지 않는다. |
| `RESUME_CHECKPOINTS` | `{}` | run ID → 중단 checkpoint 경로. 예: `{"finetune_adaface_seed8972": "F:/.../finetune_mobilefacenet_1024d_adaface_epoch_5.pt"}`. 자동 최신 파일 선택은 없다. |
| `REUSE_COMPLETED` | `True` | 명시된 run의 완료·설정·출처를 검증한 뒤 재사용한다. 기존 파일을 다시 학습하거나 덮어쓰지 않는다. |
| `VAL_EVERY` | `20` (01만) | 20번째마다 validation으로 분리하는 기존 정책. 이미지 단위 분할이며 신원 분리 평가가 아니다. |

기본 계획은 부모 5개, 후속 모델 15개, export 15개다. 이것은 실제 완료 수가 아니다. 03은 동일 seed의 부모 hash를 공유한다. 반복 seed는 사용자 지정값이며 논문 저자의 seed가 아니다.

## 재개·저장·실패 처리

- 완료 run을 다시 실행하면 설정·데이터 목록 hash·부모·seed·head·stage를 검사하고 일치하는 경우에만 재사용한다.
- 미완료 run 폴더에 checkpoint가 있으면 `RESUME_CHECKPOINTS`로 명시적으로 재개한다. 실패 후 01을 다시 실행하여 공유 목록을 바꾸지 않는다.
- 첫 실패에서 후속 작업을 중단한다. `reports/campaigns/<stage>_<KST 시각>_<식별자>.json`에 완료·실패·미실행 run과 실제 설정을 기록한다. 중단 후 새 실행의 summary는 별도 파일이다.
- 원본 미보존 설정에서도 완료 모델·출처와 집계 summary는 남긴다. 삭제된 중간 checkpoint 시점으로 되돌아갈 수 없으며, 초기 학습을 다시 해야 할 수 있다.
- `notebook_settings.json`은 사용자의 변경 가능한 설정이다. run 폴더의 설정 사본과 완료 모델은 기존 결과로 보존한다. 설정을 바꾸면 이미 완료된 run을 소급 변경하지 않는다.
- 코드 변경 전 기존 사용자 노트북과 실행 출력은 `fine-tune/data/notebook_backups/`에 보존했다. 이 폴더는 Git 제외 대상이며 입력 데이터/연구 결과와 구분한다.

## 환경 확인과 결과 해석

00은 CUDA 사용 가능 여부, 장치, branch, 파일 존재·크기와 메타데이터 identity 수를 표시한다. 파일 존재를 압축파일 전체 무결성 검증이라고 표시하지 않는다. `archive_integrity`와 `all_images_decode`는 실행하지 않았다면 `not_checked`다. 환경 점검은 정식 학습 완료를 의미하지 않는다.

01의 loader 확인은 일부 이미지의 존재·decode·전처리를 확인한다. 전체 이미지가 모두 정상이라는 증거는 아니다. 이후 누락/손상 이미지가 나타나면 학습은 오류로 중단된다.

캠페인 JSON의 `planned`, `completed`, `rows`, `not_started`를 함께 읽는다. `smoke`의 `completed=False`는 의도적이며 성능 평가 결과가 아니다. 각 run의 checkpoint SHA-256·config hash·부모 hash와 실제 보존 설정을 확인한다. 이 summary는 원본 점수 전체를 묶은 ZIP이 아니라 실행 상태·출처의 경량 집계다. 전체 개요는 캠페인별 JSON 하나부터 공유하고, 모델 세부 분석에는 해당 export의 `_meta.json`을 함께 사용한다. 학습별 상세 sample score는 이 workflow에서 생성하지 않는다.

04는 순수 backbone strict-load/shape를 검사한다. 실제 얼굴인식 지표, ronbun loader 일치, Continuous FIQA, TPIR/FPIR 평가는 여기서 수행하지 않는다. 메타데이터에도 미실행 상태를 남긴다.

## UCFace와의 대응 범위

MobileFaceNet 1024D, VGGFace2, Adam, batch 64, lr=0.001, 40 epoch와 주요 head 설정을 논문 IV-A/Table I에 맞춘다. 공통 ArcFace 부모, VGGFace2 95:5 이미지 분할, 같은 데이터의 40+40 학습, dropout 위치·전처리 세부는 현재의 명시적 구현 선택이다. 노트북 제어 개선만으로 저자의 미공개 recipe까지 정확히 재현한 것이 되지는 않는다. UCFace 추가 contrastive loss는 이번 baseline 생성 범위에 포함하지 않는다.

- [UCFace 공식 논문](https://ieeexplore.ieee.org/document/10595448)
- [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo)
- [AdaFace 공식 코드](https://github.com/mk-minchul/AdaFace)
- [PyTorch Windows FAQ](https://docs.pytorch.org/docs/2.7/notes/windows.html)

## 검증 명령

```powershell
py -3.11 -B -m pytest fine-tune/tests -q -p no:cacheprovider
```

단위 테스트, 임시 소량 이미지 학습, 전체 VGGFace2 정식 학습은 구분한다. 실제 검증 결과는 해당 작업 보고서에 기록한다.
