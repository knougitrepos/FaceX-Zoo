# SCFace 확보 후 별도 후속 학습 안내

현재 SCFace는 미확보 상태로 실행이 차단되어 있다. 데이터 확보 후 아래 준비와 설정을 완료하면 기존 공통 부모에서 SCFace 모델만 추가 생성할 수 있다. 데이터 복사만으로 자동 준비되지는 않는다.

## 1. 데이터와 분할 준비

- 기본 원본 경로는 `F:/FaceX-Zoo/scface`다. 다른 위치라면 아래 `raw_root`를 바꾼다.
- UCFace 보충자료 II-A/Table IX 기준으로 **학습 50명/800장**과 신원이 겹치지 않는 평가 목록을 준비한다. 수량만 맞춘 임의 분할을 논문과 동일한 분할이라고 표현하지 않는다.
- 다음 CSV 두 개를 준비한다. 이미지 경로는 `raw_root` 기준 상대 경로이고, `source_identity_id`는 학습·평가에 공통으로 적용되는 원래 인물 ID다. 각 split에서 새로 부여한 label을 원래 ID 대신 사용하지 않는다.
  - `fine-tune/data/protocols/local_qmul_v1/scface/training_manifest.csv`: 필수 열 `image_path,source_identity_id,protocol_role,split`; 모든 행의 `protocol_role=training`, `split=training`.
  - 같은 폴더의 `evaluation_manifest.csv`: 필수 열 `image_path,source_identity_id`. 공식 평가 대상 이미지와 원래 인물 ID를 기록한다.
- SCFace의 전체 이미지 존재/decode 및 분할의 출처를 별도로 검증하고 기록한 뒤 활성화한다. 현재 TinyFace·SurvFace용 자동 raw 폴더/MAT 전체 감사는 SCFace에 구현되어 있지 않다. SCFace 공통 protocol 검사는 manifest 수량·신원 분리·hash 등을 확인하므로 이 검사만으로 전체 이미지 검증이 완료되었다고 판단하지 않는다.
- 최종 평가 데이터를 FR 학습이나 checkpoint 선택에 넣지 않는다. FIQA 평가에서도 데이터셋의 공식 프로토콜을 유지한다.

## 2. 00 노트북의 사용자 설정

`00_environment_and_data.ipynb` 첫 설정 셀에서 기존 두 줄을 아래와 같이 바꾼다. 반드시 `settings = resolve_settings(...)` 호출 **앞**에 둔다.

```python
FINETUNE_DATASETS = ["scface"]  # SCFace만 추가 생성
FINETUNE_PROFILES = default_profiles(BASE_DIR)
FINETUNE_PROFILES["scface"].update(
    enabled=True,  # 위 데이터·분할 검증을 마친 뒤에만 활성화
    raw_root="F:/FaceX-Zoo/scface",
)
```

| 사용자 변수 | 설정 및 주의사항 |
| --- | --- |
| `HEADS` | `["arcface", "adaface", "magface", "cosface"]` 유지 |
| `SEEDS` | `[8972, 8973, 8974, 8975, 8976]` 유지 시 SCFace 모델 20개 |
| `PARENT_CHECKPOINT`, `PARENT_SHA256` | 기존 공통 부모와 hash 유지. `None`이면 기존 `PATHS`와 `PARENT_SEED`가 지정하는 02 부모를 사용한다. |
| `PATHS` | 기존 공통 부모를 사용할 때 checkpoint root를 임의 변경하지 않는다. SCFace 출력은 데이터셋 이름으로 자동 분리된다. |
| `KEEP_RAW_RESULTS` | 상세 학습 산출물 보존 여부. 원본 데이터·공통 부모를 삭제하는 옵션이 아니다. |
| `NUM_WORKERS` / `PREFETCH_FACTOR` | Windows 현재 권장 4 / 4. 별도 학습 프로세스로 실행하며 문제가 있으면 worker 0으로 진단한다. |

클래스 수는 SCFace 프로필의 50으로 후속 학습 시 적용된다. VGGFace2용 `DATASET.num_classes`나 네 head의 기본 클래스 수를 SCFace 때문에 50으로 바꾸지 않는다. Backbone·학습 일정·head 파라미터는 기존 논문 대응 설정을 유지한다.

## 3. 실행 순서

1. 커널을 재시작하고 **00을 실행해 설정을 저장**한다.
2. 기존 공통 부모가 완료되고 출처/hash가 맞으면 **01·02를 다시 수행할 필요는 없다**. 부모가 없거나 미완료라면 먼저 VGGFace2 준비와 02 사전학습을 완료한다.
3. **03** 첫 셀에서 `FINETUNE_DATASETS=["scface"]`, 활성화된 프로필과 부모를 확인한다. `EXECUTE=True`, `RUN_SMOKE_TEST=False`로 실행한다. 분할 검증이 성공해야 학습이 진행된다.
4. **04**에서도 같은 SCFace 설정을 확인하고 `EXECUTE=True`로 완료 모델을 export한다.

03/04는 00에서 저장한 설정을 기본으로 읽는다. 나중에 다시 00을 실행할 때 `default_profiles(BASE_DIR)`만 사용하면 SCFace는 다시 비활성화되므로 위 활성화 설정을 유지해야 한다. 기존 TinyFace·SurvFace 모델은 별도 run 경로로 보존되며 SCFace 추가를 이유로 재학습하지 않는다.

실행/완료 모델/export와 실제 얼굴인식·Continuous FIQA 평가는 구분한다. 전체 공통 실행 안내는 [README.md](README.md)를 따른다.
