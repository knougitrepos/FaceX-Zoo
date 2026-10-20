# adjust1 baseline correction

## 기준과 범위

UCFace 본문·부록에서 명시한 값을 우선하고, 그 외 항목은 FaceX-Zoo conventional training을 기준으로 한다. 참고 채팅은 `01a14ff0-c555-7480-b3d5-5d72b30a87ce`의 최종 A 방식 합의다. 교정 시작점은 `365e939`, 저장소의 원본 main 참조는 `16b793a7564a4b9308cf94e62bdb2ffacb3a725a`다. 런타임에서 실제 모델/학습 코드의 SHA-256을 checkpoint 설정에 기록한다.

- 부모는 MobileFaceNet 1024D + ArcFace `m=0.35, margin_am=0, s=32`다. 이는 공개되지 않은 부모 설정에 FaceX-Zoo 기본값을 적용한다는 실험 선택이다. UCFace 부모의 실제 margin을 확인했다는 뜻은 아니다.
- 후속 학습은 새 부모의 backbone을 복사하고 네 head를 새로 초기화한다. ArcFace `m=.6,s=64`, CosFace `m=.3,s=64`, AdaFace `m=.4,h=.333,s=64`, MagFace `(10,110,.45,.8,20)`을 유지한다.
- MobileFaceNet 1024D/dropout .6, Adam, lr .001, batch 64, 40 epoch, milestones 10/20/30, decay .1을 유지한다. 부모에 공통 레시피를 적용하는 점, dropout 위치, 공통 부모 하나의 공유, seed 목록은 구현/실험 선택이다.
- flip + affine scaling/rotation은 부록을 따른다. 범위·확률·중심·padding은 공개되지 않은 세부값이며 노트북에서 설정하고 hash에 포함한다.
- A 방식은 로컬 구현이다. 비타깃 제곱근 연산을 제거하고 기존 cosine clamp·조건 분기·scale·MagFace norm 경로를 유지한다. 정답 `cosine=+/-1`의 특이점은 남는다. 기존 loss/gradient 검사로 optimizer 업데이트 전에 중단하며, 오염된 현재 상태를 recovery로 저장하지 않는다.
- AdaFace 고유의 정규화/clamp와 CosFace 수식은 변경하지 않는다. 상위 `head/` 및 SDK 원본은 수정하지 않고 `fine-tune/modules/target_margin.py`를 모델 생성기에 연결한다.

## 승인된 전처리 (2026-10-20)

사용자가 저해상도 세 데이터셋의 resize-only 처리를 명시적으로 승인했다. 이는 UCFace의 미공개 정렬 절차를 확인했다는 뜻이 아니다.

- **VGGFace2**: `VGGFace2/meta/bb_landmark.tar.gz`의 `bb_landmark/loose_landmark_train.csv`를 읽고, 이미지 이름으로 공식 추정 5점 좌표를 대응시킨다. 좌표는 배포 loose crop의 픽셀 좌표이며 FaceX-Zoo 원본 `FaceRecImageCropper`로 112×112 정렬한다. 새로운 검출기/랜드마크 추정 모델은 호출하지 않는다.
- **TinyFace·SurvFace·SCface**: 제공 얼굴 crop 전체를 112×112로 resize한다. 재검출, 랜드마크 정렬, 추가 crop, padding은 수행하지 않는다. 기본 보간법은 OpenCV `INTER_CUBIC`이다. `ALIGNMENT["low_resolution"]["interpolation"]`에서 `bicubic` 또는 `bilinear`를 명시하며 정책/hash에 기록한다.
- 112×112는 UCFace 부록 Augmentations에 명시된 입력 해상도다. SurvFace 원 논문 §5.1은 bicubic 모델 입력 크기 변환을 명시한다. TinyFace/SCface의 동일 보간법 및 정렬 생략은 사용자 승인 구현 선택이다.
- BGR 및 `(pixel - 127.5) / 128` 정규화는 유지한다. 기존 학습 증강은 변경하지 않으며, 평가에는 학습용 무작위 증강을 적용하지 않는다.
- 원본 데이터와 상위 FaceX-Zoo 소스는 수정하지 않는다. PNG와 목록·manifest·report는 `fine-tune/data/adjust1/preprocessed_v2/`에 별도로 만든다. 구 SDK 감사 결과는 보존하지만 새 입력으로 재사용하지 않는다.

## 통합 실행 순서

1. **새 커널에서 00**: 기본 설정을 검토하고 저장한다. 새 설정 파일은 `data/adjust1/notebook_settings_v2.json`이다. 버전은 `adjust1_preprocessing_v2`이며 이전 설정·체크포인트를 자동 승격하지 않는다.
2. **01**: `EXECUTE=True`로 VGGFace2 공식 training 전체 목록을 준비한다. 압축 해제를 이미 완료했다면 `EXTRACT_TRAIN_ARCHIVE=False`를 유지한다. `VAL_EVERY=None`, `val_list=None`이다.
3. **01 전처리**: `RUN_PREPROCESSING=True`, `PREPROCESS_MAX_IMAGES=32`로 소규모 감사를 수행한다. 전체 준비에는 `PREPROCESS_MAX_IMAGES=None`을 사용한다. 랜드마크 CSV는 임시 SQLite 색인으로 조회하며 모든 행의 형식·유한값·중복 키를 검사한다. 색인은 `fine-tune` 캐시에 만들고 정상 종료 시 제거한다. 표본 감사는 전체 이미지 성공을 의미하지 않는다.
4. **02**: 전체 준비 report와 원본/출력 hash 검증 후 새 부모를 학습한다. 기본 `EXECUTE=False`이며 학습 실행은 별도로 선택한다. `RUN_BENCHMARK=True`도 준비된 부모 입력을 사용한다.
5. **03**: 공식 training/evaluation manifest와 신원 분리를 검증한다. `EXECUTE=True`, `RUN_PREPROCESSING=True`, `PREPROCESS_MAX_IMAGES=32`이면 선택한 데이터셋의 resize 감사만 수행하고 학습은 건너뛴다. 전체 준비와 후속 학습에는 `PREPROCESS_MAX_IMAGES=None`을 사용한다. 전체 준비를 마쳤다면 `RUN_PREPROCESSING=False`로 준비 결과를 재사용한다. SCface는 확보·프로토콜 검증 전까지 비활성이다.
6. **04**: 학습 시 전처리·보간법·입력 manifest·코드 hash가 일치하는 checkpoint만 export한다. 부모는 공식 좌표 정렬 정책으로, 후속 checkpoint는 resize-only 정책으로 각각 검증한다. export metadata에는 실제 후속 입력 정책이 보존된다. 평가 측에서도 같은 전체-crop resize/보간법/BGR/정규화를 사용해야 한다. 외부 평가 저장소 수정은 이번 범위에 포함하지 않는다.

CLI도 동일한 경로를 호출한다. 기존 함수/스크립트의 `alignment` 명칭은 호출 호환성을 위해 유지하며 report의 `prepared`는 정렬과 resize를 모두 포함한 준비 성공 수다.

```powershell
python fine-tune/scripts/prepare_alignment.py --settings fine-tune/data/adjust1/notebook_settings_v2.json --dataset vggface2 --max-images 32 --output-dir fine-tune/data/adjust1/audits/vggface2_v2_001
python fine-tune/scripts/prepare_alignment.py --settings fine-tune/data/adjust1/notebook_settings_v2.json --dataset tinyface --max-images 32 --output-dir fine-tune/data/adjust1/audits/tinyface_v2_001
```

전체 준비는 `--max-images`와 `--output-dir`을 생략한다. 이미 존재하는 출력 폴더는 덮어쓰지 않는다. 중단/실패한 전체 준비를 다시 실행할 때는 새 출력 경로를 지정하고 설정의 `alignment_report`도 그 경로로 맞춘다.

## 실패 및 검증

재검출 결과에 따른 통과/탈락 규칙은 없다. 이미지 decode 실패, 좌표 누락·중복, 파일 쓰기 실패 등은 실제 입력 오류로 보고한다. 이미지별 처리 실패는 manifest에 남기고, 완전한 준비가 아닌 결과를 학습에 넘기지 않는다. 학습 목록에서 조용히 삭제하거나 다른 전처리로 대체하지 않는다.

마지막 불완전 배치 유지, 비정상 loss/gradient 중단, 모델·설정 hash 기반 재개 검증은 기존대로 유지한다. 구 `shared_parent_v3` 결과와 SDK 감사 기록도 보존한다.

검증 명령은 `python -m pytest fine-tune/tests -q`다. 실제 원본 표본 처리와 합성 이미지 기반 부모→후속 네 head→export 검증을 구분해 보고한다. 테스트 통과는 전체 전처리·정식 학습·인식 성능 검증 완료를 의미하지 않는다.
