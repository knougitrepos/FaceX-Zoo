# FaceX-Zoo 연구 목표 및 작업 기준

- 작성일: 2026-10-09 KST / 학습 구조·최적화 기준 갱신: 2026-10-11 KST / CosFace 추가: 2026-10-13 KST
- 작업 저장소: `F:\FaceX-Zoo`
- 작업 브랜치: **`add-adaface`**
- 현재 상태: **구현 및 검증 중**. 전체 학습 완료나 UCFace 수치 재현을 의미하지 않는다.
- 문서 역할: 목표, 실험 기준, 시스템 제약, 작업 범위와 완료 조건을 정한다. 실제 구현·실행 상태는 코드, 설정, 테스트와 run manifest로 확인한다.
- 이번 문서 작성은 학습 실행이나 Codex의 활성 goal 등록이 아니다.

## 1. 최종 목표

FaceX-Zoo를 공통 학습 기반으로 사용하고 공식 AdaFace head를 통합하여, **MobileFaceNet 1024D + ArcFace / AdaFace / MagFace / CosFace**의 학습 완료 baseline 4종을 생성한다. VGGFace2를 사용하며, UCFace 논문 본문에서 확인할 수 있는 baseline 학습 조건을 최대한 맞춘다.

학습 완료 checkpoint는 `C:\ronbun`에서 가중치를 고정한 특징 추출기로 사용한다. 고화질·저화질 평가 데이터와 원본·압축 임베딩에서 **Continuous FIQA 기반 사후 임계값 보정의 적용 전후 효과**를 평가하는 것이 최종 연구 목적이다.

책임을 다음과 같이 분리한다.

| 위치 | 책임 |
| --- | --- |
| `F:\FaceX-Zoo` | 데이터 준비, baseline 학습, 검증, 재개, checkpoint export |
| `F:\AdaFace` | 공식 AdaFace 구현과 라이선스·출처 참조 |
| `F:\FaceX-Zoo\VGGFace2` | 확보한 원본 데이터와 메타데이터 보존 |
| `C:\ronbun` | 모델 등록·추론, 압축, 품질 기반 임계값 보정, 평가·결과 분석 |

이번 범위는 **UCFace 논문의 비교 baseline을 참고한 FR 모델 생성**이다. UCFace의 불확실성 모듈, projection head, 추가 contrastive loss까지 구현하는 것은 현재 목표에 포함하지 않는다. baseline 생성만으로 `UCFace 방법 재현 완료`라고 표시하지 않는다.

## 2. 근거의 우선순위와 재현 범위

1. **UCFace 논문 본문을 최우선 기준으로 삼는다.** 본문이 직접 참조한 보충자료를 확보한 경우 함께 대조한다.
2. 저자 이메일은 보조 자료다. 기억에 의존한 답변을 본문보다 우선하거나 본문의 충돌을 이메일로 덮어쓰지 않는다.
3. 본문에 없는 사항은 공식 구현을 참고하거나 본 연구의 공학적 선택으로 정하고 출처를 기록한다.
4. 확인되지 않은 학습 단계·데이터 분할·초기화·전처리 설정을 추정한 뒤 논문 명시 사항으로 표기하지 않는다.
5. 저자에게 추가 질문하는 것을 진행의 필수 조건으로 두지 않는다. 확인 가능한 근거와 명시적인 구현 선택으로 진행한다.

주요 근거:

- 논문: *Uncertainty-Aware Face Embedding With Contrastive Learning for Open-Set Evaluation*, IEEE TIFS 2024, DOI `10.1109/TIFS.2024.3426973`.
- 로컬 논문: `F:\FaceX-Zoo\paper\UCFace.pdf`, IV-A, PDF 6쪽 / 인쇄 쪽수 7181.
- [논문 공식 페이지](https://ieeexplore.ieee.org/document/10595448)
- [FaceX-Zoo upstream](https://github.com/JDAI-CV/FaceX-Zoo)
- [AdaFace 공식 구현](https://github.com/mk-minchul/AdaFace)
- [현재 지정한 VGGFace2 배포본](https://huggingface.co/datasets/ProgramComputer/VGGFace2/tree/main)

소스 commit, 데이터 배포 revision, 원본 및 파생 파일 hash를 기록한다. 위 VGGFace2 배포본은 제3자 배포 경로이며, 저자가 사용한 파일과 동일하다는 사실은 별도 검증 없이 단정하지 않는다.

## 3. 논문에 맞출 주요 설정

| 항목 | 목표 설정 | 근거 및 주의사항 |
| --- | --- | --- |
| Backbone | MobileFaceNet | Table I의 비교 backbone |
| 임베딩 차원 | **1024D** | IV-A-3: 주요 실험은 1024D. 이메일의 512D를 기본으로 채택하지 않음 |
| 사전학습 데이터 | **VGGFace2** | 본문에 제시된 사전학습 데이터 중 이번 연구에서 선택. MS1MV3는 추가하지 않음 |
| 후속 학습 classifier | ArcFace / AdaFace / MagFace / CosFace | 학습된 backbone에 새 classifier를 연결하는 단계 구분 |
| Optimizer | Adam | IV-A-3 |
| MobileFaceNet 학습률 | `1e-3` | IV-A-3 |
| 학습률 감소 | 10 epoch마다 0.1배 | 40 epoch 실행에서 milestones `[10, 20, 30]`로 구성 |
| Baseline batch size | 64 | softmax classifier 학습 기준. 장비에서 실제 검증 필요 |
| 학습 epoch | 40 | 논문 대응 baseline 학습 설정. 사전학습의 전체 세부 일정까지 동일하다고 단정하지 않음 |
| Dropout 비율 | 0.6 | 비율은 본문 명시. 삽입 위치는 구현 선택으로 별도 기록 |
| ArcFace, 저해상도 설정 | `m=0.6`, `s=64` | IV-A-2 |
| CosFace, 저해상도 설정 | `m=0.3`, `s=64` | IV-A-2. 기존 `head/AM_Softmax.py` 재사용 |
| AdaFace, 저해상도 설정 | `m=0.4`, `s=64`, `h=0.333` | IV-A-2 |
| MagFace | `(l_a,u_a,l_m,u_m,lambda_g)=(10,110,0.45,0.8,20)` | IV-A-2. 추가 설정의 출처는 공식 구현과 구분 |
| AdaFace EMA 계수 | `t_alpha=0.01` | 공식 구현 사용 예시에서 채택한 값. UCFace 본문에 명시된 값으로 표현하지 않음 |

고해상도·혼합 해상도 조건은 본문에서 각 원 논문의 hyperparameter를 따른다고 설명한다. 저해상도 설정을 그대로 사용해 LFW 등을 평가할 경우, 그것은 본 연구의 고정 checkpoint 교차 데이터셋 평가이며 논문의 모든 표와 동일한 조건이라는 뜻은 아니다.

입력 `112×112`와 horizontal flip·affine(rotation/scaling)은 보충자료 PDF 3쪽에서 확인된다. BGR, `(pixel - 127.5) / 128`, weight decay, augmentation의 수치 범위, dropout 위치, 사전학습 loss·일정의 세부는 **공식 코드 / 본 연구 선택**으로 따로 기록한다. 단순 resize를 얼굴 정렬로 표시하지 않는다.

## 4. Seed 정책: 기본값 8972

**논문에서 별도의 seed 숫자를 지정하지 않은 경우 기본 seed는 `8972`로 한다.** 현재 확보한 본문에는 구체적인 seed 숫자가 확인되지 않았다. `8972`는 사용자 지정 재현성 값이며 논문 저자의 seed로 표현하지 않는다.

- 사용자 설정은 노트북 첫 설정 셀이 우선이다. 우선순위는 `YAML 기본값 < 00의 저장 설정 < 현재 노트북 변수`다. 기본 seed는 `8972`다.
- Python `random`, NumPy, PyTorch CPU/CUDA, DataLoader generator 및 worker 초기화가 같은 실행 seed 정책을 따르도록 한다.
- seed 초기화는 모델 생성, 데이터 shuffle 및 augmentation에 앞서 수행한다.
- 노트북의 기본값, Python 함수 기본값, fallback 값, 실행 안내를 통일한다. `42`를 별도 기본값으로 남기지 않는다.
- 실제 seed, 분할 hash, 라이브러리 버전, deterministic 관련 옵션, GPU 정보를 manifest에 기록한다. seed만 같다고 환경 간 bitwise 동일성을 보장하지 않는다.
- 반복 실험 시 각 run의 seed를 고유하게 지정한다. 논문의 5회 반복에 대응하는 본 프로젝트의 기본 학습 seed 목록은 **`[8972, 8973, 8974, 8975, 8976]`**으로 사전 선언한다. 이 목록 역시 본 연구 선택이다.
- **사전학습 부모는 1개만 생성한다.** `PARENT_SEED=8972`로 생성한 동일 backbone·SHA-256을 모든 후속 학습이 공유한다. 후속 학습의 `SEEDS`는 새 classifier 초기화·shuffle·augmentation의 반복이며 부모 5개를 뜻하지 않는다.
- 논문 표마다 반복 횟수가 다를 수 있으므로 모든 실험에 5회를 일괄 적용하지 않는다. 학습 5회, 평가 fold, calibration split 반복을 서로 대체하거나 독립 표본처럼 합치지 않는다.

공통 부모 공유는 사용자가 승인한 연구 설계다. 논문은 pretrained backbone에 무작위 classifier를 연결하고 5회 반복한다고 설명하지만, 각 반복·head가 정확히 같은 부모 파일을 공유했다고 명시하지는 않는다. 이를 저자의 확인된 세부 절차로 표현하지 않는다. 기존 완료 결과는 seed·구조 변경을 이유로 덮어쓰거나 재명명하지 않는다.

## 5. 학습 단계와 데이터 계약

### 5.1 데이터 준비

- 원본 압축파일·메타데이터의 출처와 무결성을 확인하고, 이미지 수·identity 수·손상·누락을 집계한다.
- Git clone, 압축파일 존재, 이미지 표본 읽기를 전체 데이터 검증 완료로 간주하지 않는다.
- 실제 압축 내부 경로와 loader의 `data_root`·인덱스 경로가 일치해야 한다.
- 네 모델은 같은 유효 이미지 목록, identity-label 매핑, 분할과 전처리를 사용한다.
- 누락·손상 이미지를 영벡터나 난수 이미지로 바꾸어 정식 학습하지 않는다. 검사 실패 시 중단하거나, 사전 정의된 제외 정책과 제외 목록을 기록한다.
- 단순 resize와 landmark 기반 얼굴 정렬을 구분한다. 정렬을 구현하지 않았으면 `정렬 완료`라고 기록하지 않는다.
- train/validation 역할을 명시한다. 같은 identity의 이미지 분할은 분류기 검증으로 사용할 수 있으나, 미관측 identity 평가라고 표현하지 않는다.
- 최종 평가 test를 FR 학습·checkpoint 선택·압축기 학습·FIQA calibration에 사용하지 않는다. VGGFace2와 평가 데이터의 identity overlap 확인 결과 또는 확인 한계를 남긴다.

### 5.2 사전학습 → fine-tuning

- 무작위 초기화에서 학습하는 단계와 부모 checkpoint를 불러와 추가 학습하는 단계를 분리한다.
- `pretrain`, `finetune`, `export`의 입력·출력·설정·완료 상태를 각각 기록한다.
- fine-tuning은 명시적으로 선택한 부모 checkpoint, 기대 SHA-256, 학습 완료 출처를 요구한다. hash 일치만으로 학습 완료를 판단하지 않는다.
- 사전학습 loss와 데이터, 후속 학습 데이터·분할, backbone 업데이트 여부, classifier 초기화 정책을 실행 전에 고정한다.
- **공통 MobileFaceNet 부모 1개를 VGGFace2로 사전학습하고, 그 backbone을 독립적으로 복사하여 대상 저해상도 데이터셋의 공식 training split에서 ArcFace / AdaFace / MagFace / CosFace를 각각 fine-tuning한다.** backbone도 업데이트하고 classifier는 대상 training identity 수에 맞게 새로 무작위 초기화한다. head 간 순차 전달 학습을 하지 않는다.
- 02의 사전학습 loss는 ArcFace, 기본 일정은 40 epoch다. 이는 세부 사전학습 recipe가 공개되지 않아 채택한 구현 선택이며, 논문이 사전학습을 반드시 ArcFace로 40 epoch 하라고 명시한 것은 아니다. `PRETRAIN_EPOCHS`와 후속 학습의 논문 명시 40 epoch를 구분한다.
- 03은 VGGFace2의 약 298만 장 목록을 다시 학습하지 않는다. 보충자료 II-A 및 Table IX의 대상 training split을 사용한다.

| 후속 학습 대상 | Training identity | Training image | 기본 실행 범위 |
| --- | ---: | ---: | --- |
| QMUL-SurvFace | 5,319 | 220,888 | 우선 대상으로 선택 |
| QMUL-TinyFace | 2,570 | 7,804 | F 드라이브 공식 원본에서 검증·연결 |
| SCFace | 50 | 800 | 미확보: 학습 비활성화·선택 차단 |

- 현재 기본 계획: 사전학습 부모 **1개**, **TinyFace·SurvFace × 4 head × 5 seed = 40개**, export 40개. SCFace는 확보·검증 후 명시적으로 활성화하기 전까지 제외한다. 모델 수와 epoch 수는 데이터 규모가 달라 직접 시간으로 환산하지 않는다.
- 입력은 `F:/FaceX-Zoo/tinyface`와 `F:/FaceX-Zoo/QMUL-SurvFace`를 사용한다. 공식 MAT membership·원래 신원·전체 decode·SHA-256을 검증하고 `fine-tune/data/protocols/local_qmul_v1/`에 목록·audit를 기록한다. 원본 파일과 split은 수정하지 않으며 기존 C 경로 protocol은 보존한다.
- 각 대상의 train/evaluation 목록, 원래 identity, 클래스·이미지 수와 hash를 확인한다. remapped evaluation label을 train identity와 직접 비교하지 않는다. test는 FR 학습·checkpoint 선택에 사용하지 않는다.
- SurvFace의 신원 번호가 있는 평가 파일은 training identity와 겹치지 않음을 검사한다. `unmated_probe/distractors_*`는 파일에서 신원을 복원할 수 없으므로 공식 분할의 분리 조건에 의존한다고 기록하며 독립적인 신원 검증 완료로 표현하지 않는다.
- 현재 `C:\ronbun` SurvFace training manifest의 44,683장은 과거 FIQA calibration으로 분류됐지만 모두 공식 `training_set` 안에 있다. 논문 training set 전체를 FR fine-tuning에 쓰면 이 과거 calibration cohort를 새 모델의 독립 calibration으로 자동 재사용해서는 안 된다. ronbun 연계 시 calibration/test 분리 설계를 별도로 확인한다. 이번 작업은 C 저장소의 목록을 수정하지 않는다.
- 정식 학습과 smoke 실행은 명시적 설정으로 선택한다. 정식 입력이 없을 때 자동으로 dummy 실행으로 전환하지 않는다.
- smoke 산출물에는 별도 경로와 상태를 부여하고 정식 부모·최종 baseline·성능 보고 대상으로 사용하지 못하게 한다.

### 5.3 AdaFace 통합

- 정규화 전 backbone 특징에서 norm을 얻고, 정규화된 특징과 원래 norm을 AdaFace head에 전달한다.
- 먼저 L2 정규화한 특징에서 norm을 계산하여 품질 신호를 없애지 않는다.
- 공식 head의 logits, gradient, EMA 통계를 같은 초기 상태·입력으로 비교한다. eval 동작 등 의도적인 변경은 기록한다.
- MagFace의 norm-dependent margin과 regularization을 보존하고 관련 gradient를 확인한다.
- 공식 AdaFace의 라이선스·저작권 고지와 참조 commit을 보존한다.

## 6. 코드·노트북·브랜치 운영

- 모든 작업은 **`add-adaface` 브랜치**에서 한다. 기존 사용자 변경을 보존하고 main 등 다른 브랜치를 reset·덮어쓰지 않는다.
- 수정 전에 `git status --short`와 대상 diff를 확인한다. uncommitted 변경이 있으므로 branch 이름만으로 변경이 안전하게 보존됐다고 가정하지 않는다.
- 학습용 노트북과 실행 안내는 모두 **`fine-tune/` 아래**에 모은다. 루트 `goal.md`는 사용자가 지정한 목표 문서다.
- 기존 `fine-tune/notebooks/00`~`04` 흐름을 우선 유지하고, 같은 역할의 노트북·안내를 중복 생성하지 않는다.
- 재사용 학습·데이터·checkpoint 로직은 `fine-tune/modules/`에 두고, 노트북은 설정·모듈 호출·결과 확인으로 구성한다.
- 패키지 import 방식을 통일하고 새 커널에서 순서대로 실행 가능하게 한다.
- 첫 설정 셀에 학습 head, stage, seed, 실행 모드, 부모 checkpoint/hash, resume checkpoint, 출력 경로, 원본 결과 보존 옵션을 노출한다.
- 실제 학습기에서 사용한 optimizer·scheduler·epoch·RNG·AdaFace 통계를 저장한다. 마지막 셀에서 optimizer를 새로 만들어 재개 checkpoint로 저장하지 않는다.
- run ID로 stage·head·seed·과학적 설정·부모 출처를 구분한다. 기존 완료 checkpoint와 결과를 덮어쓰지 않는다.
- export 대상은 run과 checkpoint를 명시한다. 파일명 순서나 가장 높은 epoch를 자동 선택하여 서로 다른 실행을 섞지 않는다.
- 기준 코드는 기존 FaceX-Zoo를 재사용하고 필요한 통합부만 변경한다. `ronbun` 안에 학습기를 다시 복제하지 않는다.

## 7. 시스템 환경과 실행 제약

| 항목 | 기준 / 확인 상태 |
| --- | --- |
| 운영 환경 | Windows, PowerShell |
| CPU / RAM | 사용자 제공 기준: Intel Core i5-10600K / 64GB |
| GPU | 2026-10-09 확인: NVIDIA GeForce GTX 1080 Ti, 약 11GiB VRAM, compute capability 6.1 |
| Python | 2026-10-09 확인: Python 3.11.9, `py -3.11` |
| PyTorch | 2026-10-09 확인: `2.7.1+cu118` |
| CUDA runtime | 2026-10-09 확인: 11.8, `torch.cuda.is_available() == True` |
| 데이터·학습 출력 | F 드라이브 우선. 경로는 설정으로 변경 가능하게 유지 |
| 저장 공간 | 실행 전 다시 확인. 압축파일·추출 이미지·전처리본·임시파일·checkpoint 총량을 고려 |

- 별도 학습 환경을 구성하더라도 위 GPU 지원 조합을 우선 유지한다. 느슨한 requirements 범위만 보고 최신 Torch/CUDA를 자동 설치하지 않는다.
- 정식 학습은 CUDA를 실제 확인하고 장치 이름·runtime을 기록한다. CUDA 실패 시 조용히 CPU로 전환하지 않는다.
- 작은 forward/backward 시험 후 **1024D·실제 클래스 수·batch 64**에서 메모리와 처리량을 검증한다. batch 4·클래스 8의 통과를 전체 조건의 메모리 검증으로 대체하지 않는다.
- OOM 때문에 batch size, AMP, accumulation을 바꾸면 설정과 변경 이유를 기록한다. accumulation은 BatchNorm·AdaFace 통계까지 큰 batch와 동일하게 만들지 않는다.
- 단일 GPU에서 네 모델은 순차 실행을 기본으로 한다. 전체 epoch·반복 소요 시간은 실제 처리량으로 추정한다.
- 이미지 해제·decode·hash·정렬은 GPU 학습과 별도 병목이므로 단계별 시간을 기록한다.

### 실행 최적화 계약

- 노트북은 별도 Python 학습 프로세스를 시작한다. Windows spawn worker가 노트북 셀·CUDA 모델을 다시 실행하지 않도록 `__main__` 진입점을 둔다.
- `NUM_WORKERS`는 0/2/4를 실제 입력으로 비교하여 선택한다. `PREFETCH_FACTOR=2`, persistent worker, pinned memory와 비동기 전송을 사용한다. worker의 OpenCV/PyTorch 스레드는 1로 제한한다.
- 수백만 개 경로·label은 디스크 memmap으로 공유하여 worker별 Python 문자열 복제를 줄인다. 원본 이미지와 학습 목록은 바꾸지 않는다.
- sample별 seed/epoch/index로 augmentation을 결정하여 worker 수에 따라 입력 난수가 달라지지 않도록 한다. 이 난수 규칙은 checkpoint config에 기록하고 이전 규칙의 부분 학습을 조용히 이어받지 않는다.
- batch 64, Adam, lr, epoch, head margin/scale, 1024D/dropout을 속도 목적으로 바꾸지 않는다. AMP, accumulation, subset, 조기 종료를 자동 적용하지 않는다.
- 약 10분 간격과 안전한 중단 요청 시 완료된 optimizer step까지의 모델·optimizer·scheduler·RNG·batch 위치를 recovery checkpoint에 저장한다. progress.json과 console.log에 epoch/batch/처리량/ETA를 남긴다.
- 2026-10-11 실제 Trainer 입력 구간 측정에 따라 `NUM_WORKERS=4`, `PREFETCH_FACTOR=4`, `LOSS_SYNC_BATCHES=20`을 권장한다. loss의 기록용 CPU 전송만 묶으며 optimizer 갱신은 매 batch 수행한다. 로그·recovery·epoch 종료 전에 완료된 batch의 loss를 원래 순서로 반영한다. 공식 학습 조건·데이터·입력 RNG는 유지하고 기존 recovery 재개 호환성을 검증한다. 짧은 반복 입력의 cache 속도를 전체 epoch 속도로 단정하지 않는다.
- 노트북 커널 강제 종료는 학습 자식 프로세스를 안전 중단시키는 명령이 아니다. 먼저 실행 셀을 interrupt하고 stop 요청·recovery 저장을 확인한다. lock이 남으면 기록된 PID의 종료 여부를 확인한 후 수동 처리한다.
- 새 학습 namespace `shared_parent_v2`를 사용하여 과거 VGGFace2 이중 학습·부분 결과를 보존한다. 처리량 측정은 accuracy·논문 수치 재현 증거가 아니다.

## 8. Export와 ronbun 연결

모델별 export 묶음에는 다음을 포함한다.

- backbone checkpoint와 SHA-256, 모델 alias/UID, architecture와 1024D 출력 차원.
- 학습 head와 hyperparameter, stage, 실제 완료 epoch, seed, 부모 checkpoint/hash.
- 코드 commit 및 dirty 상태/변경 기록, 설정 snapshot/hash, 데이터·분할 식별 정보.
- 입력 크기, 정렬/resize 방법, RGB/BGR, 픽셀 정규화, raw feature와 L2 정규화 embedding의 정의.
- 완료/부분/smoke 구분, 실제 검증 항목·결과, 검증하지 않은 항목.

FaceX-Zoo 내부에서 같은 클래스를 두 번 생성하여 비교한 것은 export round-trip 검증으로만 보고한다. **실제 `ronbun` loader·전처리 경로를 호출하여 동일 실제 이미지의 출력과 비교**해야 통합 검증으로 인정한다.

1024D 경로의 압축·입력 검증을 확인하고 기존 512D baseline을 보존한다. 새 checkpoint에는 별도 alias와 압축·calibration 산출물을 부여한다. 이전 모델의 codebook·threshold를 자동 재사용하지 않는다.

## 9. Continuous FIQA 및 결과 해석의 경계

### 후속 평가의 공식 프로토콜 유지 원칙

이 항목은 `C:\ronbun`에서 수행할 후속 평가의 연구 원칙을 기록한다. 현재 FaceX-Zoo의 FR 학습 단계에 새로운 평가 기능이나 실행 의무를 추가하는 것은 아니다.

- 우선 UCFace의 데이터셋별 실험 방식을 기준으로 삼고, threshold 및 Continuous FIQA 실험은 각 데이터셋이 제공하는 공식 프로토콜을 유지한다. 모든 데이터셋을 일괄적으로 open-set identification으로 변환하지 않는다.
- SurvFace처럼 공식 open-set 프로토콜이 제공되는 데이터셋은 해당 1:N open-set identification의 gallery/probe 구성과 평가 규칙에 따라 Continuous FIQA를 평가한다.
- LFW는 UCFace와 같이 공식 1:1 verification 프로토콜을 사용한다. 공식 pairs/folds와 threshold 결정·평가 절차를 유지하며 임의의 1:N open-set 평가로 대체하지 않는다.
- 그 외 데이터셋도 제공되는 공식 프로토콜과 지표를 따른다. TinyFace는 공식 closed-set ranking을 유지하며 FIQA threshold 실험을 위해 임의의 unknown probe를 추가하지 않는다. 공식 프로토콜에서 정의되지 않는 threshold 지표는 해당 데이터셋의 공식 결과로 보고하지 않는다.
- 보정 전후 비교는 같은 공식 평가 조건을 사용한다. FIQA fit·safety 및 threshold 선택은 공식 프로토콜이 허용하는 training/development fold나 별도의 독립 보정 데이터로 처리한다. 최종 test를 보정 학습에 재사용하거나 fit 데이터를 확보하기 위해 공식 test의 신원·이미지를 임의로 제거하지 않는다. 구체적인 보정 데이터 확보·분리 방식은 후속 평가 저장소에서 공식 규칙과 독립성을 확인하여 확정한다.

- 특징 norm의 평균·표준편차를 계산하는 것은 **norm 진단**이다. 이것만으로 Continuous FIQA 임계값 보정 실험을 수행했다고 기록하지 않는다.
- 고정 checkpoint 내에서 같은 cohort·분할을 사용하여 보정 전후를 비교한다. backbone 교체 효과와 FIQA 추가 효과를 구분한다.
- Origin과 PQ 모두 동일한 비교 원칙을 적용하고, 서로 다른 점수 공간에는 각각 보정한 threshold를 사용한다.
- LFW 1:1 verification, SurvFace 1:N open-set identification, TinyFace closed-set ranking 등 프로토콜을 구분한다. 서로 다른 프로토콜의 threshold·수치를 직접 전이하지 않는다.
- 고화질/저화질 데이터셋 간 차이를 이미지 품질만의 인과효과로 단정하지 않는다. identity, 규모, 도메인과 프로토콜 차이도 함께 명시한다.
- FR의 논문 training split 전체와 FIQA fit 데이터를 별도로 관리한다. 기존 FR training 이미지 및 과거 calibration을 새 모델의 독립 fit으로 자동 재사용하지 않는다.
- `fiqa_fit_capacity_20261011_v1.json`의 SurvFace native unknown identity별 fit/safety/test 분할은 과거 용량 검토 제안이며 적용되지 않았다. 위의 공식 프로토콜 유지 결정에 따라 이 파생 분할안을 현재 평가 계획으로 채택하지 않는다. 공식 test를 유지하면서 사용할 독립 calibration 데이터는 후속 평가 단계에서 별도로 확보·검증한다.
- TinyFace에는 공식 non-mated probe가 없으며 probe-only threshold는 rank를 바꾸지 않는다. 공식 Rank-1 결과를 Continuous FIQA FPIR 효용의 증거로 사용하지 않는다.
- Rank-1, TPIR, FPIR 등의 지표를 임의 값이나 mock 값으로 채우지 않는다. 미실행 항목은 `null` 또는 `미실행`으로 표시한다.
- 경험적 FPIR 목표 충족이나 held-out 보정을 수학적 보장으로 표현하지 않는다.

## 10. 결과 보존과 요약

- `KEEP_RAW_RESULTS` 같은 원본 결과 보존 옵션을 첫 설정 셀과 모듈 실행 경로에 연결한다. 초기 목표 기본값은 보존(`True`)으로 두고 manifest에 기록한다.
- 보존 대상은 실험이 생성한 상세 결과·중간 산출물이다. 입력 데이터셋과 부모 checkpoint를 삭제 대상으로 포함하지 않는다.
- optimizer를 포함한 재개 checkpoint, 최종 export, 필수 출처 기록의 보존 정책을 상세 결과와 구분한다.
- 원본 미보존 시에도 실제 학습 이력의 집계, 완료 수, 오류·제외 수, 설정, hash, lineage와 해석 안내를 남긴다. 요약 검증 전 원본을 삭제하지 않는다.
- 원본 이미지·대형 checkpoint·상세 결과는 Git에서 제외한다. 검증된 작은 요약과 설정·실행 안내·출처 기록은 추적 가능하게 한다.
- 요약 파일 수·크기·예상 텍스트량을 고려하고, 전체 원본을 ZIP으로 묶은 것을 분석용 요약으로 대신하지 않는다.
- 파일 정리는 이번 run이 소유한 명시적 경로에 한정한다. 기존 데이터·완료 run을 소급 삭제하지 않는다.

## 11. 현재 검수 이력과 다음 작업

2026-10-09 이전 검수에서 head 중심 테스트 17개 통과, 공식 AdaFace와 한 batch의 학습 수치 일치, GTX 1080 Ti 소규모 세 head 업데이트를 확인했다. 이는 전체 학습·실제 얼굴인식 평가·ronbun 통합 완료의 증거가 아니다.

2026-10-10 작업은 공통 부모 1개·대상 공식 training split·Windows spawn 최적화·batch recovery를 구현한다. 기본 seed는 코드·노트북·설정에서 8972다. 제한된 자동 테스트와 처리량 진단은 전체 정식 학습과 구분한다. 최신 검증 수와 처리량 근거는 `fine-tune/reports/`의 해당 보고서를 따른다.

2026-10-11 로컬 TinyFace169,403개·SurvFace473,392개 JPEG의 전체 decode/hash와 공식 MAT 참조를 검증하고 F 드라이브의 두 공식 training split을 연결했다. SCFace는 미확보로 실행을 제한한다. 기본 후속 모델은 두 대상 × 세 head × 다섯 seed =30개다. 검증100 tests/새 커널 계획5개와 데이터 수량 충돌·FIQA 용량/독립성 검토는 `fine-tune/reports/local_qmul_integrity_and_fiqa_20261011_v1.md` 및 JSON을 따른다. 정식 모델 학습과 FIQA는 이번 데이터 연결 작업에서 실행하지 않았다.

2026-10-13 CosFace를 네 번째 후속 baseline으로 추가한다. 공통 ArcFace 부모 1개를 그대로 공유하며 후속 계획은 40개다. `head/AM_Softmax.py`를 변경 없이 재사용하고 저해상도 `margin=0.3`, `scale=64`를 적용한다. checkpoint의 공개 head 이름은 `cosface`, 구현 클래스는 `AM_Softmax`이며 margin도 출처·재개·export 검증에 포함한다. 실행 중인 02 및 00 노트북, 저장된 공통 설정, trainer·backbone·기존 head 구현과 학습 launch는 이번 작업에서 변경하지 않는다. 03/04 첫 셀에서 네 head와 CosFace 값을 명시해 과거 00의 세-head 목록보다 우선한다. 현재 02 커널은 재시작하지 않으며 부모 완료 후 03/04를 새 커널에서 시작한다. 이번 변경으로 02를 재학습할 필요는 없다. 아래 초기 준비 순서는 신규 캠페인에만 해당한다.

다음 실행 순서:

1. 커널 재시작 후 00 첫 셀의 변수·새 출력 root를 확인하고 저장한다.
2. 01에서 기존 VGGFace2 추출·목록을 확인한다. 이미 완료된 추출은 다시 실행하지 않는다.
3. 02에서 짧은 worker 처리량을 확인하거나 공통 부모 1개를 정식 학습한다.
4. 03에서 대상 프로필을 확인하고 공식 split 검증 후 네 head·5 seed를 학습한다.
5. 04에서 같은 행렬의 완료 checkpoint를 export한다.
6. ronbun의 1024D loader·전처리와 새로운 calibration 분리 계약을 검증한 뒤 Continuous FIQA 실험을 진행한다.

## 12. 완료 판정

- [ ] `add-adaface`에서 변경이 관리되고 기존 사용자 작업이 보존된다.
- [ ] 노트북·실행 안내가 `fine-tune/`에 모여 있고 새 커널 순차 실행이 가능하다.
- [ ] 기본 seed `8972` 및 반복 seed 정책이 설정·코드·문서·manifest에 일치한다.
- [ ] 데이터 출처·실제 이미지·전처리·분할·제외 목록을 검증했다.
- [ ] 네 head의 출력·gradient와 공식 AdaFace 통합을 검증했다.
- [ ] 정식 학습이 누락 입력·잘못된 부모·CUDA 실패를 숨기지 않는다.
- [ ] checkpoint 재개가 실제 optimizer·scheduler·RNG·EMA·epoch를 복구한다.
- [ ] 선언된 학습 행렬의 모델·seed별 완료 checkpoint와 출처가 존재한다.
- [ ] 실제 ronbun loader·전처리·1024D 경로에서 export를 검증했다.
- [ ] 요약과 안내가 실제 기록에 근거하며 미실행·부분·smoke를 구분한다.
- [ ] 원본 보존 옵션과 Git 추적 정책이 동작하고 기존 결과를 덮어쓰지 않는다.

**코드 구현 완료, 테스트 통과, 정식 학습 완료, FIQA 실험 완료는 각각 별도로 판정한다.** 이 문서의 체크리스트는 목표이며 실행 증거 없이 완료로 표시하지 않는다.
