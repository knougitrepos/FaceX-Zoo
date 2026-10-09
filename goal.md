# FaceX-Zoo 연구 목표 및 작업 기준

- 작성일: 2026-10-09 KST
- 작업 저장소: `F:\FaceX-Zoo`
- 작업 브랜치: **`add-adaface`**
- 현재 상태: **구현 및 검증 중**. 전체 학습 완료나 UCFace 수치 재현을 의미하지 않는다.
- 문서 역할: 목표, 실험 기준, 시스템 제약, 작업 범위와 완료 조건을 정한다. 실제 구현·실행 상태는 코드, 설정, 테스트와 run manifest로 확인한다.
- 이번 문서 작성은 학습 실행이나 Codex의 활성 goal 등록이 아니다.

## 1. 최종 목표

FaceX-Zoo를 공통 학습 기반으로 사용하고 공식 AdaFace head를 통합하여, **MobileFaceNet 1024D + ArcFace / AdaFace / MagFace**의 학습 완료 baseline 3종을 생성한다. VGGFace2를 사용하며, UCFace 논문 본문에서 확인할 수 있는 baseline 학습 조건을 최대한 맞춘다.

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
| 후속 학습 classifier | ArcFace / AdaFace / MagFace | 학습된 backbone에 새 classifier를 연결하는 단계 구분 |
| Optimizer | Adam | IV-A-3 |
| MobileFaceNet 학습률 | `1e-3` | IV-A-3 |
| 학습률 감소 | 10 epoch마다 0.1배 | 40 epoch 실행에서 milestones `[10, 20, 30]`로 구성 |
| Baseline batch size | 64 | softmax classifier 학습 기준. 장비에서 실제 검증 필요 |
| 학습 epoch | 40 | 논문 대응 baseline 학습 설정. 사전학습의 전체 세부 일정까지 동일하다고 단정하지 않음 |
| Dropout 비율 | 0.6 | 비율은 본문 명시. 삽입 위치는 구현 선택으로 별도 기록 |
| ArcFace, 저해상도 설정 | `m=0.6`, `s=64` | IV-A-2 |
| AdaFace, 저해상도 설정 | `m=0.4`, `s=64`, `h=0.333` | IV-A-2 |
| MagFace | `(l_a,u_a,l_m,u_m,lambda_g)=(10,110,0.45,0.8,20)` | IV-A-2. 추가 설정의 출처는 공식 구현과 구분 |
| AdaFace EMA 계수 | `t_alpha=0.01` | 공식 구현 사용 예시에서 채택한 값. UCFace 본문에 명시된 값으로 표현하지 않음 |

고해상도·혼합 해상도 조건은 본문에서 각 원 논문의 hyperparameter를 따른다고 설명한다. 저해상도 설정을 그대로 사용해 LFW 등을 평가할 경우, 그것은 본 연구의 고정 checkpoint 교차 데이터셋 평가이며 논문의 모든 표와 동일한 조건이라는 뜻은 아니다.

`112×112`, BGR, `(pixel - 127.5) / 128`, weight decay, augmentation 범위, dropout 위치, 데이터 분할, 사전학습 loss 등은 **논문 명시 / 공식 코드 / 본 연구 선택**으로 각각 분류한다. 이미 선택한 값도 검증 없이 논문 설정이라고 승격하지 않는다.

## 4. Seed 정책: 기본값 8972

**논문에서 별도의 seed 숫자를 지정하지 않은 경우 기본 seed는 `8972`로 한다.** 현재 확보한 본문에는 구체적인 seed 숫자가 확인되지 않았다. `8972`는 사용자 지정 재현성 값이며 논문 저자의 seed로 표현하지 않는다.

- 설정의 기준값은 `fine-tune/configs/common.yaml`의 `project.seed: 8972`로 한다.
- Python `random`, NumPy, PyTorch CPU/CUDA, DataLoader generator 및 worker 초기화가 같은 실행 seed 정책을 따르도록 한다.
- seed 초기화는 모델 생성, 데이터 shuffle 및 augmentation에 앞서 수행한다.
- 노트북의 기본값, Python 함수 기본값, fallback 값, 실행 안내를 통일한다. `42`를 별도 기본값으로 남기지 않는다.
- 실제 seed, 분할 hash, 라이브러리 버전, deterministic 관련 옵션, GPU 정보를 manifest에 기록한다. seed만 같다고 환경 간 bitwise 동일성을 보장하지 않는다.
- 반복 실험 시 각 run의 seed를 고유하게 지정한다. 논문의 5회 반복에 대응하는 본 프로젝트의 기본 학습 seed 목록은 **`[8972, 8973, 8974, 8975, 8976]`**으로 사전 선언한다. 이 목록 역시 본 연구 선택이다.
- 하나의 seed로 세 head를 비교할 때 데이터 분할·전처리는 공유한다. 공통 부모를 공유하는지, run마다 부모를 다시 학습하는지는 학습 단계 설정에 명시한다.
- 논문 표마다 반복 횟수가 다를 수 있으므로 모든 실험에 5회를 일괄 적용하지 않는다. 학습 5회, 평가 fold, calibration split 반복을 서로 대체하거나 독립 표본처럼 합치지 않는다.

**작성 시점의 구현 차이:** 현재 확인한 공통 설정·노트북·`set_seed` 기본값은 아직 `42`다. 이 문서는 목표값을 `8972`로 정하며, 이번 문서 작성만으로 실행 코드가 바뀌지는 않는다. 다음 학습 전 관련 경로를 모두 수정·검증해야 한다. 기존 완료 결과는 seed 변경을 이유로 덮어쓰거나 재명명하지 않는다.

## 5. 학습 단계와 데이터 계약

### 5.1 데이터 준비

- 원본 압축파일·메타데이터의 출처와 무결성을 확인하고, 이미지 수·identity 수·손상·누락을 집계한다.
- Git clone, 압축파일 존재, 이미지 표본 읽기를 전체 데이터 검증 완료로 간주하지 않는다.
- 실제 압축 내부 경로와 loader의 `data_root`·인덱스 경로가 일치해야 한다.
- 세 모델은 같은 유효 이미지 목록, identity-label 매핑, 분할과 전처리를 사용한다.
- 누락·손상 이미지를 영벡터나 난수 이미지로 바꾸어 정식 학습하지 않는다. 검사 실패 시 중단하거나, 사전 정의된 제외 정책과 제외 목록을 기록한다.
- 단순 resize와 landmark 기반 얼굴 정렬을 구분한다. 정렬을 구현하지 않았으면 `정렬 완료`라고 기록하지 않는다.
- train/validation 역할을 명시한다. 같은 identity의 이미지 분할은 분류기 검증으로 사용할 수 있으나, 미관측 identity 평가라고 표현하지 않는다.
- 최종 평가 test를 FR 학습·checkpoint 선택·압축기 학습·FIQA calibration에 사용하지 않는다. VGGFace2와 평가 데이터의 identity overlap 확인 결과 또는 확인 한계를 남긴다.

### 5.2 사전학습 → fine-tuning

- 무작위 초기화에서 학습하는 단계와 부모 checkpoint를 불러와 추가 학습하는 단계를 분리한다.
- `pretrain`, `finetune`, `export`의 입력·출력·설정·완료 상태를 각각 기록한다.
- fine-tuning은 명시적으로 선택한 부모 checkpoint, 기대 SHA-256, 학습 완료 출처를 요구한다. hash 일치만으로 학습 완료를 판단하지 않는다.
- 사전학습 loss와 데이터, 후속 학습 데이터·분할, backbone 업데이트 여부, classifier 초기화 정책을 실행 전에 고정한다.
- 공통 ArcFace 부모에서 분기하거나 VGGFace2에서 두 단계 모두 학습하는 방식을 택할 수 있으나, 본문에서 확인되지 않은 부분은 본 연구 선택으로 기록한다. 임의로 SurvFace fine-tuning을 추가하지 않는다.
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
- 단일 GPU에서 세 모델은 순차 실행을 기본으로 한다. 전체 epoch·반복 소요 시간은 실제 처리량으로 추정한다.
- 이미지 해제·decode·hash·정렬은 GPU 학습과 별도 병목이므로 단계별 시간을 기록한다.

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

- 특징 norm의 평균·표준편차를 계산하는 것은 **norm 진단**이다. 이것만으로 Continuous FIQA 임계값 보정 실험을 수행했다고 기록하지 않는다.
- 고정 checkpoint 내에서 같은 cohort·분할을 사용하여 보정 전후를 비교한다. backbone 교체 효과와 FIQA 추가 효과를 구분한다.
- Origin과 PQ 모두 동일한 비교 원칙을 적용하고, 서로 다른 점수 공간에는 각각 보정한 threshold를 사용한다.
- LFW 1:1 verification, SurvFace 1:N open-set identification, TinyFace closed-set ranking 등 프로토콜을 구분한다. 서로 다른 프로토콜의 threshold·수치를 직접 전이하지 않는다.
- 고화질/저화질 데이터셋 간 차이를 이미지 품질만의 인과효과로 단정하지 않는다. identity, 규모, 도메인과 프로토콜 차이도 함께 명시한다.
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

문서 작성 시점에 이전 검수 이후 import, smoke 구분, seed 초기화, 평가 미실행 표기 등에 대한 후속 코드 변경이 보인다. **후속 수정 전체를 다시 검수한 상태는 아니므로**, 과거 결함을 현재에도 모두 존재한다고 단정하거나 반대로 모두 해결됐다고 기록하지 않는다.

우선 작업 순서:

1. 공통 설정·함수·노트북 기본 seed를 `8972`로 맞추고 실제 초기화·기록 경로를 검증한다.
2. 새 커널의 노트북 import와 실행 순서를 검증한다.
3. 실제 이미지 준비·경로·전처리·누락 처리·분할을 검증한다.
4. 정식/smoke 모드 격리, 부모 완료 상태·기대 hash 확인을 검증한다.
5. 실제 학습 상태 저장·재개와 모든 checkpoint의 부모·설정·seed 기록을 검증한다.
6. 미측정 수치·고정된 가짜 이력·잘못된 완료 문구가 요약에 남지 않도록 확인한다.
7. 실제 규모 메모리 시험 후 정식 사전학습·후속 학습을 수행한다.
8. 명시적으로 선택한 완료 checkpoint를 export하고 실제 ronbun 경로를 검증한다.

## 12. 완료 판정

- [ ] `add-adaface`에서 변경이 관리되고 기존 사용자 작업이 보존된다.
- [ ] 노트북·실행 안내가 `fine-tune/`에 모여 있고 새 커널 순차 실행이 가능하다.
- [ ] 기본 seed `8972` 및 반복 seed 정책이 설정·코드·문서·manifest에 일치한다.
- [ ] 데이터 출처·실제 이미지·전처리·분할·제외 목록을 검증했다.
- [ ] 세 head의 출력·gradient와 공식 AdaFace 통합을 검증했다.
- [ ] 정식 학습이 누락 입력·잘못된 부모·CUDA 실패를 숨기지 않는다.
- [ ] checkpoint 재개가 실제 optimizer·scheduler·RNG·EMA·epoch를 복구한다.
- [ ] 선언된 학습 행렬의 모델·seed별 완료 checkpoint와 출처가 존재한다.
- [ ] 실제 ronbun loader·전처리·1024D 경로에서 export를 검증했다.
- [ ] 요약과 안내가 실제 기록에 근거하며 미실행·부분·smoke를 구분한다.
- [ ] 원본 보존 옵션과 Git 추적 정책이 동작하고 기존 결과를 덮어쓰지 않는다.

**코드 구현 완료, 테스트 통과, 정식 학습 완료, FIQA 실험 완료는 각각 별도로 판정한다.** 이 문서의 체크리스트는 목표이며 실행 증거 없이 완료로 표시하지 않는다.
