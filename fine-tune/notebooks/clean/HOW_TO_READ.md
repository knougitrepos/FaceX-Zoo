# VGGFace2 Parent부터 4개 헤드의 Origin 평가까지

네 노트북은 기본 `RUN_ID="clean_mfn1024_s8972_v3"`를 공유한다. 첫 설정 셀의 `EXECUTE`, `RESUME`, `KEEP_RAW_RESULTS`는 모두 False다. 정식 실행 시 데이터와 모델 루트를 확인하고 해당 단계에서 `EXECUTE=True`로 바꾼다. 결과는 모델 루트의 `facex/clean_ucface/RUN_ID`에 저장한다. v2 중단 기록은 별도 보존한다.

새 Parent 학습 전에 TinyFace를 빠르게 대조하려면 02에서 `PARENT_SOURCE="legacy_parent_wd0"`, `PROBE_DATASETS=["tinyface"]`, `EXECUTE=True`로 실행한다. 해시가 고정된 기존 Parent를 사용해 CosFace 40 epoch를 학습하고 distractor 제외 Rank-1을 기록한다. 이후 01을 실행하고 02에서 `PARENT_SOURCE="clean_parent"`로 같은 조건을 평가하면 두 결과가 한 표에 표시된다.

| 순서 | 실행과 결과 |
| --- | --- |
| [01 Parent](01_parent.ipynb) | 새 Parent를 학습한다. Epoch 2·5·10·18·24의 LFW, SCFace, TinyFace를 저장한다. Epoch 24에서 새 Parent의 SurvFace 공식 시험 ROC를 계산한다. 같은 평가 입력으로 해시가 고정된 Parent_WD0를 평가하고 논문 Table XIII와 함께 표에 표시한다. 완료된 Parent의 평가만 실행할 때 `VALIDATE_PARENT_ONLY=True`를 사용한다. |
| [02 CosFace 탐침](02_cosface_probe.ipynb) | `PARENT_SOURCE`로 새 Parent 또는 `legacy_parent_wd0`를 선택한다. 기본 순서는 TinyFace와 SurvFace다. 각 모델의 epoch 40 Origin 결과를 다음 데이터셋 시작 전에 표시한다. TinyFace 탐침은 distractor를 제외한다. 같은 `RUN_ID`에서 양쪽을 실행하면 결과 셀에 같은 프로토콜·head를 적용한 두 조건이 표시된다. 확인 속도가 중요하면 `PROBE_DATASETS=["tinyface"]`로 실행할 수 있다. |
| [03 전체 행렬](03_full_baselines.ipynb) | 새 Parent의 CosFace 탐침을 검토한 뒤 `APPROVE_EXPANSION=True`를 지정한다. 기본 검토 대상은 TinyFace와 SurvFace이며 실제 검토한 목록을 `REVIEWED_PROBE_DATASETS`에 넣는다. 기본 순서는 CosFace → AdaFace → MagFace → ArcFace다. `EXECUTE=False`에서 선택한 조합의 완료 여부와 마지막 저장 epoch, loss, head 방향 집중도를 확인한다. |
| [04 Origin과 export](04_origin_and_export.ipynb) | 완료된 조합만 평가하고 미완료 조합을 표에 표시한다. `EXPORT_BACKBONE=True`이면 backbone-only 모델과 해시가 있는 메타데이터, 정렬 정보, `inventory.json`을 생성한다. 저장된 종합표는 완료 폴더를 `REPORT_DIR`에 지정해 조회한다. |

## 데이터와 설정

| 대상 | 학습 입력 | 학습 설정 |
| --- | --- | --- |
| VGGFace2 | 공식 학습 8,631명·3,141,890장, 5점 정렬 112×112 | MobileFaceNet 1024D, ArcFace m=0.5·s=64 및 단조 fallback, SGD momentum 0.9, batch 128, 24 epoch. PNG 로더는 `num_workers=6`, `prefetch_factor=4`, `drop_last=True`다. 첫 두 epoch 전체 step에서 LR 0.0025→0.025 선형 warmup, 10·18·22 종료 후 ×0.1. 전체 gradient L2 norm clip 5.0. Dropout 0, 좌우반전 0.5. Conv2d·Linear·BN2d·PReLU weight에 `backbone_decay=1e-4`, 출력 BN1d·head·모든 bias에 `other_decay=0.0`. |
| TinyFace | 2,570명·7,804장, PFLD 5점 | 공통 fine-tune 설정 |
| SCFace | 50명·800장, 제공 좌표 두 눈 정렬 | 공통 fine-tune 설정 |
| SurvFace `paper_table1` | 공식 학습 5,319명·220,888장, 112×112 bicubic stretch | `origin_only=True`. Table I 비교 범위. 보정용 fit·safety 신원이 학습에 포함된다. |
| SurvFace `calibration_study` | shared_v1 학습 4,255명·176,205장, 112×112 bicubic stretch | 보정용 fit·safety 신원이 FR 학습에서 분리된다. 보정 gallery의 FR 노출은 프로토콜에 기록돼 있다. |

공통 fine-tune은 Adam, batch 64, 40 epoch, dropout 0.6, WD 0, 좌우반전·affine이다. 1~3 epoch는 head LR 2e-4와 BN LR 1e-4를 학습하고 Conv·Linear·PReLU를 동결한다. 4~40 epoch는 BN2d를 eval 모드로 고정하고 output BN1d weight와 backbone을 1e-4로 학습한다. Head LR은 2e-4다. 10·20·30 종료 후 모든 LR을 ×0.1 한다. 두 학습 단계의 출력 BN bias는 0으로 고정한다.

| Head | 파라미터 |
| --- | --- |
| CosFace | AM_Softmax m=0.3, s=64 |
| AdaFace | m=0.4, h=0.333, s=64, t_alpha=0.01, safe_norms detach |
| MagFace | l_a=10, u_a=110, l_margin=0.45, u_margin=0.8, lamda=20, s=64. 적응 margin의 norm을 detach하고 단조 fallback `m*sin(m)` 적용. norm regularizer는 미분한다. |
| ArcFace | m=0.6, s=64, 단조 fallback `m*sin(m)` |

MagFace와 ArcFace의 안정화 설정은 로컬 레시피이며 `plan.json`과 `summary.json`에 기록한다. TinyFace·SCFace는 두 범위에 동일한 입력을 사용한다. 범위별 완료 증거와 Export 메타데이터는 별도로 관리한다. 기존 준비 이미지·공식 분할·이전 실험 경로는 읽기 전용으로 소비한다.

같은 `RUN_ID`의 새 Parent에서 학습 설정·seed·입력·Parent 해시가 일치하면 TinyFace와 SCFace의 완료 모델을 두 범위에서 재사용한다. 02 TinyFace CosFace도 이 대상이다. 대상 범위에는 검증된 최종 체크포인트 사본과 완료 기록을 게시한다. `reused_from`은 원래 학습 범위·해시·보존 옵션을 기록하며 중간 체크포인트는 원래 범위에 남는다. 대상에 이미 미완료 기록이 있을 때는 그 기록을 보존하고 재사용 게시를 멈춘다.

## 지표 읽기

Parent의 각 epoch 로그와 `parent/progress.json`에는 clip **전** 최대 전체 L2 gradient norm(`grad_norm_clipped_max`), PReLU 가중치의 최대 절댓값(`max_abs_prelu`), BN1d·BN2d의 최대 running variance(`max_bn_running_var`), 출력 Linear의 최대 행 L2 norm(`max_linear_row_norm`), 정규화한 클래스 방향 평균의 길이(`head_direction_norm`)가 남는다. 자동 health gate는 epoch 2부터 loss가 이전 최저 loss보다 3.0 초과 상승하거나, 어느 epoch든 PReLU 최댓값이 10.0 또는 BN variance 최댓값이 100,000을 초과하면 작동한다. 해당 epoch 체크포인트와 progress를 저장한 뒤 RuntimeError로 중단한다. `health_gate_failure`와 `last_failure.txt`에서 이유를 확인할 수 있다. 중단 epoch에서는 zero-shot 측정을 생략한다. 조건을 바꾼 실험에는 새 RUN_ID를 사용한다.

Parent는 매 epoch에 `seed + epoch`로 VGGFace2 전체 목록을 다시 섞고, 그 순서에서 마지막 2장을 제외한다. BatchNorm과 SGD 갱신에는 epoch당 24,546개 배치 × 128장 = 3,141,888장이 들어간다. `plan.json`의 `recipe.images`는 입력 목록 전체 크기(3,141,890장)이고, `recipe.drop_last`는 학습 시 마지막 미완성 배치를 제외하는 설정이다.

`grad_norm_clipped_max`가 5.0을 넘으면 그 epoch에 clipping을 적용한 배치가 있었다는 뜻이다. 최대치만 저장하므로 해당 배치 번호를 특정하려면 배치별 값이 필요하다.

Fine-tune에서는 매 epoch 종료 시 loss가 유한한지, 클래스 방향 집중도(`head_direction_norm`)가 0.85를 초과하는지, 출력 BN weight L2가 1.0 미만인지 확인한다. 기본 설정에서 backbone을 푸는 시점은 epoch 4다. Epoch 5부터는 loss가 이전 모든 epoch의 최저 loss보다 5.0을 초과해 높아져도 중단한다. Gate에 걸리면 실패 epoch 체크포인트와 진단을 저장하고 직전 정상 체크포인트도 보존한다. 같은 실패 기록에서 `RESUME=True`로 실행하면 gate 사유와 함께 다시 중단한다.

02와 03에서 `EXECUTE=False`로 마지막 셀을 실행하면 마지막 저장 epoch와 `health_gate_failure`를 확인할 수 있다. `health_gate_failed`는 해당 사유로 중단된 상태다. 02는 새 Parent와 Parent_WD0 양쪽의 학습 상태와 평가 완료 여부를 표시한다. 완료된 평가의 Rank-1 또는 TPIR은 바로 아래 비교표에 표시된다.

LFW v3은 deepfunneled 250×250 원본의 중앙 144×144 ROI에서 PFLD로 5점을 추출하고 좌표를 원본으로 되돌린 뒤 원본 픽셀에서 ArcFace 112×112로 정렬한다. `plan.json`의 `monitor_inputs.lfw_alignment` 값은 `pfld_center144_five_point_arc112_v1`이다. 이전 전체 프레임 resize 점수와의 차이에는 전처리 효과도 포함된다.

- Parent의 LFW는 공식 6,000쌍 10-fold이며 각 fold 임계값은 다른 9개 fold에서 계산한다. SCFace는 d1·d2·d3 각각 400개 probe, TinyFace 중간 점검은 distractor 제외 3,728개 probe다. Epoch 10 LFW 99.0%, 최종 99.4~99.6%는 Carbon의 기대 범위다. Parent 종합표에는 논문이 보고한 지점과 로컬 대조 값을 붙인다.
- 최종 TinyFace는 같은 3,728개 probe에서 4,443개 정답 gallery 기준 Rank-1을 보고한다. distractor 포함 열은 153,428장까지 추가한다. SCFace는 거리별 및 전체 Rank-1을 건수와 함께 보고한다. 동점은 보수적으로 순위를 계산한다.
- 최종 SurvFace는 3,000명 gallery 평균 템플릿에서 등록 60,423건·미등록 121,736건을 평가한다. ROC의 목표 FPIR은 20·10·5·1%다. TPIR@20은 정답 신원이 20위 안에 들고 정답 점수가 임계값을 통과한 비율이다. 실제 FPIR은 오수용 건수/미등록 분모다. `rank_failures`는 순위 탈락, `threshold_failures`는 순위 통과 후 점수 탈락, `unknown_cos_ge_095`는 미등록자의 최고 코사인이 0.95 이상인 건수다.
- 04의 Table I 열은 VGGFace2 MobileFaceNet 기본 모델의 5회 평균이다. 로컬 기본은 단일 seed이며 저자의 정확한 SCFace 학습 인물과 입력 픽셀 일치는 미확인이다. 비교표의 차이는 %p다. 공식 시험 점수에 따라 체크포인트나 학습 설정을 자동으로 고르지 않는다. [UCFace Table I](https://openreview.net/pdf?id=k8BijWnOHZ)

## 복구와 결과 파일

`plan.json`은 코드·데이터·Parent 해시, 실행 설정, 보존 정책을 기록한다. `progress.json`은 마지막 게시 epoch와 체크포인트 해시, 모든 epoch의 loss·LR·head 방향 집중도·출력 BN weight L2를 기록한다. 같은 설정과 `RUN_ID`에서 `RESUME=True`를 사용하면 이전 epoch의 모델·optimizer·Python·NumPy·CPU 및 CUDA RNG 상태를 복원한다. 중단된 epoch는 처음부터 다시 계산한다. 입력·Parent·실행 계획 해시 불일치는 복구를 막는다. `last_failure.txt`는 실패 스택을 남긴다.

`KEEP_RAW_RESULTS=False`에서도 마지막 복구 체크포인트, 최종 가중치, 집계표·재현 설정은 보존한다. True이면 이전 epoch 체크포인트, Parent 점검 임베딩, TinyFace·SCFace 임베딩과 SurvFace 질의별 점수도 보존한다. 집계만 보존한 경우 다른 임계값·개별 질의를 분석하려면 원본 평가의 재계산이 필요하다. `COMPLETED`는 해당 `summary.json`의 SHA-256이다. 채팅 업로드는 `bundle.json`의 순서와 파일별 크기·해시·예상 토큰을 참고한다. 종합 결과 `matrix_status=partial`에는 미완료 행이 포함된다.

`ronbun`의 `02_inputs.ipynb`에서 04의 정확한 `inventory.json`을 `MODEL_INVENTORY`에 지정한다. `paper_table1`은 `INVENTORY_PURPOSE="paper_baseline"`과 `INVENTORY_ORIGIN_ONLY=True`를 사용한다. `calibration_study`는 `"local_protocol_baseline"`을 사용하며 압축·보정 실행 시 `INVENTORY_ORIGIN_ONLY=False`를 선택한다. 가져오기 경로는 각 모델과 출처·범위를 대조한다.
