# VGGFace2 Parent부터 4개 헤드의 Origin 평가까지

같은 `RUN_ID`로 네 노트북을 순서대로 사용한다. 첫 코드 셀에서 데이터와 모델 루트를 확인한 뒤 해당 단계의 `EXECUTE=True`를 지정한다. 새 실행은 새 `RUN_ID`를 사용한다. 결과는 모델 루트의 `facex/clean_ucface/RUN_ID`에 저장한다.

| 순서 | 실행과 결과 |
| --- | --- |
| [01 Parent](01_parent.ipynb) | 새 Parent를 학습한다. Epoch 2·5·10·18·24의 LFW, SCFace, TinyFace를 저장한다. Epoch 24에서 새 Parent의 SurvFace 공식 시험 ROC를 계산한다. 같은 평가 입력으로 해시가 고정된 Parent_WD0를 평가하고 논문 Table XIII와 함께 표에 표시한다. 완료된 Parent의 평가만 실행할 때 `VALIDATE_PARENT_ONLY=True`를 사용한다. |
| [02 CosFace 탐침](02_cosface_probe.ipynb) | `PARENT_SOURCE`로 새 Parent 또는 `legacy_parent_wd0`를 선택한다. 기본 순서는 TinyFace와 SurvFace다. 각 모델의 epoch 40 Origin 결과를 다음 데이터셋 시작 전에 표시한다. TinyFace 탐침은 distractor를 제외한다. 같은 `RUN_ID`에서 양쪽을 실행하면 결과 셀에 같은 프로토콜·head를 적용한 두 조건이 표시된다. 확인 속도가 중요하면 `PROBE_DATASETS=["tinyface"]`로 실행할 수 있다. |
| [03 전체 행렬](03_full_baselines.ipynb) | 새 Parent의 CosFace 탐침을 검토한 뒤 `APPROVE_EXPANSION=True`를 지정한다. 기본 검토 대상은 TinyFace와 SurvFace이며 실제 검토한 목록을 `REVIEWED_PROBE_DATASETS`에 넣는다. 기본 순서는 CosFace → AdaFace → MagFace → ArcFace다. `EXECUTE=False`에서 선택한 조합의 완료 여부와 마지막 저장 epoch, loss, head 방향 집중도를 확인한다. |
| [04 Origin과 export](04_origin_and_export.ipynb) | 완료된 조합만 평가하고 미완료 조합을 표에 표시한다. `EXPORT_BACKBONE=True`이면 backbone-only 모델과 해시가 있는 메타데이터, 정렬 정보, `inventory.json`을 생성한다. 저장된 종합표는 완료 폴더를 `REPORT_DIR`에 지정해 조회한다. |

## 데이터와 설정

| 대상 | 학습 입력 | 학습 설정 |
| --- | --- | --- |
| VGGFace2 | 공식 학습 8,631명·3,141,890장, 5점 정렬 112×112 | MobileFaceNet 1024D, ArcFace m=0.5·s=64 및 단조 fallback, SGD momentum 0.9, batch 256, 24 epoch. 첫 epoch LR 0.01→0.1, 10·18·22 종료 후 ×0.1. Dropout 0, 좌우반전 0.5, Conv·Linear 가중치 WD 1e-4. |
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

- Parent의 LFW는 공식 6,000쌍 10-fold이며 각 fold 임계값은 다른 9개 fold에서 계산한다. SCFace는 d1·d2·d3 각각 400개 probe, TinyFace 중간 점검은 distractor 제외 3,728개 probe다. Epoch 10 LFW 99.0%, 최종 99.4~99.6%는 Carbon의 기대 범위다. Parent 종합표에는 논문이 보고한 지점과 로컬 대조 값을 붙인다.
- 최종 TinyFace는 같은 3,728개 probe에서 4,443개 정답 gallery 기준 Rank-1을 보고한다. distractor 포함 열은 153,428장까지 추가한다. SCFace는 거리별 및 전체 Rank-1을 건수와 함께 보고한다. 동점은 보수적으로 순위를 계산한다.
- 최종 SurvFace는 3,000명 gallery 평균 템플릿에서 등록 60,423건·미등록 121,736건을 평가한다. ROC의 목표 FPIR은 20·10·5·1%다. TPIR@20은 정답 신원이 20위 안에 들고 정답 점수가 임계값을 통과한 비율이다. 실제 FPIR은 오수용 건수/미등록 분모다. `rank_failures`는 순위 탈락, `threshold_failures`는 순위 통과 후 점수 탈락, `unknown_cos_ge_095`는 미등록자의 최고 코사인이 0.95 이상인 건수다.
- 04의 Table I 열은 VGGFace2 MobileFaceNet 기본 모델의 5회 평균이다. 로컬 기본은 단일 seed이며 저자의 정확한 SCFace 학습 인물과 입력 픽셀 일치는 미확인이다. 비교표의 차이는 %p다. 공식 시험 점수에 따라 체크포인트나 학습 설정을 자동으로 고르지 않는다. [UCFace Table I](https://openreview.net/pdf?id=k8BijWnOHZ)

## 복구와 결과 파일

`plan.json`은 코드·데이터·Parent 해시, 실행 설정, 보존 정책을 기록한다. `progress.json`은 마지막 게시 epoch와 체크포인트 해시, 모든 epoch의 loss·LR·head 방향 집중도·출력 BN weight L2를 기록한다. 같은 설정과 `RUN_ID`에서 `RESUME=True`를 사용하면 이전 epoch의 모델·optimizer·Python·NumPy·CPU 및 CUDA RNG 상태를 복원한다. 중단된 epoch는 처음부터 다시 계산한다. 입력·Parent·실행 계획 해시 불일치는 복구를 막는다. `last_failure.txt`는 실패 스택을 남긴다.

`KEEP_RAW_RESULTS=False`에서도 마지막 복구 체크포인트, 최종 가중치, 집계표·재현 설정은 보존한다. True이면 이전 epoch 체크포인트, Parent 점검 임베딩, TinyFace·SCFace 임베딩과 SurvFace 질의별 점수도 보존한다. 집계만 보존한 경우 다른 임계값·개별 질의를 분석하려면 원본 평가의 재계산이 필요하다. `COMPLETED`는 해당 `summary.json`의 SHA-256이다. 채팅 업로드는 `bundle.json`의 순서와 파일별 크기·해시·예상 토큰을 참고한다. 종합 결과 `matrix_status=partial`에는 미완료 행이 포함된다.

`ronbun`의 `02_inputs.ipynb`에서 04의 정확한 `inventory.json`을 `MODEL_INVENTORY`에 지정한다. `paper_table1`은 `INVENTORY_PURPOSE="paper_baseline"`과 `INVENTORY_ORIGIN_ONLY=True`를 사용한다. `calibration_study`는 `"local_protocol_baseline"`을 사용하며 압축·보정 실행 시 `INVENTORY_ORIGIN_ONLY=False`를 선택한다. 가져오기 경로는 각 모델과 출처·범위를 대조한다.
