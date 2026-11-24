# 새 VGGFace2 Parent에서 SurvFace MagFace까지

목표는 UCFace 논문 Table I의 MobileFaceNet·MagFace 행에서 비압축 cosine TPIR@20을 FPIR 10%에서 비교하는 것이다. 논문 수치 24.80%는 5회 평균이며 여기의 기본 실행은 seed 8972 한 번이다. [논문 Table I](https://openreview.net/pdf?id=k8BijWnOHZ)

| 순서 | 실행 내용 |
| --- | --- |
| [01 Parent](01_parent.ipynb) | 공식 5점 정렬 VGGFace2 8,631명·3,141,890장. 무작위 초기화한 MobileFaceNet 1024D + ArcFace. 24 epoch, batch 256, SGD momentum 0.9. 첫 epoch LR 0.01→0.1, 10·18·22 종료 시 ×0.1. m=0.5, s=64, 단조 fallback. Dropout 0, 좌우반전 0.5, backbone Conv/Linear 가중치 WD 1e-4, 그 외 0. 출력 BN bias는 0에 고정. |
| [02 Fine-tune](02_survface_magface.ipynb) | 같은 RUN_ID의 완료된 epoch 24 Parent에서 backbone만 로드하고 MagFace head를 새로 초기화한다. 전체 공식 학습 5,319명·220,888장, stretch 112×112, Adam, WD 0, batch 64, dropout 0.6, flip·affine. 1–3 epoch는 head 2e-4와 BN 1e-4, 4–40은 BN2d 고정, head 2e-4, 출력 BN weight와 backbone 1e-4. 10·20·30 종료 시 모든 LR ×0.1. 적응 margin의 norm은 detach하고 norm regularizer는 미분한다. 단조 fallback을 사용한다. |
| [03 최종 평가](03_origin.ipynb) | 같은 RUN_ID의 완료된 epoch 40을 고정한 후 SurvFace 공식 시험군에서 비압축 Origin ROC를 1회 계산한다. |

세 노트북은 같은 RUN_ID와 경로를 쓴다. 새 커널에서 기본값으로 실행하면 설정이나 저장된 결과를 조회한다. 실제 실행은 해당 노트북 첫 코드 셀에서 EXECUTE=True로 지정한다. 새 실험은 새 RUN_ID를 쓴다. 데이터 경로의 기본값은 F:/face_datasets, 모델은 F:/models이며, 결과는 모델 루트의 facex/clean_ucface/RUN_ID 아래에 쌓인다. VGGFace2 및 SurvFace의 기존 정렬 이미지·고정 학습 목록, 평가용 원본·역할 목록과 SDK의 PFLD 랜드마크 가중치를 사용한다. 기존 얼굴인식 모델 가중치를 찾는 단계는 없다. 학습 중 검증한 입력은 수정하지 않는다.

## Parent 점검표를 읽는 법

모든 학습 이미지를 사용한다. 현재 목록에서 마지막 미니배치는 Parent 2장, fine-tune 24장이다. 새 Parent·새 MagFace head의 학습 경로, 두 단계의 gradient 업데이트, 아래 세 패널을 GTX 1080 Ti에서 검증했다. 최종 단축 검증에서 입력 준비 18.7초, 준비된 패널의 추론·집계 8.6초였다. 파일 캐시와 장비 상태에 따라 달라진다. [GPU 검증 기록](/F:/models/facex/diagnostics/clean_carbon_gpu_20261124_v3.json)

Epoch 2·5·10·18·24에서 현재 backbone을 eval 모드로 평가한다. 매 지정 epoch의 지표와 분모, 평가 시간, 기준과의 %p 차이를 출력하고 parent/progress.json의 epochs에 저장한다. 최종 parent/summary.json에도 전체 이력이 남는다.

| 지표 | 우리 실행의 분모와 비교점 |
| --- | --- |
| LFW verification accuracy | deep-funneled 이미지를 bicubic 112로 변환. 공식 6,000쌍, 고정 10-fold. 각 fold의 cosine 임계값은 다른 9개 fold로 정한다. 10 epoch 99.0%, 최종 99.4–99.6%는 Carbon의 기대 범위다. 저자의 이 전처리에서 검증된 epoch별 합격선은 없다. |
| SCFace Rank-1 | 공식 눈 좌표로 2점 정렬. 로컬 고정 80명 gallery, d1·d2·d3 각각 400 probes. 각 거리와 전체 correct/probes를 확인한다. Table I의 저자 Parent는 d1 41.75%, d2 91.75%, d3 97.20%, 평균 77.08%. |
| TinyFace Rank-1 | SDK PFLD 5점으로 정렬. Gallery_Match 4,443장, probe 3,728장. distractor 제외, 최상위 이미지의 신원 비교. 저자 Parent는 55.58%. |

실행 초기에 평가 이미지를 한 번 준비하며 이 시간과 이후 backbone 추론·지표 시간을 구분한다. 모든 epoch는 같은 패널을 사용한다. 성능 추이를 살펴볼 수 있지만 저자와의 split·입력 픽셀 일치, VGGFace2 학습 인물과의 중복 여부, 최종 달성률은 미확인이다. 자동 모델 선택 없이 24 epoch Parent를 전달한다. 중간 점수를 보고 설정을 바꾸면 해당 벤치마크를 개발에 사용했다고 기록한다. SurvFace 공식 시험은 최종 모델 확정 뒤에만 사용한다.

## 결과와 중단 복구

채팅 검토에는 각 단계의 bundle.json에 적힌 순서로 summary.json, 이 안내, plan.json을 사용한다. bundle에는 파일별 크기와 SHA-256, 문자 수 기준 예상 토큰이 있다. Fine-tune의 head_direction_norm은 정규화한 클래스 가중치들의 평균 벡터 길이로, 1에 가까워질수록 분류기 방향이 한쪽에 집중된다는 뜻이다.

plan.json에는 실행 설정·입력과 코드 해시가 있다. progress.json은 마지막으로 게시한 epoch·checkpoint·SHA-256와 누적 이력이다. COMPLETED는 summary.json 해시이며 완료 시점에만 생긴다. 중단된 학습에서 같은 설정·RUN_ID로 RESUME=True와 EXECUTE=True를 지정하면 progress가 가리키는 검증된 epoch 경계에서 복구한다. 미완료 epoch는 다시 학습한다. 마지막 epoch까지 게시한 후 완료 처리 중 끊긴 경우에도 같은 방법을 사용한다. 새 프로세스가 저장한 오류는 last_failure.txt에서 확인한다.

KEEP_RAW_RESULTS=False가 기본이다. 마지막 재개용 checkpoint와 최종 가중치, 모든 epoch 요약은 보존한다. True이면 매 epoch checkpoint, Parent 점검 시점의 LFW·SCFace·TinyFace 단위 임베딩, 최종 평가의 질의별 점수·순위를 추가로 남긴다. 원본 이미지, 공식 목록, 이전 실행 결과는 정리 대상에 포함하지 않는다. 요약만으로는 새로운 임계값이나 개별 질의 분석에 필요한 점수를 복원할 수 없다.

origin/summary.json의 roc["0.10"]에서 tp/registered, fp/unknown, 실제 FPIR, TPIR@20, 순위·임계값 탈락 수를 확인한다. 3,000명 gallery의 신원별 평균 템플릿을 정규화한다. 미등록 최고 점수로 시험 ROC의 목표 FPIR 경계를 잡으며, 등록 성공은 정답의 Top-20 포함과 정답 점수의 임계값 통과를 모두 요구한다. 동점에서 오수용 예산을 초과하지 않으며 순위 경계 동점은 보수적으로 처리한다. gap_to_table_i_pp는 이 단일 run과 24.80%의 차이다. 5,319명 학습에는 기존 분리 연구에서 보정용으로 배정한 인물도 포함된다. 이 모델은 Table I 비교 전용이다.
