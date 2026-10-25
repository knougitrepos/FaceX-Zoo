# ronbun 연구 분할·1024D 연결

2026-10-23 KST. upstream backbone/head와 원본 데이터를 보존하며 변경은 fine-tune 아래에 둔다.

FaceX-Zoo는 모델 생성 연구, ronbun은 생성된 모델을 활용하는 실험 연구로 별개 운용한다.
저장소·Git 이력·실행 모듈을 하나로 합치지 않는다. FaceX-Zoo는 ronbun 코드를 import하거나 실행하지 않는다.
export와 sidecar, 명시적 데이터 분할 기록만 전달한다. split_reference는 선택한 입력 CSV 경로이며
다른 위치에 복사한 파일로 설정할 수 있다. ronbun의 Python 실행 환경을 학습에 요구하지 않는다.

00 → 01 → 02 → 03 → 04 순서다. 00은 research_profiles(BASE_DIR)로 SurvFace/TinyFace/SCFace를
모두 선택하며 새 research_v2_boundary_<backbone> checkpoint/log/export namespace를 사용한다.
설정은 fine-tune/data/research_v2_boundary/notebook_settings_mobilefacenet.json에 저장한다.
Resnet을 선택할 때는5개 노트북의 SETTINGS_PATH를 같은 별도 파일로 바꾸고00의 BACKBONE을
`dict(type="Resnet", num_layers=50, mode="ir", feat_dim=1024, out_h=7, out_w=7, dropout_rate=.6)`로 설정한다.
부모 모델도 같은 backbone으로 생성한다. 1024D가 같다고 MobileFaceNet 부모를 Resnet에 로드하지 않는다.
Resnet batch16은1080Ti용 시작 설정이며 실제값을 source_config에 기록한다. MobileFaceNet은 batch64이다.
FaceX Resnet IR-50 구현을 사용하며 저자 모델의 모든 세부조건을 재현한다고 주장하지 않는다.

03은 research_protocol.py 생산 경로로 역할 목록과 학습 목록을 만들고 검증한 뒤
기존 full preprocessing·학습 경로에 전달한다. 기존 완료 학습 폴더에 덮어쓰지 않는다.
원본 QMUL manifest와 ronbun의 기존 SurvFace split-reference를 hash로 기록한다.
SCFace는 F:/FaceX-Zoo/SCface_database의 cropped frontal과 surveillance_cameras_all cam1–5만 사용한다.
거리별 중복 폴더, IR, rotation은 이 프로토콜에 포함하지 않는다.
50/80 인물은 seed8972 고정 연구 분할이며 UCFace 저자의 정확한 ID 목록 재현은 아니다.

| 데이터셋 | FR 학습 | 별도 역할 |
|---|---|---|
| SurvFace | 4255명176205장 | FR 내부gallery3000명·PQ1255명, FR 밖fit744명·safety320명 |
| TinyFace | 2570명7804장 | training에서 압축 학습, 공식rank 평가 |
| SCFace | 50명800장 | training에서 압축 학습, 다른80명gallery80/probe1200 |
| LFW/RFW | 사용 금지 | ronbun에서 supplementary pair calibration/evaluation |

80/20은 최적값이 아닌 사전 고정 후보다. Calibration gallery/등록자probe는 FR-seen이고
비등록자fit/safety만 downstream-unseen이다. 부모 인물 overlap과 분포 이동은 별도 한계다.

최초 전달 목록은 fine-tune/data/protocols/research_v1/handoff.json에 보존한다.
준비된 데이터셋을 추가할 때는 handoffs/<내용 SHA-256>.json에 새 불변 목록을 생성한다.
03/04의 protocol 출력과 저장 설정 campaign.handoff_path에 정확한 경로가 나타난다.
ronbun의 YAML handoff에 이 경로를 명시한다. 기존 목록/완료 실험은 변경하지 않는다.
각 데이터셋의 정확한 protocol.json 경로와SHA를 고정한다. protocol은 roles.csv hash, training identity hash,
학습 목록 hash와 실제 class 수를 포함하며 ronbun의 facex_handoff.py가 검증한다.
다른 분할/seed를 추가할 때는 새 namespace와 handoff 경로를 사용한다.
같은 데이터셋의 protocol을 기존 목록에서 교체하려 하면 중단한다.
export의 low-resolution resize 정책(bicubic/bilinear)은 ronbun이 파일에서 읽어 동일하게 적용한다.

00은 계획 설정만 저장한다. 03/04의 EXECUTE=False 기본값은 학습·export를 실행하지 않는다.
03/04는 HEADS와 HEAD_PARAMETERS를 저장 설정에서 읽는다. 새 캠페인의 완료 부모와 후속 모델이
없으면 이전 단계가 필요하다. 과거 CosFace 추가 시점의 자동 head 덮어쓰기 안내는 적용하지 않는다.
00의 DATA_INTEGRITY_MODE 기본값은 content_verified_reuse다. 최초에는 원본/전처리 이미지의
전수 SHA-256 및 decode를 수행한다. 이후에도 매번 현재 바이트의 SHA-256을 전수 대조하고,
manifest·policy·검증 코드·OpenCV/NumPy 버전이 같을 때만 decode 결과를 재사용한다.
크기/mtime만 보고 신뢰하지 않으며 hash I/O 자체를 제거하지 않는다. full은 매번 decode까지 수행한다.
receipt는 각 전처리 report 옆 verification_cache/에 저장한다. 실패·부분 검증은 receipt를 게시하지 않는다.
캐시 삭제 시 다음 실행에서 전수 검사한다. 변조 receipt나 이미지 변경은 중단한다. 의도적 데이터 교체는
새 전처리 artifact를 생성한다. 검증 후 학습/export 동안에도 데이터 파일을 변경하지 않아야 한다.
실행 설정/checkpoint/export metadata에 실제 hash/decode 수, 재사용 여부와 receipt hash를 남긴다.

새 ArcFace parent와 모든 ArcFace/MagFace 후속 run에는 target_only_fp32_zero_radicand_eps_v1을
처음부터 적용한다. FP32 feature/weight에서 cosine을 계산하고 정답에만 margin을 계산한다.
1-c*c가 정확히0이면 FP32 epsilon으로 바꾼 후 sqrt를 계산한다. 일반 내부값과 MagFace adaptive-margin,
norm 정규화 경로는 유지한다. 비유한 loss/gradient 중단 검사는 유지한다. CosFace/AdaFace 수식은 그대로다.
이 정책은 로컬 수치 안정화이며 UCFace/FaceX-Zoo 저자 공식 수정 또는 동일 성능 보장이 아니다.
기존 정책 parent/resume checkpoint는 새로운 정책 학습에 혼용하지 않는다. 기존 완료 artifact는 보존한다.

연구 분할은 현행 고정 후보를 먼저 평가한다. 다음은 ronbun에서 명시적 완료 기준 report를 고정하여
같은 checkpoint·codec의 fit/safety 표본 수를 분석하는 단계다. 최종 safety/test는 선택에 쓰지 않는다.
FR/calibration 배분 변경은 필요성 검토 이후 별도 split_reference, protocol namespace와 후속 모델 재학습
계획으로 진행한다. 부모 조건이 같다면 고정 부모를 재사용할 수 있다. 현재 코드가 자동으로 비율을
탐색하거나 기존 역할을 변경하지 않는다.
학습 이후04 export와 sidecar를 ronbun의 research.embeddings.facex_zoo로 등록하고
ronbun YAML의 models에 정확한 spec을 지정한다. 4heads×2backbones의 producer/consumer 테스트는
합성 가중치의 bounded 검증이며 인식 성능 또는 정식 학습 완료를 뜻하지 않는다.

KEEP_RAW_RESULTS는 입력 원본·부모checkpoint를 삭제하지 않는다. 기존 학습 보존 정책과
fine-tune/README.md의 실행·재개 안내도 확인한다. Ronbun 결과 해석은
C:/ronbun/notebooks/WORKFLOW_GUIDE.md를 따른다.
