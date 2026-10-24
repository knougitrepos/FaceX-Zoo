# SCFace 준비와 연구 분할

현재 원본 위치는 F:/FaceX-Zoo/SCface_database이다. 가시광 cam1–5와 거리1–3의1950장,
cropped frontal130장이 이번 실험 후보이며 중복 거리 폴더·IR·회전 이미지를 추가하지 않는다.

00의 research_profiles(BASE_DIR)는SCFace를 활성화하고 seed8972로 train50/test80을 고정한다.
인물당 정면1장과 surveillance15장으로 FR800장, test gallery80/probe1200장을 구성한다.
이는 UCFace 수량에 맞춘 연구 분할이며 저자와 동일한50명 ID를 재현한다는 뜻은 아니다.
03의 EXECUTE=True에서 공통manifest를 생성·검증하고 full resize-only 전처리 이후 학습한다.
전체 준비에는 PREPROCESS_MAX_IMAGES=None이 필요하며 표본 감사만으로 학습 준비 완료를 표시하지 않는다.
04는 같은설정의 완료 checkpoint만 export한다. 부모모델은 선택한 backbone과 일치해야 한다.

공유 manifest와설정·보존·재개는 [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)를 따른다.
Ronbun은 handoff.json의 정확한 protocol hash와 export의 training identity hash를 대조한다.
SCFace rank 평가에threshold에 의한 순위 개선을 주장하지 않는다.800장의PQ 학습량은 별도검사한다.
