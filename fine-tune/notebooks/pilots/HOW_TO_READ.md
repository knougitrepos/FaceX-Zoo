# SurvFace 입력 변환 개발 대조

기존 VGGFace2 Parent_WD0에서 시작해 원본 얼굴 crop을 가로·세로 각각 112로 늘리는 방법(stretch)과, 종횡비를 유지해 긴 변을 112로 맞춘 뒤 가운데에 놓고 BGR 128로 채우는 방법(aspect_pad)을 비교한다. 보간은 두 조건 모두 bicubic이다. 학습·gallery·probe에 같은 변환을 적용한다. 두 조건 모두 Carbon s7e의 학습률, 3 epoch BN·head 예열, 7 epoch BN2d 고정, 출력 BN bias 0, detached margin과 monotonic fallback을 사용한다. 마지막 epoch 10만 개발 평가한다.

노트북 첫 설정 셀에서 모델·데이터 루트, 출력 경로, 실행·원본 보존 여부를 지정한다. 기본 실행은 계획과 지정된 기존 결과 조회다. 실제 재실행은 별도 새 RUN_ID로 수행한다. 같은 디렉터리에 결과가 있으면 실행을 거부한다. 중단된 디렉터리는 진단용으로 보존하고 새 ID로 두 짧은 학습을 처음부터 실행한다.

1. **plan.json**에는 GPU와 라이브러리, Parent·코드·입력 코호트 해시, 판정 기준이 있다. shared_v1에서 FR 학습이 허용된 SurvFace 원본 training_set 인물만 사용한다. 사진이 16장 이상인 인물 중 seed 8972의 고정 해시 순서로 학습 256명×16장, 등록 개발 128명×8장(인당 gallery 2장, query 6장), 미등록 threshold fit 128명×8장, 개발 FPIR 측정 128명×8장을 배정한다. 네 인물 집합은 겹치지 않는다. 등록자의 gallery와 query만 인물을 공유한다. 두 조건의 초기 가중치, 증강 난수와 이미지 순서는 같다.
2. **summary.json**에서 FP 분모 1,024, TP 분모 768, threshold와 실제 FPIR을 확인한다. 각 모델은 fit 미등록자 최고 코사인의 상위 10% 오수용 예산으로 threshold를 정한다. 경계 동점은 예산을 넘기지 않도록 제외한다. 개발 미등록자가 그 threshold를 통과하면 FP다. 등록자의 정답이 Top-20에 들고 정답 점수까지 통과하면 TP다. rank_failure와 threshold_failure는 각각 순위 탈락과 순위 통과 후 점수 탈락이다. 저장 비율은 0~1, delta 및 신뢰구간은 %p다.
3. **판정**은 pad−stretch TPIR 차이의 등록 인물 단위 paired bootstrap 95% 구간 하한이 0을 넘고 pad의 실제 개발 FPIR이 10% 이하여야 통과다. 구간은 동일 gallery·fit·학습 seed에 조건부다. 양성 결과는 더 큰 독립 대조로 진행할 근거다. 최종 시험, 12종 모델 생성, 정식 inventory export는 별도 연구 결정이다. 128명 gallery, 256개 학습 클래스와 사진 수가 많은 인물로 제한한 결과를 논문의 3,000명 gallery·5회 평균 24.80%와 직접 비교할 수 없다. VGGFace2와 SurvFace 간 인물 중복은 미감사 상태다.
4. **보존** 기본값 False에서 개별 임베딩·파일별 결과·후보 가중치는 저장하지 않는다. True로 새로 실행하면 두 조건의 개발 임베딩·인물/역할 목록 NPZ와 최종 backbone PT를 해당 실행 폴더에 남긴다. 요약으로 가능한 분석은 표의 두 조건 비교이며 새 임계값 분석과 개별 오류 확인에는 NPZ 또는 재실행이 필요하다. 재개용 optimizer checkpoint는 만들지 않는다.
5. **완료와 업로드**는 COMPLETED가 가리키는 summary.json과 해당 plan 해시를 확인한 뒤 판단한다. 도중에 남은 단일 arm 결과는 부분 결과다. 채팅 분석용은 bundle.json에 지정한 summary, plan, 이 안내 순서로 읽는다. 원본 및 가중치는 외부 모델 루트에 보존한다.
