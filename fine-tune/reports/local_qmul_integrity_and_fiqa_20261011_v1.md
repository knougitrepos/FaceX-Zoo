# 로컬 QMUL 데이터 검증·후속 학습 연결·FIQA fit 용량 검토

작성: 2026-10-11T01:20:39.438452+09:00. 코드: `add-adaface`, HEAD `0d9be3f76a5afbd43e7775d22d0768e3702f54c7`, 미커밋 변경 포함.
실행 당시 핵심 코드 SHA-256과 원본/파생 입력 출처는 같은 이름의 JSON에 기록한다.

## 판정 및 변경 범위

TinyFace·SurvFace의 로컬 공식 폴더를 읽기 전용으로 검사하고, UCFace 보충자료 II-A/Table IX의 전체 training split을 후속 학습에 연결했다. SCFace는 미확보로 선택 시 오류를 내도록 제한했다. 원본 이미지/MAT와 C:/ronbun 코드는 수정하지 않았다. 정식 FR 학습, 실제 ronbun loader 통합, Continuous FIQA fit/평가는 미실행이다.

기본 행렬은 공통 VGGFace2 부모 1개 → 대상별 3 head × 5 seed = TinyFace 15개·SurvFace 15개 → export 30개다. MobileFaceNet 1024D와 현재 논문 대응 설정을 유지한다. 공통 부모 파일 공유와 미공개 초기화/사전학습 세부는 사용자 승인 구현 선택이며 저자 실행의 완전한 동일성을 의미하지 않는다. TinyFace와 SurvFace를 섞거나 순차적으로 같은 후속 모델에 학습하지 않는다.

## 실제 원본 검사 결과

| 데이터/역할 | 신원 수 | 이미지 파일 수 |
| --- | ---: | ---: |
| TinyFace training | 2,570 | 7,804 |
| TinyFace gallery match | 2,569 | 4,443 |
| TinyFace registered probe | 2,563 | 3,728 |
| TinyFace gallery distractor | 복원 불가 | 153,428 |
| SurvFace training | 5,319 | 220,888 |
| SurvFace gallery | 3,000 | 60,294 |
| SurvFace mated probe | 3,000 | 60,423 |
| SurvFace unmated probe | native 2,319 + anonymous 신원 불명 | 121,736 |
| SurvFace verification 폴더 | pair 평가 별도 | 10,051 |

TinyFace **169,403개**, SurvFace **473,392개** JPEG 파일 전부 read/decode 및 SHA-256 계산을 완료했다. SurvFace 수에는 identification/training 463,341개와 별도 verification 폴더 10,051개가 포함되며 서로 중복되는 원본 이미지가 있을 수 있어 unique 이미지 수로 해석하지 않는다. gallery/probe의 MAT 참조와 실제 파일 집합 일치, 중복 MAT 사본 의미 일치, native→official label 대응, 알려진 train/test 신원 분리, source snapshot 불변을 검사했다. 실제 후속 loader는 resize/정규화 계약을 따르며 원본 저해상도 crop을 변경해 저장하지 않는다.

SurvFace positive/negative pair는 각각 5,320개다. pair에서 참조하는 9,913개 이미지는 모두 존재하며 verification 폴더의 나머지 138개는 미참조 이미지로 보존/검사했다. 미참조 파일이 있다고 평가 pair를 추가하지 않는다.

수량/출처 충돌은 다음과 같다.

- TinyFace README/UCFace Table IX의 probe 신원 2,569와 실제 MAT의 2,563이 다르다. 3,728개 참조 파일은 전부 존재하며 gallery의 6개 신원에는 probe가 없다. 임의 probe를 생성하지 않는다.
- SurvFace 공식 홈페이지 training 220,890·test 242,617·전체 463,507과 로컬 training 220,888·test 242,453·전체 463,341이 다르다. training은 UCFace Table IX와 일치한다. 공식 archive checksum이 없어 버전 차이/표기 오류/원저자 archive와의 바이트 동일성을 확정하지 않는다.
- UCFace 표의 SurvFace gallery/probe 이미지 수 표기와 제공 MAT의 역할별 수량이 다르다. gallery MAT 60,294, mated probe MAT 60,423을 실제 역할 기준으로 보존한다.
- Anonymous distractor의 실제 신원 분리는 파일만으로 독립 검증할 수 없다. 공식 분할 가정에 의존한다.

따라서 **로컬 MAT·논문 training 수량·전체 이미지 decode 검증 완료**이며, 공식 저자 원본의 모든 바이트가 동일하다고 단정하지 않는다.

## 실행 연결과 보존

00 첫 설정 셀의 `FINETUNE_DATASETS = ["tinyface", "survface"]` 및 `default_profiles(BASE_DIR)`가 F 드라이브 원본을 선택한다. 기존 저장 JSON을 자동 덮어쓰지 않았으므로 커널 재시작 후 **00을 다시 저장**해야 한다. 01은 준비된 VGGFace2 목록을 재사용하고 02는 공통 부모, 03은 두 대상의 세 head·반복 seed, 04는 같은 행렬의 완료 모델을 처리한다. 03/04의 `AUDIT_WORKERS=4`는 CPU 데이터 검사 worker 수이며 FR 학습 worker와 별개다.

03/04 및 정식 자식 프로세스의 preflight는 로컬 검사를 확인한다. 최초 검사는 전체 decode이며 이후 cache는 이미지 경로/크기/mtime·MAT/출처 snapshot과 파생 파일 hash를 확인한다. 매 실행마다 모든 이미지 바이트를 재해시하지 않으므로 동일 크기·동일 mtime으로 악의적으로 바뀐 원본까지 검출하는 계약은 아니다. 소스가 정상적으로 바뀌면 기존 immutable protocol 재사용을 거부한다.

파생 manifest/list/이미지별 hash/local_audit/protocol은 `fine-tune/data/protocols/local_qmul_v1/<dataset>/`에 둔다. 이전 protocol과 사용자 변경을 보존한다. 두 원본 폴더와 대용량 검증 CSV는 Git에서 제외하고 이 작은 보고서/JSON은 추적할 수 있다. 입력 원본과 입력 감사 출처는 `KEEP_RAW_RESULTS` 삭제 대상이 아니다. 본 보고서 JSON은 상세 점수 전체를 압축한 파일이 아니라 입력 수량·hash·검증/미실행 상태·제안 용량의 집계다.

## Continuous FIQA fit 용량과 독립성

**FR training split 전체는 유지**한다. 현재 C 저장소의 과거 SurvFace calibration 44,683장은 이번 전체 FR training에 포함되므로 새 checkpoint의 독립 FIQA fit으로 자동 재사용하면 안 된다. 현재 C manifest의 일부 unknown label은 이미지별 pseudo identity라 같은 실제 신원이 fit/safety로 섞일 수 있다. 향후 FIQA 연결 시 새 manifest의 native `source_identity_id`를 기준으로 그룹 분리해야 하며 이번 작업에서는 C 코드/분할을 변경하지 않았다.

SurvFace non-mated 중 native 신원이 있는 2,319명/93,260장과 anonymous 28,476장이 있다. seed8972로 알려진 unknown 신원의 development 비율40%, 그 안의 safety 비율30%를 선언해 계산한 용량은 아래와 같다. **제안만 산출했으며 실험 분할·fit에 적용하지 않았다.**

| 제안 역할 | 신원 수 | 이미지 수 |
| --- | ---: | ---: |
| fit | 623 | 21,138 |
| safety | 306 | 14,113 |
| test: native unknown | 1,390 | 58,009 |
| test: anonymous | 복원 불가 | 28,476 |

Gallery 3,000명과 mated probe 60,423장은 유지 가능하다. 최종 unknown test는 86,485장으로 줄어든다. 모든 Global/Continuous × Origin/PQ 비교에서 같은 cohort를 사용하고 각각의 점수 공간에 별도 threshold를 fit해야 한다. 알려진 unknown만으로 fit한 함수의 anonymous test 혼합에 대한 분포 차이도 별도로 평가해야 한다.

현재 ronbun fit/safety 최소20개 요구에는 충분하지만, 실제 품질 구간 coverage·임베딩·점수·압축기·수치 안정성은 검증하지 않았다. 같은 신원의 여러 이미지는 상관되어 있으므로 이미지 수를 독립 표본 수로 간주하지 않는다. 저 FPIR의 tail 신뢰성이나 수학적 FPIR 보장을 주장할 수 없다.

이 분할안을 사용하면 **UCFace 공식 전체 평가와 다른 파생 FIQA protocol**이다. 공식 전체 gallery/probe를 완전히 유지해 FIQA test로 쓰려면 외부 독립 calibration을 확보해야 한다. 공식 test 전체를 fit과 test에 중복 사용하는 것은 허용하지 않는다. 단일 데이터셋 내 독립 fit 확보와 공식 전체 평가를 동시에 충족한다고 보고하지 않는다.

TinyFace 공식 protocol에는 non-mated probe가 없어 현 FPIR threshold calibration을 그대로 적용할 수 없다. probe-only threshold는 ranking을 바꾸지 않으므로 Rank-1 개선을 FIQA 효용으로 주장하지 않는다. TinyFace는 공식 FR ranking 검사, SurvFace는 FIQA 1:N 주 평가로 사용하거나 TinyFace unknown-probe 파생 protocol을 별도로 선언해야 한다.

## 검증 및 읽는 순서

- `py -3.11 -B -m pytest fine-tune/tests -q -p no:cacheprovider`: **100 passed**, 실패0. synthetic 데이터의 누락/overlap/decode/cache 변조·SCFace 차단·행렬·용량 제안 검사를 포함한다.
- 별도 새 커널에서 00–04를 `EXECUTE=False`, 임시 설정 JSON으로 순차 실행: **5개 성공**. 실제 학습/export 성공의 증거가 아니다.
- 로컬 두 원본 전체 audit와 공식 training manifest 등록: 완료. 실제 CUDA 전체 epoch/모델 정확도/ronbun loader/FIQA: 미실행.

먼저 본 문서의 판정·수량 충돌·FIQA 독립성 항목을 읽고, 같은 이름의 JSON에서 완료 상태와 출처 hash를 확인한다. 모델별 결과가 필요할 때는 이후 학습 campaign/export metadata를 추가한다. 이 요약만으로 개별 이미지 내용을 복원하거나 원본 없는 재학습을 수행할 수 없다.

## 근거

- 로컬 `paper/UCFace.pdf` IV-A, `paper/부록.pdf` II-A/Table IX와 저해상도 fine-tune 비교 표. 실제 파일 경로는 현재 저장소 paper 목록을 따른다.
- [SurvFace 공식 protocol](https://qmul-survface.github.io/protocols.html), [TinyFace 공식 페이지](https://qmul-tinyface.github.io/).
- 실제 원본 MAT/readme/평가 .m 및 `local_audit.json`, `protocol.json`; 파일 SHA-256은 동명 보고서 JSON 참조.
- C:/ronbun/research/datasets/manifests.py의 unknown label 생성과 research/calibration/continuous.py의 fit/safety 조건을 읽기 전용 확인.
