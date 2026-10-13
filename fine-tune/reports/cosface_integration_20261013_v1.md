# CosFace 후속 baseline 추가 및 사전학습 보존 검증

- 작성: 2026-10-13 KST. 저장소 `F:/FaceX-Zoo`, 브랜치 `add-adaface`.
- 기준 commit: `6e86b1caaf6c0fefe05df9178889b3a68e7d9263`와 이번 미커밋 변경.
- 상태: 코드 통합 및 제한된 CPU 검증 완료. 정식 CosFace 학습·GPU smoke·Continuous FIQA 평가 미실행.

## 변경과 논문 근거

`fine-tune/modules/model_builder.py`가 기존 `head/AM_Softmax.py`를 수정 없이 CosFace로 연결한다. UCFace 본문 IV-A-2, PDF 6쪽 / 인쇄 7181쪽을 직접 확인하여 저해상도 margin 0.3, scale 64를 `configs/cosface.yaml`, 03/04 첫 설정 셀과 정식 학습 guard에 반영했다. 입력·학습 데이터·backbone 구조·기존 세 head 계산은 바꾸지 않았다.

03/04의 `HEADS` 기본값은 ArcFace, AdaFace, MagFace, CosFace다. 과거 00에서 저장한 세-head 설정이 있어도 현재 노트북 변수가 우선한다. 공통 ArcFace 부모 1개를 공유해 TinyFace·SurvFace × 네 head × 다섯 seed(8972–8976) = 후속 모델 40개와 export 40개를 계획한다. SCFace는 비활성 상태를 유지한다. 공통 부모 공유는 기존 사용자 선택이며 저자 recipe 전체의 동일성 확인을 뜻하지 않는다.

run과 파일명·metadata의 head는 `cosface`, 저장된 모델 설정의 실제 클래스는 `AM_Softmax`다. `checkpoint_manager.py`가 `margin`을 기록·hash·비교하여 margin이 다른 재개/export를 거부한다. 04는 backbone export만 하며 별도 학습을 수행하지 않는다.

## 진행 중인 02 보존

00·02 노트북, 저장된 공통 설정, 현재 launch request, trainer, dataset loader, process runner, run_campaign, backbone, upstream AM_Softmax와 기존 YAML 네 개를 수정 전후 SHA-256으로 비교해 모두 동일했다. 기존 ArcFace/AdaFace/MagFace도 동일 seed·9-class CPU 모델의 초기 state_dict hash와 모델 설정 hash가 수정 전후 일치했다. 각 hash는 동명 JSON의 `preservation`에 기록한다.

현재 launch가 참조한 `pretrain_mobilefacenet_1024d_recovery_e29_b562_266f36f0.pt`를 CPU로 읽어 새 코드의 backbone·head strict load와 학습 설정 hash 일치를 확인했다. 현재 checkpoint를 덮어쓰거나 새 학습을 시작하지 않았다. 진행 중인 프로세스는 이미 import한 코드를 사용하며 hot reload를 실행하지 않았다. 마지막 관측(13:12:08 KST)은 PID 27156, epoch 36, step 10221/46637, 약 536.4장/초였다. 이는 당시 상태이며 이후 완료·안정성을 보장하지 않는다.

## 검증 결과와 한계

- CPU 재검증: 두 명령에서 38개·61개 통과, 환경 검사 1개가 중복되어 고유 테스트 98개 통과. 최종 테스트 자식 환경은 `CUDA_VISIBLE_DEVICES=-1`, OMP/MKL threads 1이었다.
- 최초 테스트는 37개 통과·환경 검사 1개 실패였다. 빈 CUDA visibility 설정에서 `Invalid device id`가 발생해 `-1`로 명시한 환경으로 재검증했다. 이를 해결하기 위한 학습 코드나 GPU 드라이버 변경은 하지 않았다.
- CosFace 목표 logit의 additive cosine margin, gradient 및 backbone 갱신, 1024D 추론 형상·정규화 검사 통과.
- 임시 소량 이미지: 기존 세-head 설정으로 만든 부모 → 네 head·두 seed의 독립 후속 학습 → export·완료 재사용 통과. CosFace margin 변경 export 거부, 중간 batch 복구, loss 동기화 시험 포함.
- 03/04 전체 코드 셀을 임시 과거 설정에서 계획 모드로 실행해 각각 40개 행 확인. 이 테스트에서는 실제 launch와 데이터 준비 호출을 차단했다.
- `git diff --check` 통과. 정식 데이터 학습·CosFace CUDA 처리량·모델 인식 성능·ronbun 통합·FIQA는 검증 범위가 아니다.

사용한 명령(저장소 루트):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
py -3.11 -B -m pytest fine-tune/tests/test_cosface_integration.py fine-tune/tests/test_notebook_workflow.py fine-tune/tests/test_optimized_workflow.py fine-tune/tests/test_deferred_loss.py fine-tune/tests/test_local_datasets.py -q -p no:cacheprovider
py -3.11 -B -m pytest fine-tune/tests/test_notebook_workflow.py::test_environment_never_claims_unchecked_integrity fine-tune/tests/test_audit_fixes.py fine-tune/tests/test_pipeline.py fine-tune/tests/test_heads_common.py fine-tune/tests/test_adaface_head.py -q -p no:cacheprovider
```

위 환경 변수는 CPU 검증용이다. GPU 정식 학습을 실행할 셸에는 적용하지 않는다.

## 다음 실행

현재 02는 중단·커널 재시작 없이 완료까지 유지한다. 부모 완료 후 03을 별도 새 커널에서 첫 설정 셀부터 실행하고 네 HEADS, 다섯 SEEDS, 두 FINETUNE_DATASETS, 부모·출력 경로와 40개 계획을 확인한다. 정식 실행은 EXECUTE=True, RUN_SMOKE_TEST=False다. 04는 같은 설정으로 실행한다. 이번 추가만을 위해 00/01/02를 다시 실행할 필요는 없다. 기존 세 head를 이미 완료했다면 03/04의 HEADS를 cosface 하나로 명시하여 추가 모델 10개만 생성할 수 있다.

운영 안내는 `fine-tune/README.md`, 목표와 제약은 `goal.md`에 반영했다. 기존 사용자 변경·미추적 보고서는 보존했다. 새로운 40개 계획은 과거 완료 artifact의 설정이나 기록을 소급 변경하지 않는다.
