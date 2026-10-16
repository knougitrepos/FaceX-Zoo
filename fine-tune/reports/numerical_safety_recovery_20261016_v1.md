# ArcFace 수치 안정화 및 복구 준비 — 2026-10-16 KST

상태: 코드 수정·제한된 검증·잘못된 산출물 격리 완료. **정식 학습 재개는 실행하지 않음.**

## 확인한 문제

- 대상 run: `finetune_survface_arcface_seed8974`, MobileFaceNet 1024D, SurvFace 5,319 classes.
- 로그상 최초 비유한 loss: 2026-10-16 11:35:06, epoch 11 step 401. 실제 최초 이상값 배치는 직전 로그 step 381 이후 구간으로만 좁혀진다.
- epoch 10 checkpoint는 모델/optimizer 텐서가 모두 유한하며 설정 hash가 일치한다. epoch 11, epoch 36 및 중지 시 epoch 37 recovery에서는 backbone/head/optimizer에 NaN/Inf가 확인됐다.
- 원래 ArcFace의 `sqrt(1-cos**2)`는 cosine이 정확히 ±1이면 유한 loss에서도 NaN gradient를 생성했다. 기존 head 코드로 CPU에서 재현했다. 실제 실패 배치의 cosine/gradient가 저장되지 않았으므로 이번 학습의 직접 원인을 이 경계값으로 확정하지 않는다.
- 기존 Trainer는 loss/gradient의 유한성 확인 없이 optimizer.step을 수행하여 오류 상태로 계속 학습했다.

## 변경과 적용 범위

- `head/ArcFace.py`: 제곱근 입력에 `torch.finfo(cos_theta.dtype).eps` 하한을 적용. float32에서는 약 1.19209e-7이며, 하한 미만 경계 구간의 계산/미분을 변경하는 명시적인 수치 안정화 정책이다. 하한 밖의 출력/gradient는 기존 계산과 동일하다. 구조·출력 차원·margin·scale은 유지한다.
- `fine-tune/modules/trainer.py`: loss는 backward 전, gradient는 optimizer.step 전에 매 배치 검사. gradient inf-norm을 사용하고 clipping은 하지 않는다. NaN/Inf를 발견하면 업데이트하지 않고 실패 배치를 기록한 뒤 예외로 종료한다. BatchNorm/EMA 변경 가능성 때문에 실패 상태를 recovery로 저장하지 않는다. 직전 저장 파일은 별도로 검증해서 재개한다.
- 새 저장 파일의 `extra_meta.training_numerics`에 정책 버전 `20261016_nonfinite_guard_v1`, `resume_source`에 재개 파일 경로/SHA-256 및 이전 수치 정책을 기록한다. 이전 checkpoint에 해당 정책이 없으면 null로 남긴다. scientific config hash에 정책을 소급 삽입하거나 기존 hash 검증을 해제하지 않는다.
- 03 노트북의 사용자 변경, 기존 공통 parent, 다른 run 및 데이터셋은 수정하지 않는다.
- 기존 `progress.json`의 Windows 접근 거부 재시도 문제는 이번 변경 범위에 포함하지 않았다.

## 격리 결과

보관 폴더: `F:/FaceX-Zoo/fine-tune/checkpoints/_quarantine/20261016_arcface_seed8974_nonfinite_v1/`

- epoch 11~36 checkpoint 26개, epoch 37 recovery 1개, 기존 progress.json 1개: **28개 파일, 2,250,176,822 bytes** 이동.
- `quarantine_manifest.json`에 각 원래 경로·이동 경로·크기·SHA-256·이동 상태를 기록하고 이동 전후 hash 일치를 확인했다.
- `5694f324bd6a41fd91e352298ceaadb8/console.log`는 원본을 남기고 보관 폴더에 복사했다.
- 원래 run 폴더에는 epoch 1~10 checkpoint 10개와 기존 설정을 보존했다. 기존 캠페인 보고서와 launch 로그는 역사 기록으로 그대로 둔다.
- 격리본은 정식 모델·재개·export에 사용하지 않는다. 삭제하거나 덮어쓰지 않았다.

## 검증

- `test_numerical_safety.py`: **15 passed**. ±1의 정답/비정답 class 경계, float32/float64, 내부 구간의 기존 출력/gradient 동일성, 두 번째 배치에 NaN/Inf loss/gradient 주입 시 추가 optimizer 업데이트·checkpoint 저장 차단, 실패 위치 및 저장 provenance 검증.
- `test_optimized_workflow.py`: **14 passed**. 기존 4개 head의 배치 중간 중지/재개 결과 일치 및 관련 입력/실행 계약 포함. 합계 **29개 고유 테스트 통과**. 첫 실행에서 새 테스트의 잘못된 helper 이름으로 발생한 5개 실패는 테스트 코드를 고친 후 해당 파일 15개 전체를 재실행하여 해결했다.
- PyTorch 2.7.1+cu118 / GTX 1080 Ti: 작은 GPU head 시험에서 cosine ±1의 4개 조합 forward/backward 유한성 및 NaN/Inf gradient 감지 통과. 생산 모델 optimizer 업데이트는 수행하지 않았다.
- 실제 epoch 10 checkpoint: 수정 후 모델 정의로 backbone/head/Adam/scheduler/RNG 및 scientific config 검증을 통과하여 **다음 epoch=11** 확인. CPU 임의 입력 2장 forward/backward에서 유한 loss/gradient 확인. 정식 학습·실제 오류 배치 재현·수정 후 전체 epoch 성능 측정은 미실행.
- 모든 기존 완료 모델의 성능/유한성을 이번 작업에서 재감사한 것은 아니다. 수정이 과거 완료 결과에 소급 적용되는 것도 아니다.

## 사용자 재개 절차

정상 checkpoint:
`F:/FaceX-Zoo/fine-tune/checkpoints/shared_parent_v3/finetune_survface_arcface_seed8974/finetune_mobilefacenet_1024d_arcface_epoch_10.pt`

SHA-256: `2641fcadd3d7f7899fe7e6e62501bf9a4082ade088f2b12fae8711f73658ad4b`

03 노트북의 설정을 아래와 같이 직접 변경한다. 이전 seed8973의 resume mapping을 현재 run으로 교체한다.

```python
RESUME_CHECKPOINTS = {
    "finetune_survface_arcface_seed8974": (
        "F:/FaceX-Zoo/fine-tune/checkpoints/shared_parent_v3/"
        "finetune_survface_arcface_seed8974/"
        "finetune_mobilefacenet_1024d_arcface_epoch_10.pt"
    )
}
REUSE_COMPLETED = True
EXECUTE = True
RUN_SMOKE_TEST = False
```

중지된 상태에서 03 커널을 재시작하고 설정 셀부터 순서대로 실행한다. 02를 재실행하지 않는다. 03은 새 자식 프로세스에서 수정 모듈을 읽으며, 첫 로그에서 `Training numerical policy`와 `Starting training from epoch 11`을 확인한다. 기존 완료 run은 검증 후 재사용한다. 같은 모델의 epoch 11부터 다시 계산하며, 향후 다른 원인의 비유한 loss/gradient도 검사로 중단되므로 재발 시 새 오류 기록을 먼저 확인한다.
