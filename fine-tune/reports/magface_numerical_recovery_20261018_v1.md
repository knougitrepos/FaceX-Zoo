# MagFace 경계 안정화 및 재개 검증 — 2026-10-18

## 변경과 적용 범위

`head/MagFace.py`의 `sqrt(1 - cos_theta**2)` 입력에
`clamp_min(torch.finfo(cos_theta.dtype).eps)`를 적용했다. 기존 ArcFace와 같은
경계 처리다. backbone, embedding 차원, margin, norm 정규화 항, 데이터 분할,
전처리, optimizer 및 학습 일정은 변경하지 않았다.

이 보정은 제곱근 입력이 epsilon 미만인 구간의 계산과 미분을 바꾼다.
모든 구간·dtype에서 원본과 100% 같다는 뜻이나 UCFace 원본 구현과 완전히
동일하다는 뜻은 아니다. 경계에서 무한 미분이 생겨 비정답 클래스의 마스킹을
거쳐 NaN gradient가 발생하는 문제를 막는 수치 보정으로 기록한다.

새 checkpoint에는 `training_numerics.version=20261018_magface_sqrt_floor_v2`와
`magface_sqrt_floor`가 기록된다. 재개 원본의 hash와 이전 수치 정책도
`resume_source`에 유지한다. 기존 checkpoint를 소급 변경하지 않는다.

## 검증 결과

- `test_numerical_safety.py`, `test_optimized_workflow.py`: **39 passed**.
- MagFace의 float32/float64, cosine ±1, 정답/비정답 경계에서 유한한 forward와
  backward를 확인했다. 테스트한 내부 값에서는 원본 logits, regularizer,
  gradient가 정확히 같았다.
- 실제 GTX 1080 Ti / PyTorch 2.7.1+cu118에서 아래 epoch 20 checkpoint의
  모델·optimizer·scheduler·RNG를 엄격한 설정 검사와 함께 복원했다.
- epoch 21의 300 batch를 메모리에서만 재실행했다. 모든 검사 대상 loss와
  전체 모델 gradient가 유한했으며, 정식 checkpoint는 저장하지 않았다.
- batch 271에서 동일한 feature와 head 가중치를 원본 `main:head/MagFace.py`에
  입력하면 feature/head gradient가 비정상으로 재현된다. 수정본은 정상이다.
  sample 35, classifier class 312에서 cosine=1.0이었고 정답은 5215였다.
  이 batch의 원본/수정본 logits는 같고 loss도 15.614091873168945로 같았다.
- 원본 checkpoint, 해당 `progress.json`, 03 노트북의 SHA-256이 진단 전후
  동일함을 확인했다. `progress.json`의 기존 failed 상태는 의도적으로 보존했다.

기계 판독 결과: [JSON](magface_numerical_recovery_20261018_v1.json).
이는 300 batch의 제한된 검증이며 남은 전체 epoch 완료나 얼굴인식 성능 검증은 아니다.
기존 완료 모델은 변경하지 않았으며 이번 보정만으로 재학습이 필요하다고 판정하지 않는다.

## 사용자 재개 절차

03 노트북의 커널을 재시작하고 설정 셀부터 순서대로 실행한다.
02 사전학습은 다시 실행하지 않는다. 다음 설정을 유지한다.

```python
REUSE_COMPLETED = True
EXECUTE = True
RUN_SMOKE_TEST = False
RESUME_CHECKPOINTS = {
    "finetune_survface_magface_seed8975": (
        "F:/FaceX-Zoo/fine-tune/checkpoints/shared_parent_v3/"
        "finetune_survface_magface_seed8975/"
        "finetune_mobilefacenet_1024d_magface_epoch_20.pt"
    )
}
```

재개 원본 SHA-256:
`7992b8d9d2317b5e1f630fdcefae85090836e7e218cc495ba0d8e3bd1368d204`.
재개 로그의 `Next epoch: 21`을 확인한다. 진단의 300 batch는 저장하지 않았으므로
정식 실행은 epoch 21 처음부터 진행한다. 완료 run 재사용은 기존 설정·출처 검증을 따른다.

정식 실행은 이번 수정 작업에서 시작하지 않았다. 향후 별도 non-finite 오류가
발생하면 기존 중단 가드를 유지하고 새 오류를 확인한다.

## 진단 재현 명령

정식 campaign이 중지되고 lock이 없는 상태에서만 실행한다. 출력 JSON은
기존 파일을 덮어쓰지 않으므로 새 파일명을 지정해야 한다.

```powershell
py -3.11 -u fine-tune/scripts/diagnose_magface_recovery.py --request fine-tune/data/launches/f0d19318f27941b78a3fd6f047d6ae5f/request.json --checkpoint fine-tune/checkpoints/shared_parent_v3/finetune_survface_magface_seed8975/finetune_mobilefacenet_1024d_magface_epoch_20.pt --report fine-tune/reports/magface_replay_new.json
```

기본 진단 범위는 300 batch이며 `--steps` 상한은 1000이다. 정식 학습 상태
저장 경로를 차단하고 기존 파일 hash를 검사한다. 진단 결과를 모델 완료로 취급하지 않는다.
