# program.md — TTNS (Test-Time Nuisance Subspace) 자율 실험

## 연구 목표

Paired data 없이, test batch 통계만으로 nuisance subspace를 추정하여 FM-AD의 distribution shift 문제를 해결한다.
NSP(paired oracle)의 80%+ 성능을 unpaired test-time method로 달성하는 것이 목표.

**목표**: MVTec AD 2 I-AUROC를 최대화 (unpaired, test-time only)
**지표**: `ad2_auroc` — higher is better (secondary: `ad1_auroc` ≥ 95%)
**현재 베이스라인**:
- No projection baseline: AD2 59.9%, AD1 96.5% (Mahalanobis, L8)
- NSP K=100 (paired oracle): AD2 83.8%, AD1 96.3%
- GO 기준: AD2 ≥ 67% (NSP의 80%)

## 파일 스코프

### 수정 가능

- `scripts/ttns_experiment.py` — TTNS 핵심 코드 (nuisance estimation, projection, scoring)

### 읽기 전용

- `program.md`
- `scripts/phase3_nsp_experiment.py` — NSP oracle (비교용)
- `scripts/sapp_experiment.py` — SAPP (실패 참고)
- `scripts/prepare_*.py` — 데이터 레이아웃

## 실행 환경

### 실행 명령어
```bash
docker run --rm --gpus all --shm-size=4g \
  -v /home/hun/Volume/DATA:/home/hun/Volume/DATA \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  pilot-anomalydino bash -c "
    pip install scikit-learn tqdm -q 2>/dev/null
    python3 /workspace/scripts/ttns_experiment.py \
      --data_root_ad2 /home/hun/Volume/DATA/mvtec_ad_2_compat \
      --data_root_ad1 /home/hun/Volume/DATA/MVTecAD \
      --output_dir /workspace/results/ttns
  " > run.log 2>&1
```

### 시간 예산
- **실험 시간**: ~5분
- **타임아웃**: 15분

### 지표 추출
```bash
grep "MEAN" run.log | tail -2
```

### 하드웨어
- GPU: NVIDIA RTX 4090, VRAM 24GB

## 실험 전략

### 핵심 방향: 구조적 개선 (hyperparameter 튜닝 금지)

1. **Test-time covariance shift decomposition**: ΔΣ = Σ_test - Σ_train → positive eigenvalue 방향 = nuisance
2. **Mean shift direction 결합**: d = μ_test - μ_train (1차원 추가)
3. **Robust estimation**: anomaly 오염 대응 (iterative trimming, MCD)
4. **Adaptive K selection**: positive eigenvalue 기반 자동 결정 (threshold 불필요)
5. **비교군 구현**: Random projection, AdaBN

### 시도하지 말 것

- K값 sweep (hyperparameter tuning)
- α, τ 등 gating parameter 조정 (SAPP 실패 교훈)
- Paired data 사용 (NSP는 oracle로만)
- Soft projection (hard가 최적이라는 강한 증거)
- Multi-layer fusion (L8+L11 concat 실패)

### 판정 기준

**Keep**: ad2_auroc > 이전 best + 0.5pp, ad1_auroc ≥ 95%
**Discard**: 그 외
**GO/NO-GO**: Day 1-3에 TTNS ≥ 67% 확인

## 12개 실패 방법의 교훈 (반드시 참조)

- Projection은 바꾸지 마라 → hard orthogonal이 최적
- Estimation만 바꿔라 → paired → unpaired가 이번 핵심
- Soft/partial removal은 해로움
- Normal variance ≠ shift variance (train-only 실패)
- Complexity 추가는 해로움
