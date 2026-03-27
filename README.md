# FMAD — Foundation Model Anomaly Detection Robustness

FM(DINOv2/CLIP) 기반 Visual Anomaly Detection의 distribution shift 취약성을 분석하고 해결하는 연구 프로젝트.

## 핵심 발견

1. **FM-AD는 깨진다**: 3개 SOTA 방법 모두 MVTec AD 2에서 25-34pp I-AUROC 하락
2. **원인은 Linear Entanglement**: DINOv2 feature에서 nuisance(shift)와 anomaly 신호가 같은 PC를 공유 (correlation 0.53)
3. **단순한 해법이 최적**: Post-hoc linear projection (NSP)으로 +29.4pp 개선, clean 성능 유지. 10개 복잡한 변형 모두 열등.

## 프로젝트 구조

```
├── scripts/                    # 실험 스크립트 (아래 참조)
├── baselines/                  # Baseline 방법 코드
│   ├── anomalyclip/            # AnomalyCLIP (ICLR 2024)
│   ├── anomalydino/            # AnomalyDINO (WACV 2025)
│   └── dinomaly/               # Dinomaly (CVPR 2025)
├── docker/                     # Docker 이미지 (3개 baseline)
├── results/                    # 실험 결과 JSON
│   ├── phase1/                 # Baseline 재현
│   ├── phase2/                 # Feature 분석
│   └── phase3/                 # NSP 실험
├── skill_graph/                # 연구 노트 & 실험 보고서
│   ├── experiments/            # 6단계 실험 보고서
│   ├── analysis/               # 분석 문서
│   ├── papers/                 # 논문 요약
│   └── ideas/                  # 아이디어
├── plan.md                     # 5-Phase 연구 계획
├── program.md                  # Autoresearch 루프 설정
├── handoff.md                  # 인수인계 메모
└── results.tsv                 # 전체 실험 이력
```

## 빠른 시작

### 데이터 준비

```bash
# 데이터 경로 (수동 다운로드 필요)
/home/hun/Volume/DATA/MVTecAD/          # MVTec AD (15 categories)
/home/hun/Volume/DATA/mvtec_ad_2/       # MVTec AD 2 (8 categories)
/home/hun/Volume/DATA/RobustAD/         # RobustAD (3 categories)

# MVTec AD 2 호환 레이아웃 생성
python scripts/prepare_mvtecad2_compat.py --src /home/hun/Volume/DATA/mvtec_ad_2 --dst /home/hun/Volume/DATA/mvtec_ad_2_compat

# RobustAD 호환 레이아웃 생성
python scripts/prepare_robustad_compat.py --src /home/hun/Volume/DATA/RobustAD --dst /home/hun/Volume/DATA/RobustAD_compat
```

### Docker 빌드

```bash
docker build -t pilot-anomalydino -f docker/Dockerfile.anomalydino .
docker build -t pilot-anomalyclip -f docker/Dockerfile.anomalyclip .
docker build -t pilot-dinomaly -f docker/Dockerfile.dinomaly .
```

### NSP 실험 (Best Config)

```bash
docker run --rm --gpus all --shm-size=4g \
  -v /home/hun/Volume/DATA:/home/hun/Volume/DATA \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  pilot-anomalydino bash -c "
    pip install scikit-learn tqdm -q
    python3 /workspace/scripts/phase3_nsp_experiment.py \
      --data_root_original /home/hun/Volume/DATA/mvtec_ad_2 \
      --data_root_ad2 /home/hun/Volume/DATA/mvtec_ad_2_compat \
      --data_root_ad1 /home/hun/Volume/DATA/MVTecAD \
      --layer 8 --n_components 100 --scoring mahalanobis
  "
```

## 스크립트 참조

| 스크립트 | 용도 |
|----------|------|
| `phase3_nsp_experiment.py` | NSP 방법론 실험 (feature extraction + projection + scoring) |
| `phase2_feature_analysis.py` | DINOv2 layer별 shift 정량화 |
| `phase2_entanglement_analysis.py` | Nuisance-anomaly entanglement PCA 분석 + t-SNE |
| `prepare_mvtecad2_compat.py` | MVTec AD 2 → MVTec AD 호환 symlink 생성 |
| `prepare_robustad_compat.py` | RobustAD → MVTec AD 호환 symlink 생성 |
| `run_anomalyclip.sh` | AnomalyCLIP baseline 실행 |
| `run_anomalydino.sh` | AnomalyDINO baseline 실행 |
| `run_dinomaly.sh` | Dinomaly baseline 실행 |
| `collect_results.py` | Baseline 결과 수집 → Markdown 표 |
| `verify_datasets.py` | 데이터셋 구조 검증 |

## 현재 상태 (2026-03-27)

- **Phase 1-3 완료**: Baseline 재현 → Feature 분석 → NSP 방법론
- **Best 결과**: AD2 83.8% (+29.4pp), AD1 96.3% (-0.2pp)
- **방향**: 방법론 novelty 탐색 중, fallback으로 분석+이론 논문

상세: `handoff.md`, `skill_graph/experiments/` 참조.

## 환경

- GPU: NVIDIA RTX 4090 (24GB)
- CUDA: 12.8, Driver: 570
- Docker: PyTorch 2.5.1+cu124
