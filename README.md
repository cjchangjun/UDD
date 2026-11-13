# UDD Mamba Project: Phase 1 재현과 Phase 2 (T3)

XAI639 term project의 Phase 1(Tasks 1.3–1.5)에서 쓴 데이터, data split, 코드, 학습된 weight를 모았습니다.
Phase 2 T3(effective kernel 분석)는 `effective_kernel/`에 있고, 정리 문서는 `effective_kernel/T3_report.pdf`입니다.

## 폴더 구성

```
UDD_Mamba_Phase1/
├── mamba_starter/      과제 starter code (v1.1-student, 수정 없음)
├── data/har/           UCI HAR Dataset (직접 다운로드, 아래 "데이터 받기")
├── splits/             Phase 1에서 쓴 train/validation split
├── checkpoints/        모델별, seed별 best weight와 학습 기록
├── scripts/            학습, 재평가, frequency response 스크립트
├── effective_kernel/   Phase 2 (T3): effective kernel 코드, 분석 노트북, 정리 문서(T3_report)
├── results/            Phase 1 출력(scripts/frequency_response.py)과 T3 결과(results/t3/)
└── requirements.txt
```

| 폴더 | 내용 |
|---|---|
| `mamba_starter/` | 모델(`models.py`, `mamba_block.py`), 학습(`train.py`, `train_ema.py`), 평가(`evaluate.py`) 코드. 파일이 `MANIFEST.json`의 SHA-256과 모두 일치합니다. |
| `checkpoints/<model>/seed_<s>/` | `best.pt`(best validation macro-F1 epoch의 weight, 설정과 split 포함), `history.json`(epoch별 기록), `validation.json`, `config.json` |
| `checkpoints/<model>/summary.json` | seed 3개의 결과(Phase 1 보고서 수치의 원본) |
| `checkpoints/ema_logreg/` | EMA filter bank + logistic regression의 seed별 `joblib` 파일 |

`config.json`과 `best.pt` 안의 `data_root`, `out` 경로는 원래 학습한 서버의 경로입니다. 스크립트는 이 값을 쓰지 않고 `data/har`를 직접 지정합니다.

## 설치

Python 3.12에서 확인했습니다.

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # CPU: .../whl/cpu
pip install -r requirements.txt
```

## 데이터 받기

데이터는 포함하지 않았습니다. starter의 다운로드 함수로 공식 UCI 사이트에서 받습니다.

```bash
cd mamba_starter
python -c "from data import load_har; load_har('../data/har', download=True)"
cd ..
```

- `data/har/UCI HAR Dataset/`이 생깁니다.
- 받은 zip의 SHA-256이 `data/har/download_manifest.json`에 기록됩니다. Phase 1에서 쓴 파일은 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`입니다. 같으면 같은 데이터입니다.

## Data split

starter의 기본 subject-wise split을 그대로 썼습니다. 모든 모델과 seed가 같은 split을 씁니다.

- `mamba_starter/data.py`의 `subject_split`이 공식 train set의 피험자 21명을 섞어(`split_seed=42`) 20%(`val_fraction=0.2`, 올림하여 5명)를 validation으로 고릅니다. `train.py`와 `train_ema.py`가 이 기본값을 씁니다.
- 같은 데이터와 같은 기본값이면 항상 아래 split이 나옵니다. `scripts/evaluate_all.py`는 평가 전에 각 checkpoint의 split이 `splits/har_split.json`과 같은지 확인합니다.

| | 피험자 | window 수 |
|---|---|---|
| train | 1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 19, 22, 23, 27, 28, 29 (16명) | 5,392 |
| validation | 17, 21, 25, 26, 30 (5명) | 1,960 |
| official test | 사용하지 않음 (최종 프로토콜을 정한 뒤에만 사용) | – |

- train과 validation은 모두 UCI HAR 공식 train set 안에서 나눴습니다.
- `splits/har_split.json`에 피험자 목록과 window 인덱스(`train_indices`, `val_indices`)가 있습니다. 인덱스는 공식 train set(`data/har/UCI HAR Dataset/train`)의 행 번호입니다.
- 학습 seed(0, 1, 2)는 모델 초기화와 batch 순서만 바꾸고 split은 바꾸지 않습니다.
- 입력 정규화는 train 부분에서만 계산하며 각 `best.pt`에 저장되어 있습니다.

## 모델과 결과 (validation, seed 0/1/2의 mean ± sample std)

모두 starter 기본 설정입니다: 50 Hz, window 128, 20 epochs, batch 64, lr 1e-3, weight decay 0.01, best validation macro-F1 epoch 선택.

| 폴더 | 모델 | 학습 옵션 | Params | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `ema_logreg` | EMA (a = 0.5, 0.8, 0.9, 0.95, 0.99) + logistic regression | `train_ema.py --alphas 0.5 0.8 0.9 0.95 0.99` | 135 features | 0.9046 ± 0.0000 | 0.9024 ± 0.0000 |
| `cnn` | 2-layer causal 1D CNN (kernel 5, 32 ch) | `--model cnn` | 6,822 | 0.9248 ± 0.0122 | 0.9295 ± 0.0122 |
| `cnn_width59` | 같은 CNN, 59 ch (Mamba와 parameter 수를 맞춤) | `--model cnn --d_model 59` | 20,538 | 0.9320 ± 0.0085 | 0.9359 ± 0.0080 |
| `mamba_selective` | Mamba, 2 blocks (selective) | `--model mamba` | 20,454 | 0.9507 ± 0.0041 | 0.9548 ± 0.0041 |
| `mamba_lti` | Mamba, 2 blocks (LTI) | `--model mamba --lti` | 15,910 | 0.9571 ± 0.0022 | 0.9609 ± 0.0021 |

EMA의 logistic regression(lbfgs)은 결정적이라 세 seed의 결과가 같습니다(std 0).

## 사용법

명령은 모두 이 폴더(`UDD_Mamba_Phase1/`)에서 실행합니다.

**1. 저장된 weight로 결과 확인**

```bash
python scripts/evaluate_all.py
```

`checkpoints/`의 모든 모델을 validation에서 다시 평가해 위 표를 출력합니다. seed별 confusion matrix가 `summary.json`과 같으면 `OK`로 표시합니다.

- GPU에서 실행하면 모든 모델이 보고된 값과 정확히 일치합니다.
- CPU에서는 부동소수점 차이로 CNN seed 2의 1,960개 window 중 1개가 다르게 분류됩니다(accuracy 0.9194 → 0.9189). 나머지는 일치합니다.

**2. 처음부터 다시 학습**

```bash
bash scripts/train_all.sh                    # 다른 python을 쓰려면: PY=/path/to/python bash scripts/train_all.sh
python scripts/evaluate_all.py --root runs   # 다시 학습한 모델 확인
```

- 결과는 `runs/`에 저장됩니다(`checkpoints/`는 건드리지 않습니다).
- GPU 4개(0–3)를 씁니다. Mamba와 LTI의 seed s는 GPU s에서, CNN 두 종류는 GPU 3에서 학습합니다. Mamba 한 seed가 약 10분 걸립니다.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`로 cuBLAS를 결정적으로 설정합니다. 같은 GPU 종류(RTX A6000)와 위 버전에서 보고된 값이 그대로 나옵니다. 다른 GPU나 CPU에서는 window 몇 개 정도 달라질 수 있습니다.

모델 하나만 학습하려면 starter를 직접 쓰면 됩니다.

```bash
cd mamba_starter
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --dataset har --data_root ../data/har --model mamba --seeds 0 1 2 --out ../runs/my_mamba
python evaluate.py --checkpoint ../checkpoints/mamba_selective/seed_0/best.pt --data_root ../data/har --device cuda --out ../runs/eval.json
```

**3. Frequency response (Tasks 1.3, 1.5)**

```bash
python scripts/frequency_response.py
```

| 출력 | 내용 |
|---|---|
| `results/ema_filters.json` | EMA 계수별 half-life, half-power cutoff, Δ (A = −1일 때) |
| `results/ema_responses.png` | EMA의 impulse response와 frequency response (Task 1.3) |
| `results/lti_kernel_responses.png` | 학습된 LTI 모델(seed 0, layer 0)의 커널을 L = 16, 32, 128로 잘라 FFT한 결과와 잘리지 않은 커널의 정확한 응답 비교 (Task 1.5) |

## Effective kernel (Phase 2, T3)

T3 "What Filter Has Mamba Learned?" 파트입니다.

- **정리 문서**: `effective_kernel/T3_report.pdf`. 정의, 검증, 결과(그림 포함), 요구사항 점검이 들어 있습니다.
- **노트북**: `effective_kernel/t3_analysis.ipynb`. 문서의 모든 수치와 그림을 차례로 확인합니다. `effective_kernel/` 폴더에서 실행합니다.

Mamba의 SSM은 window 하나, layer 하나, SSM 채널(d_inner = 64) 하나마다 인과적 선형 사상 y = K u입니다. `effective_kernel/effective_kernel.py`는 이 K를 Ali et al. (2024)의 hidden-attention 식(Eq. 12)에 starter SSM의 skip 항 D를 대각선에 더한 형태로 계산합니다.

```
K[i, j] = Σ_n C_i[n] · exp(A[n] · (Δ_{j+1} + … + Δ_i)) · Δ_j · B_j[n]   (j ≤ i)
        + D                                                            (i = j)
```

- u는 SSM 입력(block 안의 RMSNorm → in_proj → conv1d → SiLU 이후), y는 gate(SiLU(z))를 곱하기 전의 SSM 출력입니다. 행 i는 출력 시점, 열 j는 입력 시점입니다.
- starter는 수정하지 않습니다. Δ, A, B, C는 starter의 `set_record()`로, D는 모델 파라미터에서 읽습니다.

```python
import sys, torch; sys.path[:0] = ["mamba_starter", "effective_kernel"]
from evaluate import load_checkpoint, evaluation_data
from effective_kernel import effective_kernels

model, ckpt = load_checkpoint("checkpoints/mamba_selective/seed_0/best.pt", "cuda")
x, y = evaluation_data(ckpt, data_root="data/har")              # 정규화된 validation 입력
logits, K = effective_kernels(model, torch.as_tensor(x[:64], device="cuda"))

K.shape          # (64, 2, 64, 128, 128) = (sample, layer, channel, i, j)
K[3, 0, 10]      # sample 3, layer 0, channel 10의 [128, 128] 행렬
K[3, 0]          # [64, 128, 128]
K.dense()        # 전체 (64 sample이면 float64로 약 1 GB)
```

- `K`는 Δ, A, B, C, D만 저장하고, 인덱싱할 때 요청한 행렬만 float64로 계산합니다. 인덱스는 int, slice, list, bool mask(CPU) 모두 됩니다.
- `logits`는 `model(x)`와 같습니다. `include_skip=False`로 하면 D를 뺀 논문 식 그대로의 행렬을 얻습니다.

### 파일 구성 (`effective_kernel/`)

| 파일 | 내용 |
|---|---|
| `T3_report.pdf` | T3 정리 문서 (정의, 검증, 결과, 요구사항 점검) |
| `t3_analysis.ipynb` | 문서의 수치와 그림을 확인하는 노트북 (그림 1, 2를 `results/t3/`에 저장) |
| `effective_kernel.py` | hidden-attention kernel K (위의 식), lazy 인덱싱 |
| `check_effective_kernel.py` | 검증 (LTI에서 `lti_kernel`과 일치 등) |
| `spectra.py` | K의 행 스펙트럼과 class별 요약 (시간, example, 채널에 걸친 평균) |
| `run_t3.py` | seed 0, 1, 2의 class별 스펙트럼 계산, 그림 3과 `t3_summary.json` 저장 |
| `plots.py` | 그림 3 |
| `jacobian.py` | whole-block Jacobian (확장) |
| `effective_kernel_demo.ipynb` | K의 사용 예와 행렬 그림 (간단한 확인용) |

### 검증

```bash
python effective_kernel/check_effective_kernel.py
```

selective와 LTI 모델의 모든 seed에서 확인합니다. 모델은 float64로 돌립니다.
- LTI 모델의 K가 starter `lti_kernel(128)`의 Toeplitz 행렬과 같은지(D 포함과 제외 모두, 오차 1e-15 수준)
- `K u`가 starter `ssm()` 출력과 같은지, `logits`가 `model(x)`와 같은지
- `K[s, l, c]`가 `K.dense()[s, l, c]`와 비트 단위로 같은지
- Jacobian: block Jacobian이 중심 유한 차분과 같은지, LTI에서 SSM 단계만의 Jacobian이 K와 같은지

### T3 분석 실행

```bash
python effective_kernel/run_t3.py                 # seed 0 1 2, 모든 validation window (seed당 약 8분)
python effective_kernel/run_t3.py --seeds 0       # seed 하나만 (seed별로 병렬 실행 가능)
python effective_kernel/run_t3.py --plots_only    # 저장된 결과에서 그림 3과 요약만 다시 만들기
```

validation data만 쓰고 official test set은 읽지 않습니다. 결과는 "Mamba의 frequency response"가 아니라 입력에 조건부인 요약입니다.

| 출력 (`results/t3/`) | 내용 | 만드는 곳 |
|---|---|---|
| `t3_input_windows.png` | 그림 1: 세 validation window의 입력과 예측 | 노트북 |
| `t3_kernel_matrices.png` | 그림 2: 같은 window의 K 행렬 (layer 0, 채널 0) | 노트북 |
| `t3_class_spectra.png` | 그림 3: class별 평균 스펙트럼 (layer별, LTI 커널 곡선 포함) | `run_t3.py` |
| `t3_summary.json` | 요약 설정과 class 평균 스펙트럼 사이의 거리 (seed 평균) | `run_t3.py` |
| `selective_D_seed<s>.npz`, `lti_D_seed<s>.npz` | class별 스펙트럼 (노트북이 읽음) | `run_t3.py` |

노트북은 `results/t3/`의 npz가 있으면 읽고, 없으면 class당 일부 window로 직접 계산합니다.

### Conditional summary 사용법

class별 요약 곡선은 K의 각 행(출력 시점 i의 필터)을 FFT한 크기를 시간(i 전체) → class의 window → SSM 채널 순서로 평균한 것입니다(정의는 `T3_report.pdf` 5.1절). 저장된 결과를 읽거나 직접 계산할 수 있습니다.

```python
import sys, numpy as np; sys.path[:0] = ["mamba_starter", "effective_kernel"]
from evaluate import load_checkpoint, evaluation_data
import spectra

# 1) 저장된 결과 (validation window 전체, seed 0)
spec = dict(np.load("results/t3/selective_D_seed0.npz"))

# 2) 직접 계산 (max_per_class를 빼면 전체 window, seed 하나에 약 7분)
model, ckpt = load_checkpoint("checkpoints/mamba_selective/seed_0/best.pt", "cuda")
x, y = evaluation_data(ckpt, data_root="data/har")               # validation window
spec = spectra.class_mean_spectra(spectra.hidden_attention(model), x, y, max_per_class=20)

S = spectra.channel_mean(spec)     # [class 6, layer 2, 주파수 65]: class별 요약 곡선
spec["freq"]                       # 0–25 Hz, 65개 (간격 50/128 Hz)
spec["count"]                      # class별 window 수
```

class 순서는 `plots.CLASSES`(WALKING, UPSTAIRS, DOWNSTAIRS, SITTING, STANDING, LAYING)입니다. seed 0, 1, 2의 곡선을 평균하고 표준편차를 낸 것이 그림 3입니다(`plots.fig_class_spectra`).
