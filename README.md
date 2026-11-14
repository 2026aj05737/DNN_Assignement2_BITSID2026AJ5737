# DNN Programming Assignment-2: RNN, LSTM, GRU and Transformer
- **Name:** [SURYA PRAKASH]
- **Student ID:** 2026aj05737

## Chosen Hyperparameters & Variant
- **Problem 1 Hidden Size ($h$):** 64 (applied identically to RNN, LSTM, and GRU)
- **Problem 2 Model Width ($d$):** 128 ($d_{ff} = 512$, 4 heads, 4 layers)
- **Random Seeds:** 42, 123, 999
- **Problem 1 Task 5(c) Ablation (Odd Student ID ending in 7):** LSTM forget-gate bias initialized to `0.0` vs `1.0` at $T=200$.

## Environment
- Python 3.10+, PyTorch 2.x, NumPy, pandas, Matplotlib, scikit-learn (see `requirements.txt`).
- Executed on Google Colab with an NVIDIA T4 GPU (`torch.cuda.is_available() == True`).

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open `problem1/problem1.ipynb` and run **Runtime -> Restart and run all**. Datasets (UCI HAR) download automatically via `urllib`, and the adding problem is generated in memory. Figures save to `problem1/figures/` and metrics append to `results.csv`.
3. Open `problem2/problem2.ipynb` and run **Runtime -> Restart and run all**. Tiny Shakespeare downloads automatically. Figures save to `problem2/figures/`, samples save to `problem2/generated_samples.txt`, and metrics append to `results.csv`.

## Runtime Summary (Colab T4 GPU)
- `problem1.ipynb`: ~8 minutes total across all seeds, sequence lengths, and ablations.
- `problem2.ipynb`: ~14 minutes total (including Task A HAR, Task B Shakespeare LM, and context scaling benchmarks).
- Known issues: None. All unit tests pass within `1e-5` tolerance.
