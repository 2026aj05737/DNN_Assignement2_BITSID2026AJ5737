# DNN_Assignement2_BITSID2026AJ57372026AJ05737_DNN_Assignment/
├── README.md
├── requirements.txt
├── report.pdf
├── results.csv
├── problem1/
│   ├── problem1.ipynb
│   └── figures/
└── problem2/
    ├── problem2.ipynb
    ├── generated_samples.txt
    └── figures/
```[cite: 1]

---

## 3. Implementation Details & Equations

### Problem 1: RNN, LSTM, GRU & Sequence Tasks

#### Task 1: Data Pipelines
1. **UCI HAR Split:** Split by training subjects using `np.random.default_rng(seed).choice(train_subjects, 4, replace=False)`[cite: 1].
2. **Normalization:** Standardize channels using mean ($\mu$) and standard deviation ($\sigma$) computed **only** from the training split:
   $$\hat{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-8}}$$
[cite: 1]
3. **Adding Problem Unit Test:** Validate every sequence has exactly 2 markers (one in $t < T/2$, one in $t \ge T/2$) and $y = x[i_1, 0] + x[i_2, 0]$[cite: 1].

#### Task 2: Parameter Derivation & Math Reconciliation
For input dimension $d = 9$ and hidden dimension $h = 64$[cite: 1]:

* **Vanilla RNN:**
  $$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$
  * Single bias model parameter count: $h \times d + h \times h + h = 64 \times 9 + 64 \times 64 + 64 = 4{,}736$[cite: 1].
  * PyTorch double bias parameter count: $4{,}736 + h = 4{,}800$[cite: 1].

* **LSTM Cell:**
  $$f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$$
  $$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$$
  $$g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})$$
  $$o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$$
  $$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t)$$
  * Single bias parameter count: $4 \times (h \times d + h \times h + h) = 4 \times (576 + 4096 + 64) = 18{,}944$[cite: 1].
  * PyTorch double bias parameter count: $18{,}944 + 4 \times 64 = 19{,}200$[cite: 1].

* **GRU Cell:**
  $$r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})$$
  $$z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})$$
  $$n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))$$
  $$h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$
  * Single bias parameter count: $3 \times (h \times d + h \times h + h) = 3 \times 4736 = 14{,}208$[cite: 1].
  * PyTorch double bias parameter count: $14{,}208 + 3 \times 64 = 14{,}400$[cite: 1].

---

### Problem 2: Transformer from Scratch

#### Task 1 & 2: Attention & Multi-Head Block
* **Scaled Dot-Product Attention:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$
[cite: 1]
  Where $M_{ij} = -\infty$ for causal future tokens ($j > i$)[cite: 1].
* **Positional Encoding:**
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
[cite: 1]
* **Encoder Block Weight Count Derivation:**
  * Attention weights: $W_Q, W_K, W_V, W_O \implies 4 \times d^2$[cite: 1].
  * Feed-Forward weights: $W_1 \in \mathbb{R}^{d \times d_{ff}}, W_2 \in \mathbb{R}^{d_{ff} \times d} \implies 2 \cdot d \cdot d_{ff}$[cite: 1].
  * Total weight matrices per block: $4d^2 + 2 d d_{ff}$[cite: 1]. For $d=128, d_{ff}=512$: $4(128)^2 + 2(128)(512) = 65{,}536 + 131{,}072 = 196{,}608$[cite: 1].
  * Adding biases ($4d + d_{ff} + d = 6d + d_{ff}$) and two LayerNorm scales & shifts ($4d$) gives total parameter count per block:
    $$\text{Params} = 4d^2 + 2 d d_{ff} + (6d + d_{ff}) + 4d$$

---

## 4. Key Script Drafts for Execution

### Notebook 1 Snippet: Custom Cells & Unit Verification
```python
import torch
import torch.nn as nn

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.01)
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, states):
        h_prev, c_prev = states
        gates = (x @ self.W_ih.T + self.b_ih) + (h_prev @ self.W_hh.T + self.b_hh)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# Verification Unit Test against PyTorch Official nn.LSTMCell
def test_lstm_cell():
    torch.manual_seed(42)
    custom_cell = MyLSTMCell(input_size=9, hidden_size=64)
    official_cell = nn.LSTMCell(input_size=9, hidden_size=64)
    
    # Copy parameters
    custom_cell.W_ih.data.copy_(official_cell.weight_ih)
    custom_cell.W_hh.data.copy_(official_cell.weight_hh)
    custom_cell.b_ih.data.copy_(official_cell.bias_ih)
    custom_cell.b_hh.data.copy_(official_cell.bias_hh)
    
    x = torch.randn(16, 9)
    h0 = torch.randn(16, 64)
    c0 = torch.randn(16, 64)
    
    h_c, c_c = custom_cell(x, (h0, c0))
    h_o, c_o = official_cell(x, (h0, c0))
    
    assert torch.allclose(h_c, h_o, atol=1e-5), "LSTM h state mismatch!"
    assert torch.allclose(c_c, c_o, atol=1e-5), "LSTM c state mismatch!"
    print("MyLSTMCell matched PyTorch nn.LSTMCell within 1e-5 successfully!")

test_lstm_cell()
```[cite: 1]

---

### Notebook 2 Snippet: Scaled Dot-Product & Multi-Head Attention
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    # Shapes: (B, H, L, d_k)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

# Verification Test vs torch.nn.functional.scaled_dot_product_attention
def test_attention():
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    
    out_custom, weights = scaled_dot_product_attention(q, k, v)
    out_official = F.scaled_dot_product_attention(q, k, v)
    
    # Verify row sums equal 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
    # Verify outputs match
    assert torch.allclose(out_custom, out_official, atol=1e-5)
    print("Attention unit test passed!")

test_attention()
```[cite: 1]

---

## 5. Required Files to Generate & Populate

### File 1: `results.csv` Template
Create `results.csv` with the exact required header format[cite: 1]:
```csv
problem,model,seed,seq_len,test_acc,macro_f1,test_mse,params,sec_per_epoch,val_loss_nats,bits_per_char,notes
1,RNN,0,128,0.852,0.841,,4800,1.2,,,HAR Classification
1,LSTM,0,128,0.915,0.910,,19200,2.1,,,HAR Classification
1,GRU,0,128,0.921,0.918,,14400,1.8,,,HAR Classification
1,RNN,0,200,,,0.165,4800,0.8,,,Adding Problem
1,LSTM,0,200,,,0.012,19200,1.5,,,Adding Problem (Forget bias=1)
2,Transformer_Classifier,0,128,0.902,0.898,,210000,3.4,,,HAR Task A
2,GPT_TinyShakespeare,0,128,,,,3.50,1.65,2.38,Task B Causal LM
```[cite: 1]

### File 2: `generated_samples.txt`
Sample content for `generated_samples.txt` outputting 500 characters generated at temperatures `0.2`, `0.8`, and `1.5`[cite: 1]:

```text
=== Temperature 0.2 ===
KING RICHARD III:
My lord, I cannot speak of this fair town.
What say you, my lord? I will not hear your words.
DUKE OF YORK:
I pray thee, gentle cousin, speak to him.

=== Temperature 0.8 ===
FIRST SENATOR:
If you be broad in your own counsel, sir,
The noble strength of Rome shall stay the trial.
Which of you all will stand to answer this?

=== Temperature 1.5 ===
CLARENCE:
O, zounds! where's th' standard?
Xy-9! Pox on't! Th' untamed beast hath spoke...
```[cite: 1]

---

## 6. Report Checklist & Execution Steps

1. **Notebook Execution:** Run all cells top-to-bottom without errors (`Restart and Run All`)[cite: 1]. Save inline plots in `./figures/`[cite: 1].
2. **Compile `report.pdf` (Max 6 pages):**
   * **Page 1:** Title block, data preprocessing, HAR class balance, Cell unit verification proofs & equations[cite: 1].
   * **Page 2:** HAR Classification comparison table (Mean ± Std), Confusion matrix[cite: 1].
   * **Page 3:** Adding Problem MSE curves ($T=200$) and solved run tables ($T=50, 100, 200, 400$)[cite: 1].
   * **Page 4:** Gradient norm plot $\Vert \partial L/\partial h_t \Vert$, hidden size sweep, and forget-bias ablation plot ($b_f=0$ vs $b_f=1$)[cite: 1].
   * **Page 5:** Problem 2 Transformer validation (Hand test + SDP unit test), HAR Task A comparison table, Positional encoding ablation[cite: 1].
   * **Page 6:** Tiny Shakespeare learning curves, Attention heatmaps, Head ablation ($H=1,4,8$), Causal mask removal analysis, and hardware resource plot ($O(n^2 \cdot d)$)[cite: 1].
   * **Appendix (does not count toward 6 pages):** AI-Use Statement and References[cite: 1].
3. **Submission Verification:** Download the zipped archive `2026AJ05737_DNN_Assignment.zip` after creating it to ensure `report.pdf` opens cleanly and no folders or cache files exceed 25 MB[cite: 1].
