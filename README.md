# kieu-gpt

A character-level GPT trained on [Truyện Kiều](https://vi.wikisource.org/wiki/Truy%E1%BB%87n_Ki%E1%BB%81u) by Nguyễn Du — a classic Vietnamese poem of 3,254 verses.

Based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) / makemore lecture series.

## Setup

```bash
conda activate pytorch2   # PyTorch is the only dependency
python v2.py              # trains and generates sample text
```

`input.txt` must be present in the working directory (included in this repo).

## Branches

| Branch | Data | Config |
|--------|------|--------|
| `master` | TinyShakespeare | Karpathy's original (n_embd=384, 6 heads, 6 layers) |
| `truyen-kieu` | Truyện Kiều | Tuned for smaller dataset (n_embd=128, 4 heads, 4 layers) |

## Training run

![Training output](assets/training-run.png)

Full loss curve (CPU, `torch.manual_seed(1337)`, reproducible):

```
step    0: train 5.0752, val 5.0656
step  300: train 2.2409, val 2.2449
step  600: train 2.1194, val 2.1262
step  900: train 1.9816, val 1.9916
step 1200: train 1.8815, val 1.8927
step 1500: train 1.8073, val 1.8209
step 1800: train 1.7462, val 1.7593
step 2100: train 1.6983, val 1.7198
step 2400: train 1.6589, val 1.6882
step 2700: train 1.6276, val 1.6596
step 2999: train 1.5999, val 1.6395
```

Train/val gap at convergence: **0.04** — no meaningful overfitting. CUDA runs may differ by ±0.02 due to nondeterminism.

## Parameter count

Verified with `sum(p.numel() for p in model.parameters())`:

| Component | Parameters |
|---|---|
| Token embedding (132 × 128) | 16,896 |
| Position embedding (128 × 128) | 16,384 |
| Transformer blocks × 4 | 791,552 |
| Final LayerNorm | 256 |
| LM head (128 × 132 + bias) | 17,028 |
| **Total** | **842,116** |

## Sample output

Generated from `context = [[0]]`, 500 tokens, `torch.manual_seed(1337)` (CPU):

```
Bâu đoắn trăm váo khánh mới chiều,
Thách Thóc lại mà kổ đệ thoán diêu,
Vô làm yêu nhơ có thân nỗn sốm tày xanh.
Lần danh Khóa đừ đã vì lại trong.
Ngồng lâu cho chường,
Thai mơ giọc nài ngheo bước Phòng.
Tài cơn rằng: Sầu với vày,
Ki chẳng nhen vào giở trửa ý thôi.
Hoạnh sấn lại a gửi đên.
Nghĩ đền nàng cao Tuồn gia.
```

Syllable counts (Vietnamese: 1 word = 1 syllable; lục bát = strict 6/8 alternating):

| Line | Count | Expected | Match |
|---|---|---|---|
| Bâu đoắn trăm váo khánh mới chiều | 7 | 6 (lục) | ✗ |
| Thách Thóc lại mà kổ đệ thoán diêu | 8 | 8 (bát) | ✓ |
| Vô làm yêu nhơ có thân nỗn sốm tày xanh | 10 | 6 (lục) | ✗ |
| Lần danh Khóa đừ đã vì lại trong | 8 | 8 (bát) | ✓ |
| Ngồng lâu cho chường | 4 | 6 (lục) | ✗ |
| Thai mơ giọc nài ngheo bước Phòng | 7 | 8 (bát) | ✗ |
| Tài cơn rằng: Sầu với vày | 6 | 6 (lục) | ✓ |
| Ki chẳng nhen vào giở trửa ý thôi | 8 | 8 (bát) | ✓ |
| Hoạnh sấn lại a gửi đên | 6 | 6 (lục) | ✓ |
| Nghĩ đền nàng cao Tuồn gia | 6 | 8 (bát) | ✗ |

Strict alternation match: **5/10 lines** (50%). Lines fall at 6 or 8 syllables ~60% of the time, but without consistent alternation.

## Hyperparameter notes

Truyện Kiều (~104K chars) is ~10x smaller than TinyShakespeare (~1.1M chars). The original config overfits badly (train loss 0.07, val loss 3.4 by step 3000). Reducing model capacity keeps train/val loss aligned.

| | Shakespeare config | Kiều config |
|---|---|---|
| `n_embd` | 384 | 128 |
| `n_head` | 6 | 4 |
| `n_layer` | 6 | 4 |
| `dropout` | 0.2 | 0.3 |
| `block_size` | 256 | 128 |
| `max_iters` | 5000 | 3000 |

## Limitations

- **Meter**: Character-level modeling does not reproduce lục bát meter reliably. ~60% of generated lines land at 6 or 8 syllables, but strict 6→8→6→8 alternation is absent. Syllable-level tokenization would likely improve this.
- **Corpus size**: 104K characters is a small training set. The tight train/val gap shows the model is not underfitting, but generated vocabulary includes nonsense syllables (*nỗn*, *sốm*, *diêu*) that do not exist in Vietnamese.
- **No checkpointing**: Each run retrains from scratch in ~10 minutes on CPU. `torch.manual_seed(1337)` is set; CUDA runs may produce different samples due to nondeterminism.
