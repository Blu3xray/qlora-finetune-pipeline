# QLoRA Fine-Tuning Pipeline — Gemma 2 9B on Apple Silicon (MLX)

End-to-end QLoRA fine-tuning pipeline for **Gemma 2 9B** using Apple's **MLX** framework on Apple Silicon. Includes dataset generation, JSONL validation, LoRA training, and a multi-adapter evaluation system for comparing training intensity effects.

**Use case / themed dataset:** The dataset is styled after **Arasaka Corporation** from the *Cyberpunk* universe — a hyper-corporate entity with a distinct authoritarian voice. This makes it an ideal persona for studying fine-tuning behaviour: the base model has no strong prior alignment to this style, so any persona shift is directly attributable to training. The dataset consists of in-character Q&A from a corporate spokesperson perspective. Multi-adapter comparison (`light` / `medium` / `heavy`) allows systematic measurement of "saturation" — how strongly fine-tuning overrides base model tendencies.

## Key Features

- **QLoRA via MLX** — memory-efficient fine-tuning on Apple Silicon without CUDA
- **Gemma 2 9B** (4-bit quantised) — production-scale model running locally
- **Dataset generator** — parameterised JSONL pipeline (`--train-size`, `--valid-size`)
- **JSONL validator** — checks format and Gemma chat tags before training
- **Multi-adapter comparison** — train multiple adapters at different intensities, evaluate side-by-side
- **VS Code Tasks** — one-click training, evaluation, and data generation

---

## Project Structure

```
├── .vscode/
│   └── tasks.json              # VS Code Tasks: training + comparisons + evaluation
├── generate_large_dataset.py   # Large dataset generator (1000+)
├── validate_data.py            # JSONL and Gemma tag validator
├── eval.py                     # Post-training model evaluation script
├── data/
│   ├── train_large.jsonl       # Large training set (1000)
│   └── valid_large.jsonl       # Large validation set (100)
└── adapters/                   # Trained LoRA weights storage
```

---

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Access to Hugging Face (model `google/gemma-2-9b-it`)

Before first training:
- Log in via CLI: `huggingface-cli login`
- Visit the `google/gemma-2-9b-it` model page and accept the terms (if required), otherwise downloads might be blocked.

### Installation

```zsh
pip install mlx mlx-lm
```

Optional (if you need HF CLI):

```zsh
pip install huggingface_hub
huggingface-cli login
```

---

## Data Generation

### Large Dataset (1000 train / 100 valid)

```zsh
python3 generate_large_dataset.py --train-size 1000 --valid-size 100
```

You can change the size (e.g., 3000/300):

```zsh
python3 generate_large_dataset.py --train-size 3000 --valid-size 300
```

### Validation

```zsh
python3 validate_data.py
```

---

## QLoRA Training

> **Note:** Flag names may vary depending on the MLX-LM version. Check `python3 -m mlx_lm.finetune --help`.

### Simple Training Command

```zsh
caffeinate -dimsu python3 -m mlx_lm lora \
  --model google/gemma-2-9b-it \
  --train \
  --data data \
  --iters 600 \
  --batch-size 2 \
  --num-layers 16 \
  --learning-rate 2e-5 \
  --adapter-path adapters/arasaka-gemma2-9b
```

> **Note:** The `--data` flag expects a directory containing `train.jsonl` and `valid.jsonl`.


---

## Evaluation

### Single Generation

```zsh
python3 -m mlx_lm.generate \
  --model google/gemma-2-9b-it \
  --adapter adapters/arasaka-gemma2-9b \
  --prompt "<start_of_turn>user
Why trust Arasaka with our data and lives?<end_of_turn>
<start_of_turn>model"
```

### Batch Evaluation (Validation Set)

```zsh
python3 eval.py --adapters adapters/arasaka-gemma2-9b --data data/valid_large.jsonl --samples 10
```

### Comparing Adapter Intensities

Train multiple adapters at different iterations/learning rates, then compare outputs on the same prompts:

```zsh
python3 eval.py \
  --adapters adapters/arasaka-light adapters/arasaka-medium adapters/arasaka-heavy \
  --data data/valid_large.jsonl \
  --samples 3
```

---

## Troubleshooting

| Issue                          | Solution                                         |
|--------------------------------|--------------------------------------------------|
| Out of memory                  | Decrease `batch_size` to 1                       |
| Overfitting (val_loss rises)   | Decrease `learning_rate` to 1e-5 or reduce iters |
| Unknown flags                  | Check `--help` and adjust flag names             |

---

## VS Code Tasks

The project includes `.vscode/tasks.json` with the following tasks:

- **Train QLoRA** – Full training session
- **Train QLoRA - Level 1/2/3** – Three adapters with varying intensities
- **Compare propaganda levels** – Side-by-side comparison of multiple adapters
- **Validate Data** – Checks JSONL format and tags
- **Evaluate Model** – Generates test responses via `eval.py`
- **Generate Large Dataset** – Creates `train_large.jsonl` and `valid_large.jsonl`

Run via `Cmd+Shift+P` → `Tasks: Run Task`.

---

## License

Educational / Fan-made project set in the Cyberpunk 2077 universe. Arasaka™ is property of CD Projekt RED / R. Talsorian Games.
