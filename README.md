# 🏥 Gemma-2-9b-it Healthcare Fine-Tuning with QLoRA + Unsloth

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-orange?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-5.3.0-yellow?logo=huggingface)
![TRL](https://img.shields.io/badge/TRL-5.3-green)
![PEFT](https://img.shields.io/badge/PEFT-0.18.1-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**End-to-end supervised fine-tuning of Google's Gemma-2-9b-it on a 100k medical Q&A dataset using QLoRA and Unsloth-inspired techniques — all within a single Kaggle P100 GPU (16GB VRAM).**

</div>

---

## 📌 Overview

This project fine-tunes [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) — a 9.24 billion parameter instruction-tuned language model — on the [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) dataset, which contains real-world patient–doctor conversations covering a wide range of medical topics.

The primary goal is to specialize the model for **healthcare question answering**, enabling it to respond to patient queries in a clinically informed, context-aware, and structured manner.

Fine-tuning a 9B parameter model from scratch is computationally prohibitive on consumer-grade hardware. This notebook solves that by combining:

- **4-bit QLoRA** (Quantized Low-Rank Adaptation) to drastically reduce VRAM consumption
- **Unsloth-inspired inference patterns** for efficient adapter loading and fast generation
- **Selective layer freezing** to preserve the model's general language capabilities while adapting domain-specific knowledge

The entire training pipeline runs on a **single Kaggle P100 GPU (16GB VRAM)** — no multi-GPU setup required.

---

## 📁 Repository Structure

```
├── notebook.ipynb                  # Main fine-tuning notebook (all steps)
├── README.md                       # Project documentation
└── Ali Sajid/                      # Saved LoRA adapter files
    ├── adapter_model.safetensors   # Trained LoRA weights (~130MB)
    ├── adapter_config.json         # LoRA architecture config
    ├── tokenizer.json              # Tokenizer vocabulary
    ├── tokenizer_config.json       # Tokenizer settings & chat template
    ├── chat_template.jinja         # Gemma-2 chat format template
    └── README.md                   # Model card
```
For running Step 5, go to https://wandb.ai, get an API Key, and paste it into the secret folder in the Kaggle notebook.
And you need another thing, which you get from 'Hugging Face/profile/access tokens/ select write permission and generate token and paste in secret folder, also agree to the terms and conditions on Gemma.2 9b official page on Hugging Face.

---

## 🗂️ Dataset

| Property | Details |
|----------|---------|
| **Name** | `lavita/ChatDoctor-HealthCareMagic-100k` |
| **Source** | HuggingFace Datasets |
| **Total Samples** | ~100,000 |
| **Samples Used** | 5,000 (memory-efficient subset) |
| **Train Split** | 4,500 (90%) |
| **Validation Split** | 500 (10%) |
| **Format** | Instruction / Input / Output (medical Q&A) |

The dataset consists of real patient questions sourced from HealthCareMagic, paired with detailed doctor responses. Each sample is structured as a conversational instruction-following example suitable for supervised fine-tuning.

---

## ⚙️ Model Architecture & Configuration

### Base Model
| Property | Value |
|----------|-------|
| **Model** | `google/gemma-2-9b-it` |
| **Parameters** | 9.24 Billion |
| **Type** | Decoder-only Transformer (instruction-tuned) |
| **Quantization** | 4-bit NF4 (BitsAndBytes) |

### QLoRA Configuration
| Parameter | Value |
|-----------|-------|
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Trainable Parameters** | 33.44M (0.36% of total) |
| **Frozen Layers** | 0–15 (first 16 layers) |
| **Trainable Layers** | 16–41 (upper 26 layers) |

### Why Selective Layer Freezing?
The lower layers of a large language model learn fundamental linguistic patterns (syntax, grammar, token relationships) that are universal across domains. By freezing these, we:
1. Reduce the number of trainable parameters significantly
2. Prevent catastrophic forgetting of general language understanding
3. Focus the adaptation budget entirely on higher-level reasoning layers

### 4-bit NF4 Quantization
BitsAndBytes NF4 (NormalFloat 4-bit) quantization compresses model weights from 16-bit to 4-bit precision using a quantization scheme optimized for normally-distributed neural network weights. With double quantization enabled, this reduces the base model's VRAM footprint from ~18GB to ~8GB — making it feasible to run on a 16GB P100.

---

## 🛠️ Technical Challenges & Solutions

### Challenge 1: Gemma-2 System Role Not Supported
Gemma-2's chat template does not accept a `system` role, causing a `TemplateError` during dataset formatting.

**Solution:** Merge the system instruction into the user's turn:
```python
user_content = f"{row['instruction']}\n\n{row['input']}"
messages = [
    {"role": "user",      "content": user_content},
    {"role": "assistant", "content": row["output"]},
]
```

### Challenge 2: TRL 5.x API Breaking Changes
`SFTTrainer` in TRL 5.x no longer accepts `max_seq_length` or `dataset_text_field` directly — these moved to `SFTConfig`. Additionally, `TrainingArguments` dropped `group_by_length` and `overwrite_output_dir`.

**Solution:** Replaced `TrainingArguments` with `SFTConfig` and updated all parameter names (`max_seq_length` → `max_length`).

### Challenge 3: Double LoRA Wrapping
Calling both `get_peft_model()` manually and passing `peft_config` to `SFTTrainer` caused a `ValueError` due to double-wrapping the model with PEFT.

**Solution:** Removed manual `get_peft_model()` call — let `SFTTrainer` handle LoRA application internally via `peft_config`.

### Challenge 4: BFloat16 / FP16 Conflict on P100
Gemma-2's internal layers use `bfloat16` precision. When `fp16=True` is set, PyTorch's `GradScaler` raises a `NotImplementedError` because it cannot unscale `bfloat16` gradients.

**Solution:** Disabled both `fp16` and `bf16` in `SFTConfig`. With only 33M trainable LoRA parameters, float32 training is stable and fits comfortably within VRAM.

```python
fp16=False,
bf16=False,
gradient_checkpointing_kwargs={"use_reentrant": False},
```

---

## 🏋️ Training Configuration

| Parameter | Value |
|-----------|-------|
| `num_train_epochs` | 3 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| **Effective Batch Size** | 4 |
| `learning_rate` | 2e-4 |
| `lr_scheduler_type` | cosine |
| `warmup_steps` | 100 |
| `weight_decay` | 0.001 |
| `optimizer` | `paged_adamw_32bit` |
| `max_seq_length` | 512 |
| `packing` | False |
| `eval_steps` | 100 |
| `save_steps` | 100 |
| `gradient_checkpointing` | True |
| `fp16` | False |
| `bf16` | False |

---

## Training Results

| Metric | Value |
|--------|-------|
| **Total Steps** | 1,125 |
| **Total Runtime** | ~335 minutes (~5.5 hours) |
| **Final Training Loss** | 2.1909 |
| **Final Validation Loss** | 2.077 |
| **Token Accuracy** | 54.3% |
| **W&B Run** | `giddy-morning-5` |

### Validation Loss Progression
The validation loss decreased steadily throughout training — from **2.255** at step 100 down to **2.077** at the final checkpoint — confirming that the model was learning domain-specific patterns without overfitting.

---

##  Inference — How to Use

### Load the Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./Ali Sajid")
tokenizer = AutoTokenizer.from_pretrained("./Ali Sajid")

print(" Model loaded successfully!")
```

### Run Inference

```python
def generate_response(patient_query, system_prompt=None, max_new_tokens=512):
    if system_prompt is None:
        system_prompt = (
            "You are an expert medical assistant. Provide accurate, "
            "helpful, and compassionate responses to patient queries."
        )

    # Gemma-2 does not support system role — merge into user turn
    user_content = f"{system_prompt}\n\n{patient_query}"
    messages = [{"role": "user", "content": user_content}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Example usage
query = "I have been experiencing chest pain and shortness of breath. What should I do?"
response = generate_response(query)
print(response)
```

---

## 🧰 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12 | Runtime |
| PyTorch | 2.9.0+cu126 | Deep learning framework |
| Transformers | 5.3.0 | Model loading & tokenization |
| PEFT | 0.18.1 | LoRA adapter management |
| TRL | 5.3 | SFTTrainer fine-tuning pipeline |
| BitsAndBytes | latest | 4-bit NF4 quantization |
| Datasets | latest | Dataset loading & processing |
| Weights & Biases | latest | Training monitoring & logging |
| Kaggle P100 GPU | 16GB VRAM | Training hardware |

---

## ⚠️ Disclaimer

This model is intended **for educational and research purposes only**. It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## 👤 Author

**Ali Sajid**
Fine-tuned on Kaggle · Tracked with Weights & Biases · March 2026
