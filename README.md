# LoRA Fine-Tuning GPT-OSS 20B with Unsloth + TRL

This repository demonstrates an end-to-end workflow for fine-tuning the **GPT‑OSS 20B** model using **LoRA adapters** and **Hugging Face TRL** for efficient supervised fine‑tuning (SFT). The pipeline is designed to reduce GPU memory usage with **4‑bit quantization** while enabling high‑quality instruction following and multilingual reasoning.

---

## 🚀 Features
- **4-bit quantized loading** of GPT‑OSS 20B for reduced VRAM usage.
- **LoRA adapters** to fine‑tune ~1% of model parameters efficiently.
- **Multilingual-Thinking dataset** for reasoning and chain‑of‑thought training across multiple languages.
- **Hugging Face TRL SFTTrainer** for supervised fine‑tuning.
- **Pre‑ and post‑training inference tests** with reasoning effort control (`low`, `medium`, `high`).

---

## 📂 Project Structure
```
.
├── train_gpt_oss_lora.py     # Main training script (LoRA + TRL)
├── requirements.txt          # Python dependencies
└── outputs/                  # Folder where checkpoints are saved
```

---

## ⚙️ Installation
### 1. Clone the repo
```bash
git clone https://github.com/h-abid97/sft-gpt-oss
```

### 2. Create and activate environment
```bash
conda create -n unsloth-gpt-oss python=3.10 -y
conda activate unsloth-gpt-oss
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Fine-Tuning

Run the training script:
```bash
python train.py
```

### Checkpoints & Saving
After training, checkpoints are saved to:
```
outputs/
```

---

## 🧪 Inference Example
Run a quick inference before/after training:
```python
messages = [
    {"role": "system", "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems."},
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

_ = model.generate(**inputs, max_new_tokens=256, streamer=streamer)
```

You can control **reasoning depth vs. speed** with the `reasoning_effort` parameter (`low`, `medium`, `high`) when using Unsloth chat templates.

---

## 💡 Notes
- The provided script is configured for a **short demo run** (`max_steps=60`). Increase `max_steps`, adjust learning rate, or tune LoRA parameters for production training.
- The `outputs/` directory will contain the trained checkpoints.

---

## 📜 License
This repository shares example code for research and educational purposes.
