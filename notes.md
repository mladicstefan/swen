# Unsloth — Instruct Model Fine-tuning Reference

> Covers QLoRA/LoRA SFT on instruct models only. Not base model pretraining.

---

## Why Instruct Models (Not Base)

Instruct models are already aligned to follow conversational chat templates (ChatML, ShareGPT, etc.). They require less data to fine-tune and produce usable results faster. Base models need Alpaca/Vicuna-style formatting and significantly more data to match the same output quality. Unsloth explicitly recommends starting with instruct models.

---

## Method 1 — `FastLanguageModel.from_pretrained()`

Loads the base instruct model and tokenizer with Unsloth's patched kernels applied.

| Parameter | What It Does | Watch Out For |
|---|---|---|
| `model_name` | HuggingFace model ID or local path. Names ending in `unsloth-bnb-4bit` are Unsloth's dynamic quants — higher accuracy than standard BnB at slightly more VRAM. Names ending in just `bnb-4bit` are standard BitsAndBytes. | Use Unsloth's own versions when available — they include tokenizer/chat template fixes the originals may not have. |
| `max_seq_length` | Sets the context window. Affects RoPE embeddings and VRAM for activations. Can be any value — Unsloth handles RoPE scaling internally. | Largest single driver of VRAM outside of batch size. 2048–4096 is the practical range for single-GPU consumer hardware. 8192 is possible but tight. |
| `load_in_4bit` | QLoRA mode — quantizes weights to 4-bit NF4 via BitsAndBytes. Reduces weight VRAM by ~75%. | Only one of `load_in_4bit`, `load_in_8bit`, `load_in_16bit`, or `full_finetuning` can be True at a time. Setting more than one crashes. |
| `load_in_8bit` | Loads weights in 8-bit. Middle ground between 4-bit and 16-bit for memory and accuracy. | Less commonly used than 4-bit. No large speed advantage. |
| `load_in_16bit` | Loads in full 16-bit (LoRA, not QLoRA). Slightly more accurate than 4-bit, but uses ~4x more VRAM. | Only viable on GPUs with large VRAM. |
| `full_finetuning` | Trains all parameters, no LoRA adapters. Requires the most VRAM and compute. | Unsloth's docs warn explicitly: if LoRA doesn't work, FFT almost certainly won't either. Start with QLoRA always. |
| `dtype` | Forces compute dtype (e.g. `torch.bfloat16`). `None` autodetects. | `bf16` requires Ampere GPU or newer (RTX 30xx+). On older hardware use `fp16` or `None`. |
| `gpu_memory_utilization` | Fraction of total GPU VRAM Unsloth is allowed to allocate (0.0–1.0). | Values above 0.95 risk OOM from OS/driver overhead. 0.85–0.90 is the stable range. |
| `use_gradient_checkpointing` | `True` = standard checkpointing. `"unsloth"` = Unsloth's optimized version that cuts an additional 30% VRAM versus standard. Enables very long context fine-tuning. | Has a minor speed cost (recomputes activations during backward pass). Worth it on memory-constrained hardware. |
| `trust_remote_code` | Allows execution of custom model code from HuggingFace. Required by some newer model architectures. | Security risk — only enable for models you trust. Disabled by default. |
| `token` | HuggingFace API token. Required for gated models (e.g., Llama 3). | Leave as comment until needed. |

---

## Method 2 — `FastLanguageModel.get_peft_model()`

Attaches LoRA adapter matrices to the loaded model. After this call, only adapter weights are trainable.

| Parameter | What It Does | Watch Out For |
|---|---|---|
| `r` (rank) | Bottleneck dimension of the LoRA matrices A and B. Higher = more trainable parameters, more capacity, more VRAM, slower training. | Values above 64 have rapidly diminishing returns and increase overfitting risk. For most instruct fine-tunes, 16–32 is sufficient. Too large a rank on a small dataset will overfit. |
| `target_modules` | Which linear layers inside the transformer receive adapters. Targeting both attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`) layers is empirically optimal per the QLoRA paper. | Removing modules reduces VRAM minimally but meaningfully hurts quality. Unsloth recommends targeting all 7 major layers. |
| `lora_alpha` | Scaling factor applied to LoRA updates: `alpha / rank`. Controls how strongly the adapter affects the weights. | Setting `alpha = rank` gives a scale of 1.0 (neutral). Setting `alpha = 2 * rank` makes the model learn more aggressively. Never set alpha lower than rank — that scales down updates below 1.0. |
| `lora_dropout` | Dropout probability applied to LoRA activations during training. Regularizes to prevent overfitting. | Unsloth's code has a specific optimization path when set to `0`. Non-zero is recommended only if you observe overfitting. Recent research suggests it's unreliable for short training runs. |
| `bias` | Whether to train bias terms in targeted layers. `"none"` skips them. `"all"` or `"lora_only"` enables them. | `"none"` saves memory and trains faster with negligible quality impact. No reason to change this unless experimenting. |
| `use_gradient_checkpointing` | Same as in `from_pretrained`. Should match what was set at load time. | Set to `"unsloth"` in both places for consistency. |
| `random_state` | Seed for adapter weight initialization. Ensures reproducible runs. | Any integer works. `3407` is Unsloth's conventional default. Change if you want a different initialization. |
| `use_rslora` | Rank-Stabilized LoRA. Changes the scaling from `alpha/r` to `alpha/sqrt(r)`, which is theoretically more stable at higher ranks. | Only matters at higher ranks (64+). Leave False for most runs. |
| `loftq_config` | Initializes LoRA A/B matrices using SVD decomposition of the pretrained weights rather than random values. Can improve accuracy. | Causes a significant VRAM spike at initialization. Not needed for typical instruct fine-tuning. |
| `max_seq_length` | Must be passed again here — ensures RoPE positional embeddings are extended consistently with the value set during loading. | Must match the value used in `from_pretrained`. Mismatch causes incorrect positional encodings. |

### Alpha and Rank — Key Relationship

The effective update strength is `alpha / rank`. The valid rule from Unsloth's docs:

- `alpha = rank` → scale of 1.0, neutral/safe baseline
- `alpha = 2 * rank` → scale of 2.0, more aggressive learning, common heuristic
- `alpha < rank` → scale below 1.0, effectively weakens fine-tuning, avoid

---

## Method 3 — `SFTTrainer` + `SFTConfig`

`SFTTrainer` is from the TRL library and handles the supervised training loop. `SFTConfig` holds all training hyperparameters.

### Core `SFTTrainer` Arguments

| Parameter | What It Does | Watch Out For |
|---|---|---|
| `model` | The PEFT-wrapped model returned by `get_peft_model`. | Always pass the PEFT model, not the raw base model. |
| `train_dataset` | The HuggingFace dataset to train on. | Dataset must be tokenized or have a `dataset_text_field` pointing to a text column. For chat/instruct data, apply the model's chat template before passing in. |
| `eval_dataset` | Optional validation dataset. Required if using `eval_strategy`. | Split ~5% from your train set if you have no separate val set. Omit entirely for quick exploratory runs. |
| `tokenizer` | The model's tokenizer returned by `from_pretrained`. | Must match the model. For instruct models, the chat template is embedded in the tokenizer. |
| `data_collator` | Controls how batches are assembled. Use `DataCollatorForSeq2Seq` when training on completions only. | Required when using `train_on_responses_only`. Without it, the masking won't be applied correctly. |

### `SFTConfig` — Training Arguments

| Parameter | What It Does | Watch Out For |
|---|---|---|
| `max_seq_length` | Maximum token length per sample. Sequences longer than this are truncated. | Must match or be ≤ the value set in `from_pretrained`. |
| `per_device_train_batch_size` | Samples per GPU per training step. Primary driver of VRAM usage. | Increase only if VRAM allows. OOM most commonly comes from this. Lower to 1 first when debugging OOM. |
| `gradient_accumulation_steps` | Number of micro-steps before a weight update. Effective batch size = `batch_size × accumulation`. | Unsloth has fixed the standard HuggingFace bug where gradient accumulation and batch size were not truly equivalent. In Unsloth they are interchangeable. Prefer increasing this over batch size when VRAM is tight. |
| `num_train_epochs` | Full passes over the dataset. Use for production runs. | 1–3 epochs recommended. Beyond 3 risks overfitting on most instruct datasets. Monitor eval loss. |
| `max_steps` | Overrides epochs — hard stop after N optimizer steps. Use for exploratory runs. | Takes precedence over `num_train_epochs` if both are set. Remove it for full training runs. |
| `learning_rate` | Step size for weight updates. | Recommended range: `1e-4` to `2e-4` for QLoRA/LoRA instruct fine-tuning. Too high = unstable or diverging loss. Too low = underfitting or prevents learning entirely (not just slow). |
| `lr_scheduler_type` | How the learning rate changes over training. `cosine` decays smoothly to zero. `linear` decays linearly. | `cosine` is the standard choice for most fine-tuning runs. Leave as default for exploratory work. |
| `warmup_steps` / `warmup_ratio` | Ramps LR from 0 to target over N steps. `warmup_ratio` scales with total steps automatically. | Use `warmup_ratio=0.05` for full runs. Use `warmup_steps=5–10` for short exploratory runs. Prevents unstable early updates. |
| `optim` | Optimizer type. `adamw_8bit` stores optimizer states in 8-bit, saving ~75% optimizer VRAM vs standard AdamW. | `adamw_8bit` is the default for Unsloth. Requires BitsAndBytes. |
| `weight_decay` | Regularization — penalizes large weights to reduce overfitting. | Start at `0.01`. Values above `0.1` can harm learning. |
| `bf16` / `fp16` | Compute precision. `bf16=True` on Ampere+ (RTX 30xx/40xx). `fp16=True` on older hardware. | Do not set both True. The 4070 Ti Super supports `bf16`. |
| `logging_steps` | How often to log loss to the console. | Set to `1` for short runs to see every step. Set to `10–50` for long runs to reduce noise. |
| `save_strategy` / `save_steps` | When to checkpoint. `"steps"` saves every N steps. | Always use for full runs. Without checkpointing, a crash at step 900/1000 loses everything. |
| `save_total_limit` | Maximum number of checkpoints to keep on disk. Older ones are deleted. | Set to 2–3 to avoid filling disk. |
| `eval_strategy` / `eval_steps` | When to evaluate. Requires `eval_dataset`. | Set to match `save_steps` so you can compare best checkpoint against eval loss. |
| `load_best_model_at_end` | Loads the checkpoint with lowest eval loss after training completes. | Requires `eval_strategy` and `save_strategy` to be compatible. |
| `seed` | Global RNG seed for reproducibility. | Separate from `random_state` in LoRA config — both should be set. |
| `dataset_num_proc` | CPU workers for tokenization/preprocessing. | Set to 1 if you encounter multiprocessing errors. Increase to 4+ for faster preprocessing on large datasets. |
| `packing` | Packs multiple short samples into one sequence to maximize token utilization per batch. | Can cause label bleed between samples if not handled carefully. Unsloth has a 3x faster packing implementation. |

---

## Key Technique — `train_on_responses_only`

Masks out the user/instruction tokens so the model only computes loss on assistant responses. Per the QLoRA paper, this can improve accuracy meaningfully on multi-turn conversational data.

Applied after trainer construction using `from unsloth.chat_templates import train_on_responses_only`. Requires `DataCollatorForSeq2Seq` as the data collator. Instruction and response part markers must match the model's exact chat template tokens.

---

## Training Loss Interpretation

| Loss Range | Meaning |
|---|---|
| Dropping steadily | Learning is occurring |
| 0.5–1.0 at end | Typical healthy range for instruct fine-tuning |
| Below 0.2 | Likely overfitting — model is memorizing training data |
| Not dropping / flat | Underfitting — check LR, dataset format, and template |
| NaN / exploding | LR too high, precision mismatch, or data issue |

---

## Overfitting vs Underfitting — Signals and Fixes

**Overfitting** — training loss drops but eval loss rises or stagnates.

Fixes: reduce epochs, lower LR, increase `weight_decay`, increase `lora_dropout` to 0.05–0.1, increase effective batch size, expand dataset, or scale LoRA alpha down by 0.5x after training.

**Underfitting** — loss never drops meaningfully, outputs are generic.

Fixes: increase LR, increase epochs, increase rank `r`, decrease batch size to 1, verify dataset format matches the model's chat template exactly.

---

## Workflow Overview — Instruct Fine-tuning

1. Load an instruct model via `FastLanguageModel.from_pretrained` with `load_in_4bit=True`.
2. Apply LoRA adapters via `get_peft_model` targeting all 7 major linear layers.
3. Prepare dataset in the model's chat template format (ChatML, ShareGPT, etc.).
4. Run a short exploratory run with `max_steps=100` to confirm loss is dropping and no OOM occurs.
5. If the exploratory run is stable, switch to `num_train_epochs=1–3`, add `eval_dataset`, enable checkpointing, and run the full job.
6. Optionally wrap trainer with `train_on_responses_only` for multi-turn conversational data.
7. Save final adapter with `model.save_pretrained()`. Adapter is typically ~100MB.
8. For inference, load base model + adapter via `FastLanguageModel.from_pretrained` pointing to adapter path, then call `FastLanguageModel.for_inference(model)` to enable Unsloth's 2x faster inference path.

---

## Quick Diagnostics — OOM Recovery Order

1. `per_device_train_batch_size = 1`
2. `gpu_memory_utilization = 0.80`
3. Halve `max_seq_length`
4. `r = 16` if you were using a higher rank
5. Increase `gradient_accumulation_steps` to compensate for smaller batch
