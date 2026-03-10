from unsloth import FastLanguageModel
import torch.cuda as cuda
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

max_seq_length = 2048


def main():
    DATASET = load_dataset(
        "Trendyol/All-CVE-Chat-MultiTurn-1999-2025-Dataset", split="train"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.5-4B",
        max_seq_length=max_seq_length,
        load_in_16bit=False,
        load_in_4bit=True,
        full_finetuning=False,
        # gpu_memory_utilization=0.90,
        use_gradient_checkpointing="unsloth",
    )  # pyright: ignore[reportUnknownVariableType]

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # or 32 if u can support it sometime in the future.... :(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        # max_seq_length=max_seq_length, REMOVED DUE TO OOM
    )

    EOS_TOKEN = tokenizer.eos_token

    def formatting_func(examples):
        texts = []
        for system, user, assistant in zip(
            examples["System"], examples["User"], examples["Assistant"]
        ):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text + EOS_TOKEN)
        return {"text": texts}

    DATASET = DATASET.map(
        formatting_func, batched=True, remove_columns=DATASET.column_names
    )

    split = DATASET.train_test_split(test_size=0.02, seed=3407)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=max_seq_length,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            per_device_eval_batch_size=1,
            save_total_limit=3,
            load_best_model_at_end=False,
            output_dir="outputs_qwen35_full",
            optim="adamw_8bit",
            dataloader_pin_memory=False,
            seed=3407,
            dataset_num_proc=8,
            bf16=True,
            fp16=False,
        ),
    )

    cuda.empty_cache()

    trainer.train()

    model.save_pretrained("qwen35_cve_lora_final")
    tokenizer.save_pretrained("qwen35_cve_lora_final")


if __name__ == "__main__":
    main()
