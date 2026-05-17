## Quick-note
This is a configuration for QLoRA fine-tuning of Qwen 3.5, but with an important caveat.

Due to the nature of CVE data, the arbitrary prompt limit of 2048 won't work, plus the rank being 16 is also too low (22 million trainable parameters approx). This was ran on a single RTX 4070ti 16 GB VRAM and after about 20k iters the loss plateaued. To increase your results, you need more VRAM, and to increase the prompt length + matrix rank.
