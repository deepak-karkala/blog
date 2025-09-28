# SFT and DPO Finetuning Plan

**Fine-tuning: Considerations**

1.  **Dataset Format & Preprocessing:**
    *   Is your dataset currently in a specific file format (e.g., JSONL, CSV, Hugging Face `Dataset` object)?
    *   For SFT, the prompt will be combined with the chosen reply. For Llama 3.1 Instruct, it's crucial to format this combination using its specific chat template. The `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` model is an instruct-tuned model. Do you have a plan for applying this chat template to your `(prompt, chosen_reply)` pairs? Unsloth and Hugging Face `tokenizers` have utilities for this (e.g., `tokenizer.apply_chat_template`).
    *   Similarly, for DPO, the `prompt`, `chosen`, and `rejected` fields will likely also need to be formatted using the chat template to represent full conversational turns if they aren't already.
2.  **Evaluation during Fine-tuning:** The Colab notebooks show some basic inference testing. Do you want to incorporate more systematic evaluation (e.g., perplexity on a validation set, or generating sample outputs at checkpoints) within the fine-tuning scripts, or will the LLM-as-Judge evaluation be purely post-fine-tuning?
3.  **RunPod GPU Choice:** Do you have a specific GPU in mind on RunPod (e.g., A100, H100, RTX 4090)? The 8B model, especially with Unsloth's 4-bit QLoRA, is manageable on GPUs with ~16-24GB VRAM, but more VRAM allows larger batch sizes.
4.  **Saving Model Artifacts:**
    *   The primary goal is to deploy with vLLM. For this, you'll typically want the merged model (base + LoRA adapters) saved in Hugging Face format. Unsloth provides methods for this.
    *   Do you also want to save the LoRA adapters separately at each stage (SFT, DPO)? This can be useful for experimentation.
    *   Do you intend to push the final model (or adapters) to the Hugging Face Hub?


**Fine-tuning Workflow: Llama 3.1 8B for Chandler Bing using Unsloth (SFT + DPO)**

**Phase 0: Environment Setup and Data Preparation on RunPod**

*   **Step 0.1: Launch RunPod Instance**
    *   Choose a GPU instance. For an 8B model with Unsloth QLoRA, consider options like:
        *   NVIDIA RTX A4000 (16GB) - might be tight, small batches
        *   NVIDIA RTX A5000 (24GB) - good balance
        *   NVIDIA RTX 4090 (24GB) - excellent consumer option
        *   NVIDIA A100 (40GB/80GB) - premium, allows larger batches/longer sequences
    *   Select a PyTorch-compatible Docker image or set up a custom environment. Unsloth often provides recommended base images or setup scripts.
    *   Ensure sufficient disk space for the model, dataset, and saved checkpoints.
*   **Step 0.2: Connect and Basic Setup**
    *   SSH into your RunPod instance.
    *   Create a project directory (e.g., `mkdir chandler_finetune && cd chandler_finetune`).
*   **Step 0.3: Install Dependencies**
    *   Install Unsloth. For the latest Llama 3.1 support, use the command recommended in their notebooks (likely includes specific bitsandbytes):
        ```bash
        pip install "unsloth[colab-newest-bitsandbytes]" -U  # Or specific CUDA version if needed
        # Or from the Llama 3.1 notebook:
        # pip install "unsloth[colab-newest-bitsandbytes-old-cuda]" --upgrade
        ```
    *   Install other necessary libraries:
        ```bash
        pip install "transformers>=4.41.1" "trl>=0.8.6" "accelerate>=0.30.1" "datasets>=2.19.0" "peft>=0.11.1"
        ```
        (Adjust versions based on Unsloth compatibility if specified).
*   **Step 0.4: Upload Your Dataset**
    *   Transfer your prepared dataset (e.g., a JSONL file where each line is `{"prompt": "...", "chosen": "...", "rejected": "..."}`) to the RunPod instance (e.g., using `scp` or RunPod's volume features). Let's assume it's `dataset.jsonl`.
*   **Step 0.5: Dataset Loading and Preprocessing Script (Python)**
    *   Create a Python script (`prepare_data.py`) or do this within your main fine-tuning script.
    *   Load the dataset using Hugging Face `datasets.load_dataset('json', data_files={'train': 'dataset.jsonl'})`.
    *   **Crucially, define a formatting function that applies the Llama 3.1 chat template to your data.**
        *   The Llama 3.1 Instruct template looks like:
            ```
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

            {{ user_prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            {{ assistant_response }}<|eot_id|>
            ```
        *   For SFT: Your "prompt" column in the dataset becomes `{{ user_prompt }}`, and "chosen" becomes `{{ assistant_response }}`. You'll need a system prompt for Chandler Bing.
        *   Example formatting function for SFT:
            ```python
            # In your fine-tuning script
            from datasets import load_dataset

            # Global system prompt for Chandler
            CHANDLER_SYSTEM_PROMPT = "You are Chandler Bing from the TV show Friends. You are known for your sarcasm, witty jokes, and often self-deprecating humor. You sometimes end your sentences with a questioning intonation. Respond to the user as Chandler Bing."

            def formatting_sft_prompt_func(example):
                # 'prompt' from dataset is user's turn, 'chosen' is Chandler's reply
                # This assumes tokenizer.apply_chat_template will be used by SFTTrainer or you apply it here.
                # SFTTrainer usually expects a single text field.
                # If SFTTrainer's formatting_func is used, it might expect a list of messages.
                # Let's assume SFTTrainer will handle the template if provided with structured input or a formatting_func
                # Simpler: Create one text column where the template is already applied.
                messages = [
                    {"role": "system", "content": CHANDLER_SYSTEM_PROMPT},
                    {"role": "user", "content": example['prompt']},
                    {"role": "assistant", "content": example['chosen']}
                ]
                # Later, the tokenizer will apply the chat template to this list of messages.
                # For SFTTrainer directly, it might be easier to provide one formatted string if not using `dataset_kwargs={"chat_template": ...}`
                # For now, let's prepare for SFTTrainer by providing a column that can be easily formatted by it.
                # The Alpaca notebook creates a single text field.
                # We will use tokenizer.apply_chat_template within the SFTTrainer or by mapping the dataset.
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

            # Later, load tokenizer and then map the dataset
            # tokenizer = AutoTokenizer.from_pretrained(...)
            # dataset = dataset.map(formatting_sft_prompt_func) # Will need tokenizer initialized
            ```
        *   For DPO: `prompt`, `chosen`, and `rejected` fields will each be formatted as a complete conversation up to that point, where the final turn is the chosen/rejected assistant response.
            ```python
            def formatting_dpo_prompt_func(example):
                # This prepares data for DPOTrainer which expects 'prompt', 'chosen', 'rejected' fields
                # where each is a list of messages or a fully formatted string.
                prompt_messages = [
                    {"role": "system", "content": CHANDLER_SYSTEM_PROMPT},
                    {"role": "user", "content": example['prompt']}
                ]
                chosen_messages = prompt_messages + [{"role": "assistant", "content": example['chosen']}]
                rejected_messages = prompt_messages + [{"role": "assistant", "content": example['rejected']}]

                return {
                    "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True), # or just user turn
                    "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False),
                    "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
                }
            # dataset_dpo = dataset.map(formatting_dpo_prompt_func)
            ```
        *   Split into train/validation if desired (e.g., 90/10 split).

**Phase 1: Supervised Fine-Tuning (SFT) with Unsloth**

*   **Step 1.1: Python Script for SFT (`sft_finetune.py`)**
*   **Step 1.2: Import Libraries**
    ```python
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, AutoTokenizer
    from trl import SFTTrainer
    from datasets import load_dataset
    ```
*   **Step 1.3: Load Model and Tokenizer**
    ```python
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length = 2048 # Or choose based on your data

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None, # Unsloth handles dtype
        load_in_4bit = True,
    )
    # Set padding token if not already set for Llama 3.1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ```
*   **Step 1.4: Add LoRA Adapters (QLoRA)**
    ```python
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # LoRA rank
        lora_alpha = 32, # LoRA alpha
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Recommended by Unsloth
        random_state = 3407,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"], # Common for Llama
    )
    ```
*   **Step 1.5: Load and Prepare SFT Dataset**
    ```python
    # Assuming your dataset.jsonl has 'prompt' and 'chosen' fields
    CHANDLER_SYSTEM_PROMPT = "You are Chandler Bing..." # Define as above

    def format_sft_data(example):
        messages = [
            {"role": "system", "content": CHANDLER_SYSTEM_PROMPT},
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['chosen']}
        ]
        # SFTTrainer expects a 'text' field by default if not using chatml/packing
        # This will be tokenized by the trainer.
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
    dataset = dataset.map(format_sft_data, batched=False) # Apply formatting
    # Optional: train_test_split if you want a validation set for SFT
    ```
*   **Step 1.6: Configure `TrainingArguments`**
    ```python
    sft_training_args = TrainingArguments(
        output_dir = "./sft_output",
        per_device_train_batch_size = 2, # Adjust based on VRAM
        gradient_accumulation_steps = 4, # Effective batch size = 2 * 4 = 8
        warmup_steps = 10,
        num_train_epochs = 1, # Or 2-3 epochs for SFT
        learning_rate = 2e-4, # Common for LoRA
        fp16 = not torch.cuda.is_bf16_supported(), # Use bf16 if available
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # Unsloth recommends adamw_bnb_8bit or adamw_8bit
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_strategy = "epoch", # Or "steps" with save_steps
        # report_to="wandb", # Optional: if using Weights & Biases
    )
    ```
*   **Step 1.7: Initialize `SFTTrainer`**
    ```python
    sft_trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text", # Matches the output of format_sft_data
        max_seq_length = max_seq_length,
        args = sft_training_args,
        # packing = True, # Consider if your sequences are short for efficiency
    )
    ```
*   **Step 1.8: Start SFT Training**
    ```python
    sft_trainer.train()
    ```
*   **Step 1.9: Save SFT Adapters**
    ```python
    sft_adapter_path = "./sft_output/sft_adapters"
    sft_trainer.save_model(sft_adapter_path) # Saves LoRA adapters
    # tokenizer.save_pretrained(sft_adapter_path) # Not strictly needed if base tokenizer unchanged
    print(f"SFT Adapters saved to {sft_adapter_path}")
    ```
*   **Step 1.10: (Optional) Qualitative Test SFT Model**
    *   Load base model + SFT adapters.
    *   Use `UnslothLlamaForCausalLM.from_pretrained` with `peft_model_id=sft_adapter_path`.
    *   Generate a few responses to see if Chandler's style is emerging.

**Phase 2: Direct Preference Optimization (DPO) with Unsloth**

*   **Step 2.1: Python Script for DPO (`dpo_finetune.py`)**
*   **Step 2.2: Import Libraries**
    ```python
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, AutoTokenizer
    from trl import DPOTrainer
    from datasets import load_dataset
    from peft import PeftModel # To load SFT adapters
    ```
*   **Step 2.3: Load SFT-Tuned Model**
    *   Load the base model first, then apply the SFT adapters.
    ```python
    base_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    sft_adapter_path = "./sft_output/sft_adapters" # Path from SFT stage
    max_seq_length = 2048

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply SFT adapters
    # Unsloth models are already PeftModels after get_peft_model.
    # To load adapters trained on an Unsloth model:
    print(f"Loading SFT adapters from: {sft_adapter_path}")
    model = FastLanguageModel.from_pretrained(
        model_name = sft_adapter_path, # Unsloth can load adapters directly if saved correctly
        base_model_name_or_path = base_model_name, # Specify base if needed
        # Or use PeftModel.from_pretrained if model above is the raw base model
        # model = PeftModel.from_pretrained(model, sft_adapter_path)
    )
    # No need to call get_peft_model again if adapters are already loaded.
    # If DPO needs new adapters on top, then yes. But usually DPO refines existing ones.
    # The Zephyr DPO notebook re-applies get_peft_model for DPO.
    # Let's follow the pattern of re-applying LoRA for the DPO stage, which is common.
    # This means DPO will train its own set of adapters based on the SFT-modified weights.
    model = FastLanguageModel.get_peft_model(
        model, # This is now the SFT-tuned model
        r = 16, # Can be same or different from SFT
        lora_alpha = 32,
        lora_dropout = 0.05, # Or 0 for DPO
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    )
    ```
*   **Step 2.4: Load and Prepare DPO Dataset**
    ```python
    CHANDLER_SYSTEM_PROMPT = "You are Chandler Bing..." # Define as above

    def format_dpo_data(example):
        # DPOTrainer expects 'prompt', 'chosen', 'rejected' fields
        # Each should be a full string representation of the conversation turn
        # including the system prompt and user prompt, then the assistant's chosen/rejected response.
        # Based on Llama-3.1 chat template.

        # Prompt for DPO is typically just the user + system part
        prompt_messages = [
            {"role": "system", "content": CHANDLER_SYSTEM_PROMPT},
            {"role": "user", "content": example['prompt']}
        ]
        # Chosen and Rejected include the assistant's response
        chosen_messages = prompt_messages + [{"role": "assistant", "content": example['chosen']}]
        rejected_messages = prompt_messages + [{"role": "assistant", "content": example['rejected']}]

        # The DPOTrainer will tokenize these.
        return {
            "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True), # TRL DPO expects prompt to not have assistant response
            "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False),
            "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
        }

    dpo_dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
    dpo_dataset = dpo_dataset.map(format_dpo_data, batched=False)
    ```
*   **Step 2.5: Configure `TrainingArguments` for DPO**
    ```python
    dpo_training_args = TrainingArguments(
        output_dir = "./dpo_output",
        per_device_train_batch_size = 1, # DPO is more memory intensive
        gradient_accumulation_steps = 4, # Effective batch size = 1 * 4 = 4
        warmup_steps = 5,
        num_train_epochs = 1, # DPO usually needs fewer epochs
        learning_rate = 5e-5, # Or lower, e.g., 1e-5, for DPO
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        seed = 3407,
        save_strategy = "epoch",
        # report_to="wandb", # Optional
    )
    ```
*   **Step 2.6: Initialize `DPOTrainer`**
    ```python
    dpo_trainer = DPOTrainer(
        model = model, # SFT-tuned model with new LoRA adapters for DPO
        ref_model = None, # TRL handles this for LoRA if PEFT model is passed
        args = dpo_training_args,
        beta = 0.1, # DPO hyperparameter
        train_dataset = dpo_dataset,
        tokenizer = tokenizer,
        max_length = max_seq_length,
        max_prompt_length = max_seq_length // 2, # Ensure prompt isn't too long
        # peft_config can be passed here if not applied to model already
    )
    ```
*   **Step 2.7: Start DPO Training**
    ```python
    dpo_trainer.train()
    ```
*   **Step 2.8: Save DPO Adapters/Model**
    ```python
    dpo_adapter_path = "./dpo_output/dpo_adapters"
    dpo_trainer.save_model(dpo_adapter_path) # Saves the DPO LoRA adapters
    # tokenizer.save_pretrained(dpo_adapter_path)
    print(f"DPO Adapters saved to {dpo_adapter_path}")
    ```

**Phase 3: Model Merging and Saving for Inference (vLLM)**

*   **Step 3.1: Python Script for Merging (`merge_model.py`)**
*   **Step 3.2: Load DPO-Tuned Model and Merge**
    ```python
    from unsloth import FastLanguageModel

    base_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    dpo_adapter_path = "./dpo_output/dpo_adapters" # Path from DPO stage
    output_merged_path = "./merged_chandler_llama3_8b"

    # Load the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name,
        # No need for 4bit loading if merging to 16bit
    )
    # Load and merge DPO adapters
    model = FastLanguageModel.from_pretrained(
        model_name = dpo_adapter_path, # Load DPO adapters onto the base model
        base_model_name_or_path = base_model_name,
    )
    # For vLLM, you typically want the full precision or a uniformly quantized model.
    # Unsloth's save_pretrained_merged handles this.
    # If you loaded the base model in 4-bit and want to merge into 16-bit:
    # model.merge_and_unload() # This dequantizes and merges.

    print(f"Saving merged model to {output_merged_path}")
    # Save in a format vLLM can use (e.g., 16-bit or original precision)
    # If you loaded base in 4bit and want to save merged in 16bit:
    model.save_pretrained_merged(output_merged_path, tokenizer, save_method="merged_16bit")
    # If you want to save in 4bit (e.g., for GGUF or other 4bit inference)
    # model.save_pretrained_merged(output_merged_path, tokenizer, save_method="merged_4bit")
    # For vLLM, merged_16bit (or full precision if base wasn't 4bit) is usually best.

    # Alternative if you want to save in GGUF format for Ollama/Llama.cpp as in Unsloth docs
    # model.save_pretrained_gguf(f"{output_merged_path}/gguf_quantized_model", tokenizer, quantization_method = "q4_k_m")
    ```
    *   **Note:** For vLLM, saving as `merged_16bit` or `merged_bf16bit` is standard. vLLM can also load some quantized formats like AWQ/GPTQ, but starting with a standard float16/bfloat16 merged model is a good baseline.

*   **Step 3.3: Verify Merged Model (Optional)**
    *   Load the merged model from `output_merged_path` using `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()`.
    *   Perform inference to ensure it works as expected.

**Next Steps after Fine-tuning:**

1.  **Transfer Merged Model:** Copy the `output_merged_path` directory from RunPod to where you will set up your vLLM inference server (this might be another RunPod instance or a different environment).
2.  **Set up vLLM:** Deploy the merged model using vLLM according to vLLM's documentation.
3.  **LLM-as-Judge Evaluation:** Use your evaluation framework to compare the SFT model, DPO model, and the original generic LLM.

