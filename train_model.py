# train_neural_memory_transformer.py (renamed from train_standard.py)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import time
import dataclasses
import json

from dataloader import (
    BPETokenizerWrapper,
    create_dataloader,
)
from neural_memory_transformer_model import (
    NeuralMemoryTransformer,
    NeuralMemoryTransformerConfig,
)

# --- Configuration ---
DATA_DIR = "smalldata"
OUTPUT_DIR = "output_titans_inspired_transformer"  # Changed output directory
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not any(File.endswith(".txt") for File in os.listdir(DATA_DIR)):
    print(f"Creating dummy data in {DATA_DIR}...")
    # ... (dummy data creation can remain the same) ...
    dummy_text_1 = (
        "This is a neural memory transformer language model training script.\n"
    )
    dummy_text_1 += "It uses a BPE tokenizer and a PyTorch DataLoader for efficient data handling.\n"
    with open(os.path.join(DATA_DIR, "doc1_nm.txt"), "w") as f:
        f.write(
            (
                (
                    dummy_text_1
                    + "The model learns to predict the next token and uses memory. " * 5
                    + "\n"
                )
                * 15
            )
        )
    dummy_text_2 = "Autoregressive models with memory generate text. This script focuses on that.\n"
    dummy_text_2 += "Modern architectures often use pre-layer normalization and advanced activations."
    with open(os.path.join(DATA_DIR, "doc2_nm.txt"), "w") as f:
        f.write(
            (
                (
                    dummy_text_2
                    + "Training involves minimizing cross-entropy loss and updating memory. "
                    * 5
                    + "\n"
                )
                * 15
            )
        )
    print("Dummy data created.")


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 3
BATCH_SIZE = 16  # Reduced for potentially more memory intensive LMM updates
LEARNING_RATE = 5e-5  # Main model LR
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0

BPE_TARGET_VOCAB_SIZE = 3000
BPE_MIN_FREQUENCY = 2
TOKENIZER_MODEL_FILENAME = "titans_bpe_tokenizer.json"
TOKENIZER_MODEL_PATH = os.path.join(OUTPUT_DIR, TOKENIZER_MODEL_FILENAME)
FORCE_REBUILD_TOKENIZER = False  # Set to True for first run

MODEL_D_MODEL = 512  # Smaller for quicker test
MODEL_N_LAYERS = 5
MODEL_N_HEADS = 8
MODEL_FFN_DIM = MODEL_D_MODEL * 4
MODEL_DROPOUT_P = 0.1

# Neural Memory (LMM) specific config
MODEL_MEMORY_DIM = MODEL_D_MODEL  # LMM internal dimension
MODEL_LMM_LAYERS = 2
MODEL_LMM_LEARNING_RATE = 0.005  # θ_t for LMM updates
MODEL_LMM_MOMENTUM_DECAY = 0.9  # η_t for LMM momentum
MODEL_LMM_WEIGHT_DECAY = 0.001  # α_t for LMM weight decay
MODEL_LMM_GRADIENT_CLIP = 0.5
MODEL_LMM_UPDATE_LOSS_THRESHOLD = 0.1  # Update LMM if its MSE > this
MODEL_UPDATE_LMM_AT_TEST_TIME = True  # Enable test-time LMM learning

DATALOADER_MAX_SEQ_LEN = 128
LOG_INTERVAL_BATCHES = 250  # Log more frequently with smaller batch size
GENERATION_LENGTH = 128
GENERATION_PROMPT = ""  # "<sos>" will be used if empty
GENERATION_TEMPERATURE = 0.8


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# generate_text function remains largely the same, but model.eval() is important
# The LMM update logic is now self-contained within the model and respects model.training state
# or the update_lmm_at_test_time flag.
def generate_text(
    model_instance: NeuralMemoryTransformer,
    tokenizer_instance: BPETokenizerWrapper,
    device_instance: torch.device,
    text_prompt: str,
    max_gen_len: int,
    temperature: float = 0.7,
):
    model_instance.eval()  # Critical: sets self.training = False
    # LMM updates will only happen if config.update_lmm_at_test_time is True

    prompt_token_ids = tokenizer_instance.tokenize(text_prompt)
    current_sos_idx = tokenizer_instance.sos_idx
    current_eos_idx = tokenizer_instance.eos_idx

    if not prompt_token_ids or (
        len(prompt_token_ids) > 0 and prompt_token_ids[0] != current_sos_idx
    ):
        current_ids = [current_sos_idx] + prompt_token_ids
    elif not prompt_token_ids:
        current_ids = [current_sos_idx]
    else:
        current_ids = prompt_token_ids

    generated_sequence_ids = list(current_ids)
    max_model_seq_len = model_instance.config.max_seq_len

    with torch.no_grad():  # Outer no_grad for generation, LMM internal updates handle their own logic
        for _ in range(max_gen_len):
            input_sequence_for_model = generated_sequence_ids
            if len(input_sequence_for_model) > max_model_seq_len:
                input_sequence_for_model = input_sequence_for_model[-max_model_seq_len:]

            current_input_token_tensor = torch.tensor(
                [input_sequence_for_model], dtype=torch.long, device=device_instance
            )

            # Padding mask (all False for generation as input is not padded)
            padding_mask = torch.zeros_like(
                current_input_token_tensor, dtype=torch.bool, device=device_instance
            )

            logits_output_sequence = model_instance(
                current_input_token_tensor, padding_mask=padding_mask
            )
            next_token_logits_output = logits_output_sequence[0, -1, :]

            if temperature <= 0:
                predicted_next_token_id = torch.argmax(next_token_logits_output).item()
            else:
                scaled_logits = next_token_logits_output / temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)
                predicted_next_token_id = torch.multinomial(
                    probabilities, num_samples=1
                ).item()

            if predicted_next_token_id == current_eos_idx:
                break
            generated_sequence_ids.append(predicted_next_token_id)

    # model_instance.train() # Not needed here, will be set by training loop
    return tokenizer_instance.detokenize(
        generated_sequence_ids, skip_special_tokens=True
    )


def training_loop_main(
    model_to_train: NeuralMemoryTransformer,
    data_loader_train,
    optimizer_instance: optim.Optimizer,
    loss_criterion: nn.Module,
    device_to_use: torch.device,
    num_train_epochs: int,
    log_every_n_batches: int,
    grad_clip_norm_val: float,  # Renamed to avoid conflict
    gen_text_len: int,
    gen_text_prompt: str,
    gen_temperature: float,
):
    model_to_train.train()  # Ensure model is in training mode
    tokenizer_instance = data_loader_train.dataset.tokenizer

    for epoch_num in range(1, num_train_epochs + 1):
        print(f"\n--- Starting Epoch {epoch_num}/{num_train_epochs} ---")
        epoch_total_loss, epoch_total_batches = 0, 0
        interval_loss_sum, interval_batches_count = 0, 0
        epoch_start_time = time.time()

        for batch_num, batch_data in enumerate(data_loader_train, start=1):
            if batch_data is None or batch_data[0] is None:
                continue
            batch_input_ids, batch_target_ids = batch_data
            if batch_input_ids is None or batch_target_ids is None:
                continue

            batch_input_ids = batch_input_ids.to(device_to_use)
            batch_target_ids = batch_target_ids.to(device_to_use)

            optimizer_instance.zero_grad()

            padding_mask = batch_input_ids == tokenizer_instance.pad_idx
            batch_logits = model_to_train(batch_input_ids, padding_mask=padding_mask)

            current_ce_loss = loss_criterion(
                batch_logits.view(-1, model_to_train.config.vocab_size),
                batch_target_ids.view(-1),
            )
            # The LMM updates happen inside model_to_train forward pass automatically.
            # No separate loss for LMM needs to be added to main loss.
            total_loss = current_ce_loss
            total_loss.backward()  # Backprop main CE loss
            torch.nn.utils.clip_grad_norm_(
                model_to_train.parameters(), max_norm=grad_clip_norm_val
            )
            optimizer_instance.step()

            loss_value = total_loss.item()
            epoch_total_loss += loss_value
            interval_loss_sum += loss_value
            interval_batches_count += 1
            epoch_total_batches += 1

            if batch_num % log_every_n_batches == 0:
                avg_interval_loss = (
                    interval_loss_sum / interval_batches_count
                    if interval_batches_count > 0
                    else 0
                )
                elapsed = time.time() - epoch_start_time
                print(
                    f"Epoch {epoch_num} | Batch {batch_num} | Interval Avg Loss: {avg_interval_loss:.4f} | Elapsed: {elapsed:.2f}s"
                )

                if hasattr(model_to_train, "get_memory_stats"):
                    try:
                        memory_stats = model_to_train.get_memory_stats()
                        print(f"Memory Stats: {memory_stats}")
                    except Exception as e_mem_stats:
                        print(f"Could not retrieve memory stats: {e_mem_stats}")

                if gen_text_len > 0:
                    print(
                        f"\n--- Generating sample text (Epoch {epoch_num}, Batch {batch_num}) ---"
                    )
                    current_prompt = (
                        gen_text_prompt
                        if gen_text_prompt
                        else tokenizer_instance.detokenize(
                            [tokenizer_instance.sos_idx], skip_special_tokens=False
                        )
                    )

                    generated_text_sample = generate_text(
                        model_instance=model_to_train,
                        tokenizer_instance=tokenizer_instance,
                        device_instance=device_to_use,
                        text_prompt=current_prompt,
                        max_gen_len=gen_text_len,
                        temperature=gen_temperature,
                    )
                    print(
                        f"Sample from prompt '{current_prompt[:50]}...': '{generated_text_sample}'\n"
                    )
                    model_to_train.train()  # Ensure back to train mode after generation
                interval_loss_sum, interval_batches_count = 0, 0

        avg_epoch_loss = (
            epoch_total_loss / epoch_total_batches if epoch_total_batches > 0 else 0
        )
        print(
            f"--- Epoch {epoch_num} Summary --- \nAverage Loss: {avg_epoch_loss:.4f}\nTime taken: {time.time() - epoch_start_time:.2f}s"
        )

    print("\n--- Training Loop Finished ---")


if __name__ == "__main__":
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print("Initializing Tokenizer...")
    tokenizer = BPETokenizerWrapper()
    # ... (tokenizer training/loading logic remains the same) ...
    if FORCE_REBUILD_TOKENIZER or not os.path.exists(TOKENIZER_MODEL_PATH):
        print(
            f"Training BPE tokenizer from files in {DATA_DIR}, will save to {TOKENIZER_MODEL_PATH}"
        )
        os.makedirs(os.path.dirname(TOKENIZER_MODEL_PATH), exist_ok=True)
        if FORCE_REBUILD_TOKENIZER and os.path.exists(TOKENIZER_MODEL_PATH):
            print(
                f"Force retraining: Removing existing tokenizer model at {TOKENIZER_MODEL_PATH}"
            )
            os.remove(TOKENIZER_MODEL_PATH)
        try:
            tokenizer.train(
                text_files_dir=DATA_DIR,
                vocab_size_target=BPE_TARGET_VOCAB_SIZE,
                min_frequency=BPE_MIN_FREQUENCY,
                model_save_path=TOKENIZER_MODEL_PATH,
            )
            print(f"Tokenizer trained and saved to {TOKENIZER_MODEL_PATH}")
        except ValueError as e:
            print(f"CRITICAL: Tokenizer training failed: {e}")
            exit(1)
    else:
        try:
            tokenizer.load_model(TOKENIZER_MODEL_PATH)
            print(f"Tokenizer loaded from {TOKENIZER_MODEL_PATH}")
        except FileNotFoundError:
            print(
                f"CRITICAL: Tokenizer model not found at {TOKENIZER_MODEL_PATH}. Set FORCE_REBUILD_TOKENIZER=True."
            )
            exit(1)
    if (
        tokenizer.vocab_size == 0
        or tokenizer.pad_idx < 0
        or tokenizer.sos_idx < 0
        or tokenizer.eos_idx < 0
    ):
        print("CRITICAL: Tokenizer not properly initialized.")
        exit(1)
    print(f"Tokenizer ready. Vocab size: {tokenizer.vocab_size}")

    print("Creating DataLoader...")
    train_dataloader = create_dataloader(
        file_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_seq_len=DATALOADER_MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle_source_files=True,
    )
    # ... (dataloader check logic remains the same) ...
    if train_dataloader is None:
        exit(1)
    try:
        first_batch = next(iter(train_dataloader))
        if first_batch is None or first_batch[0] is None:
            print("DataLoader produced an empty first batch.")
        else:
            print(
                f"DataLoader created. First batch input shape: {first_batch[0].shape}"
            )
    except StopIteration:
        print("DataLoader is empty.")
        exit(1)
    except Exception as e:
        print(f"Error when testing dataloader: {e}")
        exit(1)

    print("Initializing Model...")
    model_config = NeuralMemoryTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_idx,
        max_seq_len=DATALOADER_MAX_SEQ_LEN,
        d_model=MODEL_D_MODEL,
        n_layers=MODEL_N_LAYERS,
        n_heads=MODEL_N_HEADS,
        ffn_dim=MODEL_FFN_DIM,
        dropout=MODEL_DROPOUT_P,
        memory_dim=MODEL_MEMORY_DIM,
        lmm_layers=MODEL_LMM_LAYERS,
        lmm_learning_rate=MODEL_LMM_LEARNING_RATE,
        lmm_momentum_decay=MODEL_LMM_MOMENTUM_DECAY,
        lmm_weight_decay=MODEL_LMM_WEIGHT_DECAY,
        lmm_gradient_clip=MODEL_LMM_GRADIENT_CLIP,
        lmm_update_loss_threshold=MODEL_LMM_UPDATE_LOSS_THRESHOLD,
        update_lmm_at_test_time=MODEL_UPDATE_LMM_AT_TEST_TIME,
    )
    model = NeuralMemoryTransformer(model_config).to(DEVICE)
    print(
        f"Model initialized on {DEVICE} with config: {dataclasses.asdict(model.config)}"
    )
    print(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    print("Optimizer and Loss function initialized.")

    print("\n--- Starting Training ---")
    training_loop_main(
        model_to_train=model,
        data_loader_train=train_dataloader,
        optimizer_instance=optimizer,
        loss_criterion=criterion,
        device_to_use=DEVICE,
        num_train_epochs=NUM_EPOCHS,
        log_every_n_batches=LOG_INTERVAL_BATCHES,
        grad_clip_norm_val=GRAD_CLIP_NORM,
        gen_text_len=GENERATION_LENGTH,
        gen_text_prompt=GENERATION_PROMPT,
        gen_temperature=GENERATION_TEMPERATURE,
    )

    model_save_path = os.path.join(OUTPUT_DIR, "titans_inspired_transformer_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    config_save_path = os.path.join(
        OUTPUT_DIR, "titans_inspired_transformer_config.json"
    )
    with open(config_save_path, "w") as f:
        json.dump(dataclasses.asdict(model.config), f, indent=4)
    print(f"Model config saved to {config_save_path}")

    print("\n--- Generating Text Post-Training ---")
    final_prompt = (
        GENERATION_PROMPT
        if GENERATION_PROMPT
        else tokenizer.detokenize([tokenizer.sos_idx], skip_special_tokens=False)
    )
    print(f"Prompt for final generation: '{final_prompt}'")
    generated_text_output = generate_text(
        model_instance=model,
        tokenizer_instance=tokenizer,
        device_instance=DEVICE,
        text_prompt=final_prompt,
        max_gen_len=GENERATION_LENGTH * 2,  # Generate longer text
        temperature=GENERATION_TEMPERATURE,
    )
    print("\nGenerated Text (Post-Training):")
    print(generated_text_output)
    print("\n--- Script Execution Complete ---")
