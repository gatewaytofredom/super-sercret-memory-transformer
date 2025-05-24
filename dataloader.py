import torch
from torch.utils.data import IterableDataset, DataLoader
import os


import random
import time
from tqdm import tqdm


from collections import deque


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"  # Start of Sequence
EOS_TOKEN = "<eos>"  # End of Sequence
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class BPETokenizerWrapper:
    def __init__(self):
        self.tokenizer = None
        self.vocab_size = 0
        self.pad_idx = -1  # Default to invalid
        self.unk_idx = -1  # Default to invalid
        self.sos_idx = -1  # Default to invalid
        self.eos_idx = -1  # Default to invalid

    def _find_text_files_for_training(self, text_files_dir):
        all_txt_files = []
        if not os.path.isdir(text_files_dir):
            raise FileNotFoundError(f"Directory not found: {text_files_dir}")

        for dirpath, _, filenames in os.walk(text_files_dir):
            for filename in filenames:
                if filename.endswith(".txt"):
                    all_txt_files.append(os.path.join(dirpath, filename))

        if not all_txt_files:
            raise ValueError(
                f"No .txt files found recursively in {text_files_dir} for BPE training."
            )
        return all_txt_files

    def train(self, text_files_dir, vocab_size_target, min_frequency, model_save_path):
        print(f"Training BPE tokenizer from files in {text_files_dir}...")

        files_to_train_on = self._find_text_files_for_training(text_files_dir)
        print(f"Found {len(files_to_train_on)} files for BPE training.")

        self.tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size_target,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        # Define an iterator that reads files with robust encoding handling
        def _robust_file_content_iterator(
            file_paths_list, encoding="utf-8", errors="replace"
        ):
            processed_files_count = 0
            skipped_files_count = 0
            # Use tqdm here if you want progress for file reading itself
            # for file_path in tqdm(file_paths_list, desc="Reading files for BPE"):
            for file_path in file_paths_list:
                try:
                    with open(file_path, "r", encoding=encoding, errors=errors) as f:
                        content = f.read()
                        if (
                            content.strip()
                        ):  # Yield only if there's non-whitespace content
                            yield content
                            processed_files_count += 1
                        else:
                            # Optional: print warning for empty or whitespace-only files
                            # print(f"Warning: File {file_path} is empty or contains only whitespace. Skipping for BPE training.")
                            skipped_files_count += 1
                except Exception as e_file_iter:
                    print(
                        f"Warning: Error reading file {file_path} for BPE training: {e_file_iter}. Skipping."
                    )
                    skipped_files_count += 1

            effective_num_files = len(file_paths_list) - skipped_files_count
            print(
                f"BPE trainer will process content from {processed_files_count} files (skipped {skipped_files_count} files)."
            )
            if effective_num_files == 0 and len(file_paths_list) > 0:
                raise ValueError(
                    "No valid files could be read for BPE training. Please check file contents and encodings."
                )

        # Train the tokenizer using the iterator
        # The `length` argument helps the trainer show progress.
        # It should ideally be the number of items the iterator will yield.
        # If files are skipped, this might be an overestimate, but trainer should handle it.
        num_files_for_trainer = len(files_to_train_on)
        if num_files_for_trainer == 0:
            print("No files found to train BPE tokenizer. Skipping training.")
            # Ensure properties are updated even if no training happens, or handle error
            self._update_properties_from_hf_tokenizer()  # Might be None if tokenizer isn't initialized
            return

        print(f"Starting BPE training with {num_files_for_trainer} potential files...")
        self.tokenizer.train_from_iterator(
            _robust_file_content_iterator(files_to_train_on),
            trainer=trainer,
            length=num_files_for_trainer,  # provide the original count for progress bar estimation
        )

        print(
            f"BPE tokenizer training complete. Actual vocab size: {self.tokenizer.get_vocab_size()}"
        )

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.tokenizer.save(model_save_path)
        print(f"BPE tokenizer model saved to {model_save_path}")

        self._update_properties_from_hf_tokenizer()

    def load_model(self, model_load_path):
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(
                f"Tokenizer model file not found: {model_load_path}"
            )
        print(f"Loading BPE tokenizer model from {model_load_path}...")
        self.tokenizer = Tokenizer.from_file(model_load_path)
        self._update_properties_from_hf_tokenizer()
        print(f"BPE tokenizer loaded. Vocab size: {self.vocab_size}")

    def _update_properties_from_hf_tokenizer(self):
        if not self.tokenizer:
            # Initialize with defaults if tokenizer is None
            self.vocab_size = 0
            self.pad_idx = -1
            self.unk_idx = -1
            self.sos_idx = -1
            self.eos_idx = -1
            print(
                "Warning: Tokenizer is not initialized. Properties set to default/invalid."
            )
            return

        self.vocab_size = self.tokenizer.get_vocab_size()

        # Attempt to get IDs for special tokens, handle if they don't exist
        self.pad_idx = self.tokenizer.token_to_id(PAD_TOKEN)
        self.unk_idx = self.tokenizer.token_to_id(UNK_TOKEN)
        self.sos_idx = self.tokenizer.token_to_id(SOS_TOKEN)
        self.eos_idx = self.tokenizer.token_to_id(EOS_TOKEN)

        special_token_map = {
            "PAD": self.pad_idx,
            "UNK": self.unk_idx,
            "SOS": self.sos_idx,
            "EOS": self.eos_idx,
        }
        missing_tokens = [
            name for name, idx in special_token_map.items() if idx is None
        ]

        if missing_tokens:
            print(
                f"Warning: Could not find the following special tokens in BPE tokenizer: {', '.join(missing_tokens)}"
            )
            print(
                "This can happen if they were not part of SPECIAL_TOKENS during training or were pruned."
            )
            print(
                "Consider them critical for model operation. Defaulting missing ones to -1 (invalid)."
            )
            if self.pad_idx is None:
                self.pad_idx = -1
            if self.unk_idx is None:
                self.unk_idx = -1
            if self.sos_idx is None:
                self.sos_idx = -1
            if self.eos_idx is None:
                self.eos_idx = -1

        # Verify again after potential defaults
        if (
            any(
                idx == -1
                for idx in [self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx]
            )
            and not missing_tokens
        ):
            # This case should ideally not be hit if the above logic is sound and tokens were found.
            # It's more a safeguard.
            pass  # Already handled by missing_tokens logic setting them to -1

        if self.vocab_size > 0 and all(
            idx is not None and idx >= 0
            for name, idx in special_token_map.items()
            if name in SPECIAL_TOKENS
        ):  # check original expected special tokens
            print(
                f"Special token IDs: PAD={self.pad_idx}, UNK={self.unk_idx}, SOS={self.sos_idx}, EOS={self.eos_idx}"
            )
        else:
            print(
                f"Special token IDs (some might be missing or invalid): PAD={self.pad_idx}, UNK={self.unk_idx}, SOS={self.sos_idx}, EOS={self.eos_idx}"
            )

    def tokenize(self, text):
        if not self.tokenizer:
            raise ValueError("BPETokenizerWrapper: Tokenizer not trained or loaded.")
        return self.tokenizer.encode(text).ids

    def detokenize(self, token_ids, skip_special_tokens=True):
        if not self.tokenizer:
            raise ValueError("BPETokenizerWrapper: Tokenizer not trained or loaded.")

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Filter out any potential invalid token IDs (e.g., -1 if used for padding before tokenization)
        # However, HuggingFace decode should handle valid vocabulary IDs.
        # If using -1 as PAD_IDX, ensure it's not passed to HF decode unless it's a valid token.
        # Here, we assume token_ids are valid output from tokenizer or model.

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class StreamingTextIterableDataset(IterableDataset):
    def __init__(
        self,
        file_dir,
        tokenizer,  # Expects an instance of BPETokenizerWrapper (or similar API)
        max_seq_len,
        examples_buffer_size_per_worker=2048,
        text_chunk_size=1024 * 100,
        step_size=None,
        shuffle_source_files=True,
    ):
        super().__init__()
        self.file_dir = file_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples_buffer_size = examples_buffer_size_per_worker
        self.text_chunk_size = text_chunk_size
        self.shuffle_source_files = shuffle_source_files

        if step_size is None:
            self.step_size = max_seq_len // 2
        else:
            self.step_size = step_size
        if self.step_size <= 0:
            self.step_size = 1

        # Critical check for tokenizer readiness
        if (
            self.tokenizer.pad_idx < 0
            or self.tokenizer.sos_idx < 0
            or self.tokenizer.eos_idx < 0
        ):
            raise ValueError(
                "Tokenizer special tokens (PAD, SOS, EOS) are not properly initialized with valid non-negative IDs. "
                f"Current IDs: PAD={self.tokenizer.pad_idx}, SOS={self.tokenizer.sos_idx}, EOS={self.tokenizer.eos_idx}. "
                "Please ensure BPE tokenizer is trained and special tokens are correctly registered."
            )

        self.all_source_files = self._find_text_files()
        if not self.all_source_files:
            raise ValueError(f"No .txt files found recursively in {file_dir}")
        print(
            f"StreamingTextIterableDataset: Found {len(self.all_source_files)} source files."
        )

    def _find_text_files(self):
        all_files = []
        for dirpath, _, filenames in os.walk(self.file_dir):
            for filename in filenames:
                if filename.endswith(".txt"):
                    all_files.append(os.path.join(dirpath, filename))
        return all_files

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        files_for_this_worker = list(self.all_source_files)
        if self.shuffle_source_files:
            random.shuffle(files_for_this_worker)

        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_for_this_worker = [
                f
                for i, f in enumerate(files_for_this_worker)
                if i % num_workers == worker_id
            ]

        examples_yield_buffer = []

        for filepath in files_for_this_worker:
            current_file_token_buffer = deque()
            current_file_token_buffer.append(self.tokenizer.sos_idx)

            try:
                # Read file with UTF-8 encoding and replace errors for robustness during data loading
                # This is different from BPE training; here we are consuming text for model input.
                # If BPE training used 'replace', similar characters might appear.
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    while True:
                        text_chunk = f.read(self.text_chunk_size)
                        if not text_chunk:
                            current_file_token_buffer.append(self.tokenizer.eos_idx)
                            break

                        tokenized_chunk = self.tokenizer.tokenize(text_chunk)
                        current_file_token_buffer.extend(tokenized_chunk)

                        while len(current_file_token_buffer) >= self.max_seq_len + 1:
                            current_sequence_tokens = [
                                current_file_token_buffer[i]
                                for i in range(self.max_seq_len + 1)
                            ]
                            input_ids = torch.tensor(
                                current_sequence_tokens[:-1], dtype=torch.long
                            )
                            target_ids = torch.tensor(
                                current_sequence_tokens[1:], dtype=torch.long
                            )
                            examples_yield_buffer.append((input_ids, target_ids))

                            for _ in range(self.step_size):
                                if current_file_token_buffer:
                                    current_file_token_buffer.popleft()
                                else:
                                    break

                            if len(examples_yield_buffer) >= self.examples_buffer_size:
                                random.shuffle(examples_yield_buffer)
                                for ex_pair in examples_yield_buffer:
                                    yield ex_pair
                                examples_yield_buffer = []

                while len(current_file_token_buffer) >= self.max_seq_len + 1:
                    current_sequence_tokens = [
                        current_file_token_buffer[i]
                        for i in range(self.max_seq_len + 1)
                    ]
                    input_ids = torch.tensor(
                        current_sequence_tokens[:-1], dtype=torch.long
                    )
                    target_ids = torch.tensor(
                        current_sequence_tokens[1:], dtype=torch.long
                    )
                    examples_yield_buffer.append((input_ids, target_ids))
                    for _ in range(self.step_size):
                        if current_file_token_buffer:
                            current_file_token_buffer.popleft()
                        else:
                            break

                    if len(examples_yield_buffer) >= self.examples_buffer_size:
                        random.shuffle(examples_yield_buffer)
                        for ex_pair in examples_yield_buffer:
                            yield ex_pair
                        examples_yield_buffer = []

            except FileNotFoundError:
                name = f"W{worker_info.id}" if worker_info else "Main"
                print(f"{name}: File not found {filepath}. Skipping.")
            except Exception as e:
                name = f"W{worker_info.id}" if worker_info else "Main"
                print(f"{name}: Error processing file {filepath}: {e}")

        if examples_yield_buffer:
            random.shuffle(examples_yield_buffer)
            for ex_pair in examples_yield_buffer:
                yield ex_pair


# --- Collate Function (remains the same) ---
def _collate_fn_streaming(batch_of_examples):
    if not batch_of_examples:
        return None, None
    input_ids_list = []
    target_ids_list = []
    valid_example_found = False

    for item in batch_of_examples:
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], torch.Tensor)
            and isinstance(item[1], torch.Tensor)
            and item[0].ndim == 1
            and item[1].ndim == 1
            and len(item[0]) == len(item[1])  # Ensure lengths match
            and len(item[0]) > 0  # Ensure not empty
        ):
            input_ids_list.append(item[0])
            target_ids_list.append(item[1])
            valid_example_found = True
        else:
            item_info = f"type {type(item)}"
            if isinstance(item, tuple) and len(item) == 2:
                item_info += (
                    f", item[0] type {type(item[0])}, item[1] type {type(item[1])}"
                )
                if isinstance(item[0], torch.Tensor):
                    item_info += f", item[0] shape {item[0].shape}"
                if isinstance(item[1], torch.Tensor):
                    item_info += f", item[1] shape {item[1].shape}"
            print(
                f"Warning: Bad item in _collate_fn_streaming batch: {item_info}. Skipping."
            )

    if not valid_example_found:
        # print("Warning: No valid examples found in batch for collate_fn_streaming.") # Can be noisy
        return None, None

    try:
        input_ids_batch = torch.stack(input_ids_list)
        target_ids_batch = torch.stack(target_ids_list)
    except RuntimeError as e:  # Catch specific stacking errors
        print(f"Error during torch.stack in _collate_fn_streaming: {e}")
        # This usually happens if tensors have different lengths, despite previous checks
        for i, (tensor_in, tensor_tgt) in enumerate(
            zip(input_ids_list, target_ids_list)
        ):
            print(
                f"  Item {i}: Input shape: {tensor_in.shape}, Target shape: {tensor_tgt.shape}"
            )
        return None, None  # Propagate failure
    except Exception as e:
        print(f"Unexpected error during torch.stack in _collate_fn_streaming: {e}")
        return None, None

    return input_ids_batch, target_ids_batch


# --- create_dataloader (remains largely the same, takes the new tokenizer) ---
def create_dataloader(
    file_dir,
    tokenizer,
    max_seq_len,
    batch_size,
    num_workers=0,
    examples_buffer_size_per_worker=2048,
    text_chunk_size_chars=1024 * 100,
    step_size=None,
    shuffle_source_files=True,
    # Unused args
    buffer_size=None,  # Kept for signature compatibility if needed elsewhere
    cache_output_dir=None,
    force_rebuild_cache=None,
    dataset_shuffle_buffer_size=None,
    dataset_shuffle_examples_in_cache_file=None,
):
    if cache_output_dir is not None or force_rebuild_cache is not None:
        print(
            "Warning: `cache_output_dir` and `force_rebuild_cache` arguments are provided "
            "but this is an on-the-fly streaming dataloader and does not use disk caching."
        )

    if step_size is None:
        step_size = max_seq_len // 2
    if step_size <= 0:
        step_size = 1

    dataset = StreamingTextIterableDataset(
        file_dir=file_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        examples_buffer_size_per_worker=examples_buffer_size_per_worker,
        text_chunk_size=text_chunk_size_chars,
        step_size=step_size,
        shuffle_source_files=shuffle_source_files,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate_fn_streaming,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True if num_workers > 0 and torch.cuda.is_available() else False,
    )

    # Attach tokenizer properties directly to the dataloader for convenience
    dataloader.tokenizer = tokenizer
    dataloader.pad_idx = tokenizer.pad_idx
    dataloader.sos_idx = tokenizer.sos_idx
    dataloader.eos_idx = tokenizer.eos_idx
    dataloader.unk_idx = tokenizer.unk_idx
    dataloader.vocab_size = tokenizer.vocab_size  # Also useful

    print(f"On-the-fly streaming BPE DataLoader created for directory: '{file_dir}'")
    return dataloader
