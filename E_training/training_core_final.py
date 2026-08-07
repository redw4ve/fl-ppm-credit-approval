"""
Shared PyTorch training core for the E_05 baselines and the later E_06 federated runs.

The module owns everything that must stay identical across centralized, local and federated training:
- Seeding and device selection with a loud failure when the requested accelerator is unavailable
- The memory-mapped prefix tensor cache that materializes E_04 prefix samples once per data configuration
- Multitask LSTM with categorical embeddings, fused event inputs and three prediction heads
- The masked multitask loss over E_04 encoded remaining-time targets
- The outcome, next-activity and remaining-time metric blocks used by every evaluation

E_05 and E_06 import this module, so federation effects are never mixed with architecture or metric changes.
The module reads and writes no E_04 metadata itself; callers pass cache directories and cache contexts in.

REQUIRED FILES:
    none directly (the calling script provides cache directories and contexts)

CREATED FILES:
    <cache_dir>/metadata.json: cache version, tensor manifest and cache build hash
    <cache_dir>/*.npy: one memory-mapped array per prefix sample field
    <cache_dir>/prefix_index.parquet: compact prefix references for prediction exports
"""

# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict
import json
import math
import random
import shutil
import warnings
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
)
from torch import nn

# CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------
OUTCOME_CLASSES: tuple[int, int, int] = (0, 1, 2)       # 0:canceled | 1:denied/declined | 2:accepted/approved
PREFIX_TENSOR_CACHE_VERSION: int = 2                    # Cache version gate. Increase when the sample schema changes.
PREFIX_INDEX_COLUMNS: tuple[str, ...] = ("dataset_id", "client_id", "case_id", "split", "prefix_length", "label_pos")
PREFIX_INDEX_FILENAME: str = "prefix_index.parquet"
REMAINING_TIME_TRANSFORMS: tuple[str, str] = ("raw", "log")                     # RT Transform is the optional log step.
REMAINING_TIME_SCALINGS: tuple[str, str, str] = ("raw", "median", "zscore")     # RT scaling is the normalization.

# ----------------------------------------------------------------------------------------------------------------------
# 1. SHARED DATA OBJECTS

# CLASS: Carry the three head outputs of one forward pass as one immutable record.
@dataclass(frozen=True)
class ModelOutput:
    outcome_logits: torch.Tensor
    next_activity_logits: torch.Tensor
    remaining_time_scaled: torch.Tensor

# CLASS: Carry the weighted total, the 3 loss means and the 3 loss values per sample of one batch.
# The values per sample let an evaluation average the same losses over any subset of prefixes.
@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    outcome: torch.Tensor
    next_activity: torch.Tensor
    remaining_time: torch.Tensor
    outcome_values: torch.Tensor
    next_activity_values: torch.Tensor
    remaining_time_values: torch.Tensor

# CLASS: Carry the per-dataset next-activity class mask in the model head index space.
@dataclass(frozen=True)
class NextActivityMaskContext:
    dataset_code_by_id: dict[str, int]
    mask_by_dataset: torch.Tensor
    is_noop: bool

# CLASS: Extend the frozen E_04 batch schema with a training-local dataset code tensor.
@dataclass(frozen=True)
class DatasetAwareBatch:
    categorical_ids: torch.Tensor
    numerical: torch.Tensor
    offer_numerical: torch.Tensor
    offer_feature_mask: torch.Tensor
    padding_mask: torch.Tensor
    prefix_length: torch.Tensor
    outcome_label: torch.Tensor
    next_activity_label: torch.Tensor
    next_activity_mask: torch.Tensor
    remaining_time_label: torch.Tensor
    remaining_time_mask: torch.Tensor
    dataset_code: torch.Tensor

# CLASS: Mirror PyTorch load_state_dict diagnostics without importing a private symbol.
class LoadStateDictResult(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]

# ----------------------------------------------------------------------------------------------------------------------
# 2. DEVICE AND PREFIX CACHE HELPERS

# HELPER: Seed Python, numpy and torch, so every run with the same config is reproducible.
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Seed MPS separately when the Apple backend is active.
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)

# HELPER: Resolve the training device and fail when a requested accelerator is missing.
def select_device(device_name: str = "auto") -> torch.device:
    # Honor an explicit device request and reject unavailable accelerators.
    if device_name != "auto":
        normalized = device_name.lower()
        if normalized == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device was requested, but torch.backends.mps.is_available() is False.")
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False.")
        return torch.device(device_name)

    # Prefer accelerators (CUDA first, then MPS) and use CPU only as an automatic fallback.
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# HELPER: Caches: Keep one mapped tensor array per sample field so epochs read from the disk without re-encoding.
# Testing showed that on-the-fly encoding is very (very) slow, so the tensor arrays are cached on disk.
class CachedPrefixTensorDataset:
    # Validate the cache metadata and open the memory-mapped tensor arrays.
    def __init__(self, cache_dir: Path, expected_cache_hash: Optional[str] = None) -> None:
        # Reject caches without metadata, with a stale layout version or with a mismatched build hash.
        self.cache_dir = Path(cache_dir)
        self.metadata_path = self.cache_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"missing prefix tensor cache metadata: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if int(self.metadata.get("version", -1)) != PREFIX_TENSOR_CACHE_VERSION:
            raise ValueError(f"unsupported prefix tensor cache version: {self.metadata.get('version')}")
        if expected_cache_hash is not None and self.metadata.get("cache_build_hash") != expected_cache_hash:
            raise ValueError("prefix tensor cache metadata hash does not match the expected build hash")

        # Memory-map every tensor file so samples load lazily per index.
        self.length = int(self.metadata["length"])
        self.arrays = {
            key: np.load(self.cache_dir / info["filename"], mmap_mode="r")
            for key, info in self.metadata["tensors"].items()
        }
        self.static_padding_length = int(self.metadata.get("static_padding_length", 0))
        self.prefix_index = self._load_prefix_index()

    # Report the number of cached prefix samples.
    def __len__(self) -> int: return self.length

    # Load one cached prefix sample as torch tensors.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(np.array(array[index]))
            for key, array in self.arrays.items()
        }

    # Restore the compact prefix references so prediction exports keep case provenance.
    def _load_prefix_index(self) -> list[Any]:
        filename = self.metadata.get("prefix_index_file")
        if not filename: return []
        path = self.cache_dir / str(filename)
        if not path.exists(): raise FileNotFoundError(f"missing cached prefix index: {path}")
        frame = pd.read_parquet(path)
        return [
            SimpleNamespace(
                dataset_id=str(row["dataset_id"]), client_id=str(row["client_id"]), case_id=str(row["case_id"]),
                split=str(row["split"]), prefix_length=int(row["prefix_length"]), label_pos=int(row["label_pos"])
            )
            for row in frame.to_dict(orient="records")
        ]

# HELPER: Add a numeric dataset code to samples without changing the frozen E_04 source schema.
class DatasetIdEncodedDataset:
    # Store the wrapped prefix dataset and the code table used by the next-activity mask.
    def __init__(self, dataset: Any, dataset_code_by_id: dict[str, int]) -> None:
        self.dataset = dataset
        self.dataset_code_by_id = dict(dataset_code_by_id)
        self.prefix_index = getattr(dataset, "prefix_index", None)
        if self.prefix_index is None: raise ValueError("dataset-aware loading requires prefix_index rows")

    # Report the wrapped dataset length.
    def __len__(self) -> int: return len(self.dataset)

    # Return the original tensor sample plus one training-local dataset code.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, Any] = dict(self.dataset[index])
        dataset_id = str(self.prefix_index[index].dataset_id)
        if dataset_id not in self.dataset_code_by_id:
            raise ValueError(f"missing next-activity mask for dataset_id: {dataset_id}")
        sample["dataset_code"] = torch.tensor(self.dataset_code_by_id[dataset_id], dtype=torch.long)
        return sample

# HELPER: Convert one encoded torch sample into numpy arrays before storing it on disk (tensor cache).
def _tensor_sample_to_numpy(sample: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    # Move every tensor to CPU before writing it into the cache.
    return {key: value.detach().cpu().numpy() for key, value in sample.items()}

# HELPER: Accept both the E_04 dataclass rows and the SimpleNamespace rows restored from a cache.
def _prefix_index_row_to_dict(row: Any) -> dict[str, Any]:
    # Normalize the row object before selecting the stable prefix index columns.
    payload = asdict(row) if is_dataclass(row) else dict(vars(row))
    return {column: payload[column] for column in PREFIX_INDEX_COLUMNS}

# HELPER: Persist the prefix references next to the tensor arrays so cached datasets remain ready for export.
def _write_prefix_index(source_dataset: Any, tmp_dir: Path, metadata: dict[str, Any]) -> None:
    # Skip datasets that do not expose prefix provenance rows.
    prefix_index = getattr(source_dataset, "prefix_index", None)
    if prefix_index is None: return

    # Write the compact prefix provenance table with stable data types.
    frame = pd.DataFrame([_prefix_index_row_to_dict(row) for row in prefix_index], columns=list(PREFIX_INDEX_COLUMNS))
    for column in ("dataset_id", "client_id", "case_id", "split"): frame[column] = frame[column].astype(str)
    for column in ("prefix_length", "label_pos"): frame[column] = frame[column].astype("int64")
    frame.to_parquet(tmp_dir / PREFIX_INDEX_FILENAME, index=False)
    metadata["prefix_index_file"] = PREFIX_INDEX_FILENAME

# HELPER: Return the cached dataset or None, so callers decide between cache hit and rebuild.
def try_load_prefix_tensor_cache(cache_dir: Path,
                                 expected_cache_hash: Optional[str] = None) -> Optional[CachedPrefixTensorDataset]:
    # Treat every cache validation error as a cache miss for the caller.
    try: return CachedPrefixTensorDataset(cache_dir, expected_cache_hash=expected_cache_hash)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError, TypeError): return None

# HELPER: Store every prefix sample of one split as numpy arrays for reuse across runs on disk.
def build_prefix_tensor_cache(source_dataset: Any, cache_dir: Path, overwrite: bool = False,
    progress_iter: Optional[Any] = None, cache_context: Optional[dict[str, Any]] = None) -> CachedPrefixTensorDataset:

    # Reuse a readable cache with a matching build hash instead of re-encoding the entire dataset (slow).
    cache_dir = Path(cache_dir)
    expected_cache_hash = None if cache_context is None else str(cache_context.get("cache_build_hash", ""))
    if cache_dir.exists() and not overwrite:
        cached = try_load_prefix_tensor_cache(cache_dir, expected_cache_hash=expected_cache_hash)
        if cached is not None: return cached

    # Stop when a split would produce no training samples.
    if len(source_dataset) == 0: raise ValueError("cannot build a prefix tensor cache for an empty dataset")

    # Write into a temporary directory so incomplete cache builds are not reused.
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
    if tmp_dir.exists(): shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Use the first sample to size each tensor array, then record the tensor manifest in metadata on disk.
    first_sample = _tensor_sample_to_numpy(source_dataset[0])
    metadata: dict[str, Any] = {
        "version": PREFIX_TENSOR_CACHE_VERSION,
        "length": int(len(source_dataset)),
        "static_padding_length": int(getattr(source_dataset, "static_padding_length", 0)),
        "tensors": {},
    }
    if cache_context is not None: metadata.update(cache_context)
    arrays: dict[str, Any] = {}
    for key, value in first_sample.items():
        filename = f"{key}.npy"
        shape = (len(source_dataset), *value.shape)
        array = np.lib.format.open_memmap(tmp_dir / filename, mode="w+", dtype=value.dtype, shape=shape)
        array[0] = value
        arrays[key] = array
        metadata["tensors"][key] = {"filename": filename, "dtype": str(value.dtype), "shape": list(shape)}

    # Encode the remaining samples once. This encoding step to cache dominates the wall clock on the first run.
    index_iter: Any = range(1, len(source_dataset))
    if progress_iter is not None: index_iter = progress_iter(index_iter, total=max(len(source_dataset) - 1, 0))
    for index in index_iter:
        sample = _tensor_sample_to_numpy(source_dataset[index])
        for key, value in sample.items(): arrays[key][index] = value

    # Flush the arrays and promote the temporary directory to the final cache path.
    for array in arrays.values(): array.flush()
    _write_prefix_index(source_dataset, tmp_dir, metadata)
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    # Replace the old cache only after the new cache is complete.
    if cache_dir.exists(): shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    return CachedPrefixTensorDataset(cache_dir, expected_cache_hash=expected_cache_hash)

# HELPER: Load the warm cache when valid, otherwise build it from the source dataset.
def load_or_build_prefix_tensor_cache(source_dataset: Any, cache_dir: Path, overwrite: bool = False,
    progress_iter: Optional[Any] = None, cache_context: Optional[dict[str, Any]] = None) -> CachedPrefixTensorDataset:

    # Reuse a valid cache first so repeated E_05 runs avoid renewed tensor materialization.
    cache_dir = Path(cache_dir)
    expected_cache_hash = None if cache_context is None else str(cache_context.get("cache_build_hash", ""))
    if cache_dir.exists() and not overwrite:
        cached = try_load_prefix_tensor_cache(cache_dir, expected_cache_hash=expected_cache_hash)
        if cached is not None: return cached

    # Build the cache when no valid warm cache exists.
    return build_prefix_tensor_cache(
        source_dataset, cache_dir, overwrite=overwrite, progress_iter=progress_iter, cache_context=cache_context
    )

# ----------------------------------------------------------------------------------------------------------------------
# 3. MODEL HELPERS

# HELPER: Choose a bounded embedding width per categorical field: larger vocabularies get wider vectors up to 32.
def embedding_dim_for_vocab(vocab_size: int) -> int:
    # Reject empty vocabularies because embedding tables need at least one row.
    if vocab_size < 1: raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    return min(32, max(8, round(math.sqrt(vocab_size) * 4)))

# HELPER: Average only over masked entries, so padded or invalid samples contribute no gradient.
def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Convert the task mask into a loss-value dtype for weighted averaging.
    weights = mask.to(values.device, dtype=values.dtype)
    denom = weights.sum()

    # Preserve differentiability with a zero loss when no valid labels exist.
    if float(denom.detach().cpu()) == 0.0: return values.sum() * 0.0
    return (values * weights).sum() / denom

# HELPER: Assemble a dataset-aware batch from E_04 tensor samples plus a local dataset code.
def collate_dataset_aware_batch(samples: list[dict[str, torch.Tensor]]) -> DatasetAwareBatch:
    return DatasetAwareBatch(
        categorical_ids=torch.stack([sample["categorical_ids"] for sample in samples]),
        numerical=torch.stack([sample["numerical"] for sample in samples]),
        offer_numerical=torch.stack([sample["offer_numerical"] for sample in samples]),
        offer_feature_mask=torch.stack([sample["offer_feature_mask"] for sample in samples]),
        padding_mask=torch.stack([sample["padding_mask"] for sample in samples]),
        prefix_length=torch.stack([sample["prefix_length"] for sample in samples]),
        outcome_label=torch.stack([sample["outcome_label"] for sample in samples]),
        next_activity_label=torch.stack([sample["next_activity_label"] for sample in samples]),
        next_activity_mask=torch.stack([sample["next_activity_mask"] for sample in samples]),
        remaining_time_label=torch.stack([sample["remaining_time_label"] for sample in samples]),
        remaining_time_mask=torch.stack([sample["remaining_time_mask"] for sample in samples]),
        dataset_code=torch.stack([sample["dataset_code"] for sample in samples]),
    )

# HELPER: Build the per-dataset next-activity mask from each dataset's own observed target classes.
def build_next_activity_mask_context(presence_by_dataset: dict[str, np.ndarray],
                                     single_dataset_noop: bool = False) -> NextActivityMaskContext:

    # Single-dataset runs keep the whole softmax head visible for a byte-identical no-op.
    dataset_ids = sorted(str(dataset_id) for dataset_id in presence_by_dataset)
    if not dataset_ids: raise ValueError("next-activity mask requires at least one dataset")
    n_classes = int(np.asarray(next(iter(presence_by_dataset.values()))).shape[0])
    dataset_code_by_id = {dataset_id: index for index, dataset_id in enumerate(dataset_ids)}
    if single_dataset_noop:
        full = torch.ones((len(dataset_ids), n_classes), dtype=torch.bool)
        return NextActivityMaskContext(dataset_code_by_id=dataset_code_by_id, mask_by_dataset=full, is_noop=True)

    # Each dataset mask allows exactly the next-activity classes observed in that dataset's own data.
    masks = torch.zeros((len(dataset_ids), n_classes), dtype=torch.bool)
    for dataset_id in dataset_ids:
        presence = np.asarray(presence_by_dataset[dataset_id], dtype=bool)
        if presence.shape != (n_classes,):
            raise ValueError(f"next-activity presence for {dataset_id} must cover {n_classes} classes")
        masks[dataset_code_by_id[dataset_id]] = torch.from_numpy(presence)
        if not bool(masks[dataset_code_by_id[dataset_id]].any()):
            raise ValueError(f"next-activity mask for {dataset_id} contains no classes")
    return NextActivityMaskContext(dataset_code_by_id=dataset_code_by_id, mask_by_dataset=masks, is_noop=False)

# HELPER: Set logits outside each example's dataset activity set to minus infinity.
def mask_next_activity_logits(logits: torch.Tensor, dataset_code: torch.Tensor,
                              context: Optional[NextActivityMaskContext],
                              valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if context is None or context.is_noop: return logits
    if dataset_code.shape[0] != logits.shape[0]:
        raise ValueError("dataset_code length must match next-activity logits batch size")

    # Select one boolean class mask per example in the batch.
    codes = dataset_code.to(logits.device, dtype=torch.long)
    class_masks = context.mask_by_dataset.to(logits.device)[codes]

    # Rows without a valid next target keep full logits because their CE value is masked out later.
    if valid_mask is not None:
        valid = valid_mask.to(logits.device, dtype=torch.bool).view(-1, 1)
        class_masks = torch.where(valid, class_masks, torch.ones_like(class_masks))
    return logits.masked_fill(~class_masks, float("-inf"))

# HELPER: Invert softplus for RT bias initialization while staying stable near zero and near large values.
def _inverse_softplus(value: float) -> float:
    clipped = max(float(value), 1e-6)
    if clipped > 20.0: return clipped
    return float(math.log(math.expm1(clipped)))

# CLASS: Multitask LSTM with categorical embeddings, fused event inputs, shared trunk and three heads.
class MultitaskLSTM(nn.Module):
    # Build embeddings, the LSTM trunk and all three task heads.
    def __init__(
        self,
        categorical_vocab_sizes: dict[str, int],
        numerical_dim: int,
        offer_dim: int,
        next_activity_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.20,
        head_hidden_size: int = 64,
        outcome_classes: int = 3,
        remaining_time_softplus: bool = True,
        outcome_head_dropout: Optional[float] = None,
        lstm_cls: type[nn.Module] = nn.LSTM,
    ) -> None:
        super().__init__()
        # One embedding table per categorical field with [PAD] pinned to index 0.
        self.categorical_names = list(categorical_vocab_sizes)
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, embedding_dim_for_vocab(vocab_size), padding_idx=0)
                for name, vocab_size in categorical_vocab_sizes.items()
            }
        )

        # Fuse embeddings, numeric features, offer features and the offer mask into one event vector.
        embedding_width = sum(module.embedding_dim for module in self.embeddings.values())
        self.input_dim = int(embedding_width + numerical_dim + offer_dim + 1)
        self.hidden_size = int(hidden_size)
        self._uses_opacus_lstm_keys = getattr(lstm_cls, "__name__", "") == "DPLSTM"
        self.lstm = lstm_cls(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # Softplus keeps non-negative RT targets positive. A z-score target needs the linear head for negative values.
        # Softplus maps x to log(1 + exp(x)).
        self.remaining_time_softplus = bool(remaining_time_softplus)

        # The outcome head can receive its own dropout. It equals the global dropout by default.
        resolved_outcome_dropout = dropout if outcome_head_dropout is None else float(outcome_head_dropout)

        # Attach three task-specific prediction heads with two layers to the shared sequence state.
        self.outcome_head = self._build_head(hidden_size, head_hidden_size, outcome_classes, resolved_outcome_dropout)
        self.next_activity_head = self._build_head(hidden_size, head_hidden_size, next_activity_classes, dropout)
        self.remaining_time_head = self._build_head(hidden_size, head_hidden_size, 1, dropout)

    # Build one feed-forward prediction head with two layers. Use the same two-layer MLP shape for all three task heads.
    @staticmethod
    def _build_head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    # Concatenate all inputs per event into the fused LSTM input sequence.
    def _event_features(self, batch: Any) -> torch.Tensor:
        # Look up one embedding tensor per categorical sequence field.
        categorical_parts: list[torch.Tensor] = []
        for index, name in enumerate(self.categorical_names):
            categorical_parts.append(self.embeddings[name](batch.categorical_ids[:, :, index]))

        # Append numerical features, offer values and the offer-present mask.
        numeric_parts = [
            batch.numerical.float(),
            batch.offer_numerical.float(),
            batch.offer_feature_mask.float().unsqueeze(-1),
        ]
        return torch.cat(categorical_parts + numeric_parts, dim=-1)

    # Run one batch through the shared trunk and the three task heads.
    def forward(self, batch: Any) -> ModelOutput:
        # Encode the padded prefix sequence with the shared LSTM trunk.
        features = self._event_features(batch)
        outputs, _ = self.lstm(features)

        # Last state pooling matches traces cut at the outcome event, so the decision sits at the final valid timestep.
        # The padded tail cannot leak in because the LSTM is unidirectional.
        prefix_length = batch.prefix_length.to(outputs.device)
        last_indices = (prefix_length - 1).clamp(min=0)
        batch_indices = torch.arange(outputs.shape[0], device=outputs.device)
        sequence_state = outputs[batch_indices, last_indices]

        # Softplus keeps the scaled RT prediction non-negative for non-negative target combinations.
        # The linear head is used only for z-score targets, which are centered and therefore take negative values.
        remaining_time_value = self.remaining_time_head(sequence_state).squeeze(-1)
        if self.remaining_time_softplus: remaining_time_value = nn.functional.softplus(remaining_time_value)

        # Return all three task heads as one shared output record.
        return ModelOutput(
            outcome_logits=self.outcome_head(sequence_state),
            next_activity_logits=self.next_activity_head(sequence_state),
            remaining_time_scaled=remaining_time_value,
        )

    # Translate Opacus DPLSTM keys into nn.LSTM keys for canonical E_06 serialization.
    @staticmethod
    def _canonical_lstm_key(key: str) -> str:
        # Convert only DPLSTM recurrent keys and leave all heads and embeddings untouched.
        prefix = "lstm.l"
        if not key.startswith(prefix): return key
        parts = key.split(".")
        if len(parts) != 4: return key
        layer = parts[1][1:]
        direction = parts[2]
        kind = parts[3]
        if direction not in {"ih", "hh"} or kind not in {"weight", "bias"}: return key
        return f"lstm.{kind}_{direction}_l{layer}"

    # Translate canonical nn.LSTM keys back into Opacus DPLSTM keys before loading a DP model.
    @staticmethod
    def _opacus_lstm_key(key: str) -> str:
        # Convert only canonical recurrent keys and leave all heads and embeddings untouched.
        if not key.startswith("lstm."): return key
        name = key.removeprefix("lstm.")
        parts = name.split("_")
        if len(parts) != 3: return key
        kind, direction, layer_token = parts
        if kind not in {"weight", "bias"} or direction not in {"ih", "hh"} or not layer_token.startswith("l"):
            return key
        return f"lstm.{layer_token}.{direction}.{kind}"

    # Return a canonical state dict so nn.LSTM and DPLSTM trunks share one serialized key space.
    def state_dict(self, *args: Any, **kwargs: Any) -> OrderedDict[str, torch.Tensor]:
        state = super().state_dict(*args, **kwargs)
        if not self._uses_opacus_lstm_keys: return state
        translated = OrderedDict((self._canonical_lstm_key(key), value) for key, value in state.items())
        metadata = getattr(state, "_metadata", None)
        if metadata is not None: setattr(translated, "_metadata", metadata)
        return translated

    # Load canonical nn.LSTM keys into DPLSTM trunks by reversing the key translation.
    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        if self._uses_opacus_lstm_keys:
            # Validate against the canonical key set exposed by this wrapper.
            expected = set(self.state_dict().keys())
            received = set(state_dict.keys())
            missing = sorted(expected - received)
            unexpected = sorted(received - expected)
            if strict and (missing or unexpected):
                raise RuntimeError(f"Error(s) in loading state_dict: missing={missing}, unexpected={unexpected}")

            # Copy canonical tensors into the actual Opacus parameter keys without using PyTorch recursion.
            actual_state = nn.Module.state_dict(self)
            with torch.no_grad():
                for key, value in state_dict.items():
                    actual_key = self._opacus_lstm_key(key)
                    if actual_key not in actual_state: continue
                    target = actual_state[actual_key]
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
            return LoadStateDictResult(missing, unexpected)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

# HELPER: Build AdamW parameter groups for the shared trunk and the three task heads.
def build_multitask_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, outcome_lr_scale: float,
                              next_activity_lr_scale: float, remaining_time_lr_scale: float) -> torch.optim.Optimizer:

    # Define head prefixes so parameter names decide each learning rate group.
    head_prefixes = {
        "outcome": ("outcome_head.", float(outcome_lr_scale)),
        "next_activity": ("next_activity_head.", float(next_activity_lr_scale)),
        "remaining_time": ("remaining_time_head.", float(remaining_time_lr_scale)),
    }

    # Collect head parameters separately and keep every remaining parameter in the shared trunk group.
    head_param_groups: dict[str, list[nn.Parameter]] = {key: [] for key in head_prefixes}
    trunk_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        matched = next((key for key, (prefix, _) in head_prefixes.items() if name.startswith(prefix)), None)
        if matched is None: trunk_params.append(param)
        else: head_param_groups[matched].append(param)

    # Build one AdamW group for the trunk and one group per task head.
    param_group_specs: list[dict[str, Any]] = [
        {"params": trunk_params, "lr": float(learning_rate), "group_name": "trunk"}
    ]
    for key, (_, scale) in head_prefixes.items():
        param_group_specs.append(
            {"params": head_param_groups[key], "lr": float(learning_rate) * scale, "group_name": key}
        )

    # Return one optimizer so E_05 and E_06 share identical grouping semantics.
    return torch.optim.AdamW(param_group_specs, weight_decay=float(weight_decay))

# HELPER: Initialize the RT head bias at the scaled train median so epoch one starts near the baseline.
# Use softplus for non-negative RT targets and linear output for z-score targets, which can be negative.
def initialize_remaining_time_head_bias(model: MultitaskLSTM, remaining_time_repr: RemainingTimeRepr) -> float:

    # The RT head must end with a linear layer so its bias can be initialized directly.
    final_layer = model.remaining_time_head[-1]
    if not isinstance(final_layer, nn.Linear): raise TypeError("remaining_time_head must end with a linear layer")

    # Choose the bias value according to the head activation used for the target representation.
    scaled_median = float(remaining_time_repr.median_model_units)
    # Softplus head: Invert softplus so the activated output starts at the scaled train median.
    if remaining_time_repr.use_softplus: bias_value = _inverse_softplus(max(scaled_median, 1e-6))
    # Linear head: Use the scaled train median directly as the bias.
    else: bias_value = scaled_median

    # Apply the initialized bias without tracking gradients.
    with torch.no_grad(): final_layer.bias.fill_(bias_value)
    return scaled_median

# ----------------------------------------------------------------------------------------------------------------------
# 4. LOSS HELPERS

# HELPER: Move every batch field to the training device while preserving the batch dataclass type.
def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    batch_type = type(batch)
    return batch_type(**{field: getattr(batch, field).to(device) for field in batch.__dataclass_fields__})

# HELPER: Combine the three masked task losses into the weighted multitask objective.
# The RT representation stays in the signature as the caller's contract check because the loss reads the E_04
# encoded labels in model units directly and every caller must prove it holds the matching representation.
# noinspection PyUnusedLocal
def compute_multitask_loss(outputs: ModelOutput, batch: Any, outcome_loss: nn.Module, next_activity_loss: nn.Module,
    remaining_time_repr: RemainingTimeRepr, huber_beta: float = 1.0, outcome_weight: float = 1.0,
    next_activity_weight: float = 0.5, remaining_time_weight: float = 0.5,
    next_activity_mask_context: Optional[NextActivityMaskContext] = None) -> LossBreakdown:

    # Outcome and next activity use cross-entropy loss per sample. The RT uses Huber in model units.
    outcome_values = outcome_loss(outputs.outcome_logits, batch.outcome_label)
    next_logits = outputs.next_activity_logits
    if next_activity_mask_context is not None and not next_activity_mask_context.is_noop:
        if not hasattr(batch, "dataset_code"):
            raise ValueError("dataset-aware next-activity masking requires dataset_code")
        next_logits = mask_next_activity_logits(
            next_logits, batch.dataset_code, next_activity_mask_context, valid_mask=batch.next_activity_mask
        )
    next_values = next_activity_loss(next_logits, batch.next_activity_label)

    # Use the E_04 encoded RT labels directly in model units.
    remaining_target_model_units = batch.remaining_time_label.float()
    remaining_values = nn.functional.smooth_l1_loss(
        outputs.remaining_time_scaled, remaining_target_model_units, beta=huber_beta, reduction="none"
    )

    # Apply the E_04 masks so unknown next activities and final decision prefixes don't contribute to the gradient.
    outcome = outcome_values.mean()
    next_activity = _masked_mean(next_values, batch.next_activity_mask)
    remaining_time = _masked_mean(remaining_values, batch.remaining_time_mask)

    # Combine task losses with the locked multitask contribution weights.
    total = outcome_weight * outcome + next_activity_weight * next_activity + remaining_time_weight * remaining_time
    return LossBreakdown(
        total=total, outcome=outcome, next_activity=next_activity, remaining_time=remaining_time,
        outcome_values=outcome_values, next_activity_values=next_values, remaining_time_values=remaining_values,
    )

# HELPER: Combine the three task loss means into the weighted multitask total.
def weighted_total_loss(outcome: float, next_activity: float, remaining_time: float, outcome_weight: float,
                        next_activity_weight: float, remaining_time_weight: float) -> float:
    return float(outcome_weight * outcome + next_activity_weight * next_activity + remaining_time_weight * remaining_time)

# CLASS: Store the RT target representation used consistently by loss, metrics and head bias initialization.
@dataclass(frozen=True)
class RemainingTimeRepr:
    transform: str               # raw or log
    scaling: str                 # raw, median or zscore
    center: float                # subtracted in the transform space, 0 except for zscore
    scale: float                 # divisor in the transform space, 1 for raw scaling
    use_softplus: bool           # softplus head for non-negative targets, linear head for zscore
    median_model_units: float    # scaled train median, used for the head bias initialization

# HELPER: Invert E_04 model units back to raw seconds for metrics.
def remaining_time_model_units_to_seconds(model_units: torch.Tensor, repr_temp: RemainingTimeRepr) -> torch.Tensor:
    transformed = model_units * repr_temp.scale + repr_temp.center
    if repr_temp.transform == "log": return torch.expm1(transformed)
    return transformed

# ----------------------------------------------------------------------------------------------------------------------
# 5. METRIC HELPERS

# HELPER: Report RT MAE and RMSE in raw seconds with predictions clamped at zero.
def compute_remaining_time_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.detach().bool().cpu()

    # Return schema stable zero metrics when the split contains no valid targets.
    if int(valid.sum()) == 0: return {"mae": 0.0, "rmse": 0.0}

    # Clamp predictions at zero for reporting because negative durations are invalid.
    true = y_true.detach().cpu().float()[valid]
    pred = y_pred.detach().cpu().float().clamp(min=0.0)[valid]

    # Compute absolute (MAE) and squared-error (RMSE) metrics in raw seconds.
    errors = pred - true
    return {
        "mae": float(errors.abs().mean().item()),
        "rmse": float(torch.sqrt((errors ** 2).mean()).item()),
    }

# HELPER: Report macro-F1, weighted-F1, balanced accuracy, AUC, confusion matrix and rows per class.
def compute_outcome_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    # Convert logits to probabilities and raw argmax predictions for all outcome metrics.
    probabilities = _softmax_np(logits)
    predictions = probabilities.argmax(axis=1)

    # Suppress undefined-metric warnings for absent classes in small evaluation splits.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, per_class_f1, support = precision_recall_fscore_support(
            y_true, predictions, labels=list(OUTCOME_CLASSES), zero_division=0,
        )

        # Let AUC degrade to NaN instead of failing when a class is absent.
        try:
            macro_auc = float(roc_auc_score(
                y_true, probabilities, labels=list(OUTCOME_CLASSES), multi_class="ovr", average="macro")
            )
        except ValueError: macro_auc = float("nan")
        if not math.isfinite(macro_auc): macro_auc = float("nan")
        balanced_accuracy = float(balanced_accuracy_score(y_true, predictions))

    # Return headline aggregate metrics plus confusion matrix and analysis per class.
    return {
        "macro_f1": float(
            f1_score(y_true, predictions, labels=list(OUTCOME_CLASSES), average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, predictions, labels=list(OUTCOME_CLASSES), average="weighted", zero_division=0)),
        "balanced_accuracy": balanced_accuracy,
        "macro_auc_ovr": macro_auc,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(OUTCOME_CLASSES)).astype(int).tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(OUTCOME_CLASSES)
        },
    }

# HELPER: Report next activity top-1, top-3 and F1 metrics over the masked targets only.
def compute_next_activity_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    # Return schema-stable zero metrics when no masked next activity labels exist.
    if len(y_true) == 0: return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    # Compute top-1 and top-3 predictions from the masked next activity logits.
    predictions = logits.argmax(axis=1)
    top_k = min(3, logits.shape[1])
    top3 = np.argpartition(-logits, kth=top_k - 1, axis=1)[:, :top_k]

    # Restrict averaging to observed labels so tiny splits do not dilute macro-F1 with absent classes.
    labels = sorted(set(y_true.tolist()) | set(predictions.tolist()))
    return {
        "top1_accuracy": float((predictions == y_true).mean()),
        "top3_accuracy": float(np.array([label in row for label, row in zip(y_true, top3)]).mean()),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)),
    }

# HELPER: Numerically stable softmax for converting stored logits into probabilities.
def _softmax_np(logits: np.ndarray) -> np.ndarray:
    # Shift logits by their row maximum so exponentiation remains numerically stable.
    shifted = logits - logits.max(axis=1, keepdims=True)

    # Normalize exponentiated logits into class probabilities per row.
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────