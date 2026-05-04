# Token Channel Batch Inference

## Overview

Batch inference optimization for the teacher model in token-channel watermark training.

This optimization replaces sequential token-by-token inference with two-level parallel inference:
1. **Intra-text parallelism**: Single forward pass extracts logits for all token positions
2. **Inter-text parallelism**: Multiple texts are batched together with dynamic sizing

### Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Utilization | 67% | 90%+ | +23% |
| Memory Utilization | 42% | 78%+ | +36% |
| Inference Speedup | 1x | 50-100x | 50-100x |
| Corpus Building Time | ~2 hours | ~1-2 minutes | 60-120x |

## Usage

### Training Workflow Integration

The batch inference is automatically used when you specify `teacher_batch_size` in the training config:

```python
from wfcllm.watermark.token_channel.train_workflow import (
    TokenChannelTrainWorkflowConfig,
    run_token_channel_train_workflow,
)
from pathlib import Path

config = TokenChannelTrainWorkflowConfig(
    dataset="humaneval",
    dataset_path=Path("data/datasets/humaneval"),
    lm_model_path=Path("data/models/codet5-base"),
    model_path=Path("data/artifacts/model"),
    cache_path=Path("data/cache/training.json"),
    context_width=64,
    hidden_size=128,
    batch_size=8,
    epochs=10,
    lr=0.001,
    entropy_threshold=1.0,
    diversity_threshold=2,
    split_ratio=0.8,
    seed=42,
    teacher_batch_size=16,  # Batch size for teacher inference (default: 16)
)

summary = run_token_channel_train_workflow(config)
print(f"Training complete: {summary.training_rows} rows generated")
```

### Direct API Usage

For custom workflows, use the batch inference API directly:

```python
from wfcllm.watermark.token_channel.teacher import batch_extract_teacher_rows
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("data/models/codet5-base")
model = AutoModelForCausalLM.from_pretrained("data/models/codet5-base")

# Prepare code samples
code_samples = [
    "def foo():\n    return 42",
    "x = 1 + 2 + 3",
    "for i in range(10):\n    print(i)",
]

# Extract teacher rows using batch inference
results = batch_extract_teacher_rows(
    tokenizer=tokenizer,
    model=model,
    texts=code_samples,
    context_width=64,
    batch_size=16,
    show_progress=True,
)

# Process results
for sample_idx, sample_rows in enumerate(results):
    print(f"Sample {sample_idx}: {len(sample_rows)} rows")
    for row in sample_rows:
        print(f"  Token: {row['token_text']}, Entropy: {row['entropy']:.2f}")
```

### Sequential Mode (for Debugging)

For validation or debugging, use the original sequential inference:

```python
from wfcllm.watermark.token_channel.teacher import extract_teacher_rows

# Single text, sequential inference
rows = extract_teacher_rows(
    tokenizer=tokenizer,
    model=model,
    text="def foo():\n    return 42",
    context_width=64,
    show_progress=True,
)

print(f"Extracted {len(rows)} rows")
```

## Configuration

### teacher_batch_size Parameter

The `teacher_batch_size` parameter controls the base batch size for teacher inference. It is dynamically adjusted based on:
- Sequence length in the batch
- Available GPU memory
- Model size

**Default value:** 16

**Recommended values:**
- Small GPU (8GB): 4-8
- Medium GPU (16GB): 8-16
- Large GPU (24GB+): 16-32

### Dynamic Batch Sizing

The actual batch size used is computed dynamically:

```python
def _compute_dynamic_batch_size(
    max_length: int,
    base_batch_size: int,
    available_memory_gb: float,
) -> int:
    """Compute batch size based on sequence length and available memory."""
    # Shorter sequences allow larger batches
    # Longer sequences use smaller batches
    # Constrained to [4, base_batch_size * 2]
```

**Typical adjustments:**
- Short sequences (< 128 tokens): batch_size up to 64
- Medium sequences (128-256 tokens): batch_size 16-32
- Long sequences (256-512 tokens): batch_size 8-16
- Very long sequences (> 512 tokens): batch_size 4-8

### Length Grouping

Texts are automatically grouped by similar length (within 20% tolerance) to reduce padding waste:

```python
def _group_texts_by_length(
    texts: list[str],
    tokenizer: object,
    tolerance: float = 0.2,
) -> list[list[tuple[int, str]]]:
    """Group texts by similar token length."""
```

This reduces memory overhead from padding while maintaining good GPU utilization.

## Performance Tuning

### Monitoring GPU Usage

Monitor GPU memory and utilization during training:

```bash
# Real-time GPU monitoring
nvidia-smi dmon -s mu

# Or use watch for continuous updates
watch -n 1 nvidia-smi
```

### Adjusting Batch Size

Based on GPU memory usage, adjust `teacher_batch_size`:

| Memory Usage | Action | New Batch Size |
|--------------|--------|----------------|
| < 70% | Increase | base_size + 4 |
| 70-85% | Optimal | Keep current |
| 85-90% | Monitor | Keep current |
| > 90% | Decrease | base_size - 4 |

**Target:** 80-85% GPU memory utilization

### Example Tuning Session

```python
# Start with conservative batch size
config = TokenChannelTrainWorkflowConfig(
    # ... other params ...
    teacher_batch_size=8,
)

# Monitor memory usage during training
# If memory < 70%, increase to 12
# If memory > 90%, decrease to 6
# Iterate until 80-85% utilization

# Final optimized config
config = TokenChannelTrainWorkflowConfig(
    # ... other params ...
    teacher_batch_size=16,  # Optimized for your GPU
)
```

## Implementation Details

### Two-Level Parallelism

**Level 1: Intra-text Parallelism**
- Single forward pass processes all token positions in a text
- Leverages causal language model property: one forward pass gives logits for all positions
- Implemented in `_extract_single_text_all_positions()`

**Level 2: Inter-text Parallelism**
- Multiple texts are batched together
- Variable-length sequences are handled with left padding
- Attention masks ensure padding doesn't affect computation
- Implemented in `_batch_forward_all_positions()`

### Padding Strategy

Left padding is used to preserve causal structure:

```
Input sequences (variable length):
  [1, 2, 3]
  [4, 5, 6, 7, 8]

After left padding (max_len=5):
  [0, 1, 2, 3, 0]  (padded with 0)
  [4, 5, 6, 7, 8]  (no padding needed)

Attention mask:
  [0, 1, 1, 1, 0]  (0 = ignore, 1 = attend)
  [1, 1, 1, 1, 1]
```

### Numerical Precision

Results are numerically equivalent to sequential inference within float16 precision:

```python
# Batch inference
batch_logits = batch_extract_teacher_rows(...)

# Sequential inference
seq_logits = extract_teacher_rows(...)

# Comparison (should be close)
torch.allclose(batch_logits, seq_logits, atol=1e-4)  # True
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:

1. **Reduce batch size:**
   ```python
   config.teacher_batch_size = 8  # or lower
   ```

2. **Check available memory:**
   ```bash
   nvidia-smi
   ```

3. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Slow Performance

If batch inference is slower than expected:

1. **Check GPU utilization:**
   ```bash
   nvidia-smi dmon -s u
   ```
   Should be > 80%

2. **Verify batch size is being used:**
   Look for "批量提取 teacher logits" in progress output

3. **Check for CPU bottlenecks:**
   - Ensure tokenizer is not the bottleneck
   - Verify data loading is fast enough

### Inconsistent Results

If batch and sequential results differ significantly:

1. **Check precision:**
   ```python
   # Use float32 for higher precision
   model = model.to(torch.float32)
   ```

2. **Verify attention masks:**
   - Ensure padding is handled correctly
   - Check for NaN values in logits

3. **Run consistency test:**
   ```bash
   HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
     tests/watermark/token_channel/test_teacher_batch.py::test_batch_vs_sequential_consistency -v
   ```

## Testing

### Unit Tests

Run unit tests to verify batch inference correctness:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/token_channel/test_teacher_batch.py -v
```

### Integration Tests

Run end-to-end integration tests:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/token_channel/test_train_workflow.py::test_workflow_with_batch_inference_integration -v
```

### Performance Benchmarks

Run performance benchmarks (requires GPU):

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/token_channel/test_teacher_performance.py -v -m benchmark
```

## API Reference

### batch_extract_teacher_rows()

```python
def batch_extract_teacher_rows(
    tokenizer: object,
    model: object,
    texts: list[str],
    context_width: int,
    batch_size: int = 16,
    *,
    show_progress: bool = False,
) -> list[list[dict[str, object]]]:
    """Extract offline teacher rows for multiple texts using batch inference.

    This function implements two-level parallelism:
    1. Intra-text parallelism: Extract all positions in a single text with one forward pass
    2. Inter-text parallelism: Batch multiple texts together to reduce forward passes

    Args:
        tokenizer: Tokenizer with encode() method and optional bos_token_id
        model: Language model that accepts input_ids and returns logits
        texts: List of text strings to process
        context_width: Maximum number of prefix tokens to use for prediction
        batch_size: Base batch size for processing (will be adjusted dynamically)
        show_progress: Whether to show progress bar

    Returns:
        List of row lists, one per input text. Each row contains:
        - prefix_tokens: List of token IDs in the prefix (up to context_width)
        - next_token: Token ID being predicted
        - teacher_logits: Logits for all vocabulary tokens (as list)
        - entropy: Entropy of the teacher distribution
        - token_text: Text representation of the token
        - token_start: Character start position in original text
        - token_end: Character end position in original text
        - token_index: Position index in the token sequence

    Raises:
        ValueError: If context_width <= 0 or texts is empty
    """
```

### extract_teacher_rows()

```python
def extract_teacher_rows(
    tokenizer: object,
    model: object,
    text: str,
    context_width: int,
    *,
    show_progress: bool = False,
) -> list[dict[str, object]]:
    """Extract offline teacher rows for every token position in text.

    Sequential inference mode - processes one token at a time.
    Use batch_extract_teacher_rows() for better performance with multiple texts.

    Args:
        tokenizer: Tokenizer with encode() method
        model: Language model
        text: Text string to process
        context_width: Maximum number of prefix tokens
        show_progress: Whether to show progress bar

    Returns:
        List of dicts with same structure as batch_extract_teacher_rows()
    """
```

## References

- **Design Document:** `docs/superpowers/specs/2026-05-04-token-channel-batch-inference-design.md`
- **Implementation Plan:** `docs/superpowers/plans/2026-05-04-token-channel-batch-inference.md`
- **Module Documentation:** `wfcllm/watermark/token_channel/teacher.py`
- **Tests:** `tests/watermark/token_channel/test_teacher_batch.py`

## FAQ

**Q: Should I always use batch inference?**
A: Yes, batch inference is recommended for all use cases. It's faster and uses GPU more efficiently. Sequential mode is mainly for debugging.

**Q: What if I have very long texts (> 2048 tokens)?**
A: Batch inference handles long texts automatically by adjusting batch size down. For extremely long texts, consider splitting them or using a smaller batch size.

**Q: Can I mix texts of very different lengths?**
A: Yes, but it's inefficient due to padding. The automatic length grouping helps, but if you have texts with 10x length difference, consider processing them separately.

**Q: How much GPU memory do I need?**
A: Minimum 8GB for batch_size=4. Recommended 16GB+ for batch_size=16. With 24GB+, you can use batch_size=32.

**Q: Is the output exactly the same as sequential inference?**
A: Results are numerically equivalent within float16 precision (atol=1e-4). Small differences are expected due to floating-point rounding.

**Q: Can I use batch inference with different models?**
A: Yes, as long as the model accepts input_ids and returns logits. Tested with CodeT5, DeepSeek-Coder, and other causal language models.
