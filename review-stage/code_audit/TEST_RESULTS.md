# Targeted Test Results

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Post-Repair Test Results

Updated: 2026-06-08T12:31:30Z

After user-approved repairs, the targeted suites were rerun on the remote server in `/root/autodl-tmp/WFCLLM` through `/root/miniconda3/bin/conda run -n WFCLLM` with:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

### Latest Remote Summary

- PASS: `tests/watermark/` -> 501 passed, 29 warnings.
- PASS: `tests/extract/` -> 131 passed, 1 warning.
- PASS: `tests/common/ tests/datasets/ tests/lang/ tests/evaluation/ tests/integration/test_cli_resolution.py tests/integration/test_orchestration.py` -> 228 passed, 4 warnings.
- PASS: `tests/encoder/ tests/ablation/` -> 86 passed, 14 warnings.
- PASS: `python -m compileall wfcllm run.py scripts tools`.

### Latest Remote Commands

```bash
cd /root/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/ -v --tb=short
```

Result: 501 passed, 29 warnings.

```bash
cd /root/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/extract/ -v --tb=short
```

Result: 131 passed, 1 warning.

```bash
cd /root/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/common/ tests/datasets/ tests/lang/ tests/evaluation/ tests/integration/test_cli_resolution.py tests/integration/test_orchestration.py -v --tb=short
```

Result: 228 passed, 4 warnings.

```bash
cd /root/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM pytest tests/encoder/ tests/ablation/ -v --tb=short
```

Result: 86 passed, 14 warnings.

```bash
cd /root/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 /root/miniconda3/bin/conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Result: passed.

### Latest Local Pre-Sync Checks

- Original four token-channel short-block failures: 4 passed.
- Adjacent watermark target set: 21 passed.
- Local `tests/watermark/`: 501 passed, 29 warnings.
- Local `python -m compileall wfcllm run.py scripts tools`: passed.
- Local phase-order focused tests after stale test update: 3 passed.

### Latest Local Pre-Commit Checks

```bash
cd /home/monglitay/PycharmProjects/WFCLLM
git diff --check
```

Result: passed.

```bash
cd /home/monglitay/PycharmProjects/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ tests/watermark/ tests/integration/test_orchestration.py -v --tb=short
```

Result: 667 passed, 30 warnings.

```bash
cd /home/monglitay/PycharmProjects/WFCLLM
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Result: passed.

### Warnings

- Token-channel model tests emit the expected PyTorch nested-tensor warning for `norm_first=True`.
- Some evaluation collection warnings are caused by dataclasses named `TestCase`/`TestExecutor`; they did not fail tests.
- Full `pytest tests/` was still not run; targeted suites now cover the audit-relevant areas.

## Environment

- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`.
- Commands used `/root/miniconda3/bin/conda run -n WFCLLM pytest ...` with a 600 second timeout per group.

## Summary

- PASS: `common` rc=0 elapsed=3.661s log=`review-stage/code_audit/test_logs/common.log`
- PASS: `datasets` rc=0 elapsed=1.675s log=`review-stage/code_audit/test_logs/datasets.log`
- PASS: `lang` rc=0 elapsed=1.63s log=`review-stage/code_audit/test_logs/lang.log`
- FAIL: `extract` rc=1 elapsed=4.904s log=`review-stage/code_audit/test_logs/extract.log`
- FAIL: `watermark` rc=1 elapsed=7.132s log=`review-stage/code_audit/test_logs/watermark.log`
- FAIL: `watermark_token_channel` rc=1 elapsed=5.038s log=`review-stage/code_audit/test_logs/watermark_token_channel.log`
- PASS: `evaluation` rc=0 elapsed=5.274s log=`review-stage/code_audit/test_logs/evaluation.log`
- PASS: `integration_cli_resolution` rc=0 elapsed=0.917s log=`review-stage/code_audit/test_logs/integration_cli_resolution.log`
- FAIL: `integration_orchestration` rc=1 elapsed=2.887s log=`review-stage/code_audit/test_logs/integration_orchestration.log`

## Exact Commands

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/common/ -v --tb=short
```
Result: rc=0, elapsed=3.661s, log `review-stage/code_audit/test_logs/common.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/datasets/ -v --tb=short
```
Result: rc=0, elapsed=1.675s, log `review-stage/code_audit/test_logs/datasets.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/lang/ -v --tb=short
```
Result: rc=0, elapsed=1.63s, log `review-stage/code_audit/test_logs/lang.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/extract/ -v --tb=short
```
Result: rc=1, elapsed=4.904s, log `review-stage/code_audit/test_logs/extract.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/ -v --tb=short
```
Result: rc=1, elapsed=7.132s, log `review-stage/code_audit/test_logs/watermark.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/watermark/token_channel/ -v --tb=short
```
Result: rc=1, elapsed=5.038s, log `review-stage/code_audit/test_logs/watermark_token_channel.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/evaluation/ -v --tb=short
```
Result: rc=0, elapsed=5.274s, log `review-stage/code_audit/test_logs/evaluation.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v --tb=short
```
Result: rc=0, elapsed=0.917s, log `review-stage/code_audit/test_logs/integration_cli_resolution.log`

```bash
/usr/bin/timeout 600s /root/miniconda3/bin/conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v --tb=short
```
Result: rc=1, elapsed=2.887s, log `review-stage/code_audit/test_logs/integration_orchestration.log`

## Failure Notes

- `tests/extract/`: 10 failed, 120 passed. Failures include token-channel artifact mock `.to()` assumption, joint-score expectation drift, lexical-only summary expectation drift, pipeline `MagicMock` min_blocks crash, and replay detector structure/fail-closed expectation failures.
- `tests/watermark/`: 57 failed, 443 passed. Most failures are token-channel-related.
- `tests/watermark/token_channel/`: 38 failed, 159 passed. Dominant failure is `AssertionError: embed_dim must be divisible by num_heads`; training corpus/workflow API tests also fail.
- `tests/integration/test_orchestration.py`: 1 failed, 34 passed. Failing test is stale expected `ALL_PHASES` order/list.
- No group stopped for network download risk. No group hit the 600 second timeout.

## Passed Groups

- `tests/common/`: 67 passed.
- `tests/datasets/`: 9 passed.
- `tests/lang/`: 9 passed.
- `tests/evaluation/`: 87 passed, 4 warnings.
- `tests/integration/test_cli_resolution.py`: 21 passed.

## Not Run

- Full pytest suite was not run by design.
- No tests that would require model download were forced online.
