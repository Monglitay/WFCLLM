# 双通道水印系统调优 — 自主实验 Prompt

## 目标
- Pass@1 ≈ 60%（当前 baseline 约 65%）
- TPR@5%FPR ≈ 50%（当前最好：cap=8.0 时 23.5%，无 cap 时 93.8% 但 pass=0%）

## 当前状态

### 代码位置
- 生成端 bias 逻辑：`wfcllm/watermark/orchestrator.py` 第 538 行附近，`_apply_token_channel_bias` 方法
- 提取端检测逻辑：`wfcllm/extract/token_channel.py`，`ReplayTokenChannelDetector.detect()`
- 配置文件：`configs/base_config.json`（token_channel.delta 当前=4.0）

### 当前实现：自适应 delta + cap
```python
# orchestrator.py _apply_token_channel_bias 中：
# 计算 top-k 中 best_red 和 best_green 的 gap
# effective_delta = min(gap + 1.0, delta * 2.0)  # cap = delta * 2.0
# 如果 green 已经领先，effective_delta = delta（固定值）
```

### 核心矛盾
7B 模型在 temperature=0.2 下 logit 分布极度 peaked。top token 的 raw logit 优势经常 >10。
- delta 太小（≤4.0）：bias 无法翻转 red top token → green_fraction ≈ 0.5 → z ≈ 0
- delta 太大（无 cap）：强行选语义不通的 green token → 代码乱码 → pass=0%
- cap=8.0（delta*2）：部分位置翻转成功，但大部分 gap>8 的位置仍然失败 → TPR=23.5%

### 已尝试的方案及结果

| 方案 | TPR | Pass@1 | 问题 |
|------|-----|--------|------|
| 固定 delta=2.0 | ~0% | ~65% | 信号太弱 |
| 固定 delta=4.0 | ~5% | ~65% | 信号太弱 |
| 自适应 delta 无 cap | 93.8% | 0% | 代码全毁 |
| 自适应 delta cap=8.0 | 23.5% | 60% | 信号不够强 |
| 只 bias 不确定位置（gap≤delta 才 bias） | 5.6% | 60% | 提取端无法对齐 |
| 提取端用小模型 gap 过滤 | 11.1% | 60% | 小模型和主 LLM 相关性不够 |

### 可探索的方向

1. **调整 cap 倍数**：当前 cap = delta * 2.0。试试 delta * 3.0（=12.0）或 delta * 2.5（=10.0），找 TPR/Pass 平衡点
2. **提高 temperature**：从 0.2 提到 0.4-0.6。分布更平坦，bias 更容易生效，但代码多样性增加
3. **联合调参**：temperature=0.4 + delta=3.0 + cap=delta*3（分布平坦后不需要那么大的 delta）
4. **OR 判决规则**：把 joint detection 从加权求和改为 OR（semantic_z > threshold OR lexical_z > threshold）
5. **去掉 semantic retry 中的 lexical hook**：让 retry 不受 lexical 约束，semantic 通道更容易成功

## 运行实验的方式

### 嵌入（约 15-70 分钟，取决于 retry 次数）
```bash
screen -dmS exp bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate WFCLLM && TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -u run.py --phase watermark --config configs/base_config.json --sample-limit 20 --token-channel-enabled true --token-channel-mode dual-channel --token-channel-switch-threshold -10.0 --force 2>&1 | tee /tmp/exp_embed.log; echo "EMBED DONE" >> /tmp/exp_embed.log'
```

### 提取（约 10-15 分钟，含校准）
```bash
screen -dmS extract bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate WFCLLM && TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -u run.py --phase extract --config configs/base_config.json --input-file <watermarked_file.jsonl> --token-channel-enabled true --token-channel-mode dual-channel --token-channel-switch-threshold -10.0 --force 2>&1 | tee /tmp/exp_extract.log; echo "EXTRACT DONE" >> /tmp/exp_extract.log'
```

### Pass@1 评估脚本
```python
# 保存为 /tmp/eval_pass.py，用法：python /tmp/eval_pass.py <watermarked_file.jsonl>
import json, sys, subprocess, tempfile, os, re, ast
sys.path.insert(0, '/root/autodl-tmp/WFCLLM')
from datasets import load_dataset

ds = load_dataset('openai/openai_humaneval', 'openai_humaneval', cache_dir='data/datasets/humaneval')
test_cases = {item['task_id']: item['test'] for item in ds['test']}
entry_points = {item['task_id']: item['entry_point'] for item in ds['test']}

PREAMBLE = 'from typing import List, Tuple, Dict, Optional, Any, Set\nimport math\nimport re\nimport sys\nimport hashlib\nimport collections\nfrom itertools import *\nfrom functools import *\n\n'

def extract_function(code, entry_point):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                lines = code.split('\n')
                return '\n'.join(lines[:node.end_lineno])
    except SyntaxError:
        pass
    lines = code.split('\n')
    in_func = False
    func_lines = []
    for line in lines:
        if line.startswith(f'def {entry_point}'):
            in_func = True
        elif in_func and line and not line[0].isspace() and not line.startswith('#'):
            break
        if in_func:
            func_lines.append(line)
    return '\n'.join(func_lines) if func_lines else code

with open(sys.argv[1]) as f:
    items = [json.loads(line) for line in f]

passed = failed = 0
for item in items:
    task_id = item['id']
    code = item['generated_code']
    prompt = item['prompt']
    entry = entry_points.get(task_id, '')
    match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
    if match: code = match.group(1)
    elif '```' in code:
        match = re.search(r'```\n(.*?)```', code, re.DOTALL)
        if match: code = match.group(1)
    full_code = extract_function(code if code.strip().startswith('def ') else prompt + code, entry)
    test = test_cases.get(task_id, '')
    script = PREAMBLE + full_code + '\n\n' + test + f'\n\ncheck({entry})\n'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script); tmp = f.name
    try:
        r = subprocess.run(['/root/miniconda3/envs/WFCLLM/bin/python', tmp], capture_output=True, text=True, timeout=10)
        if r.returncode == 0: passed += 1
        else: failed += 1
    except: failed += 1
    finally: os.unlink(tmp)

print(f'Pass@1: {passed}/{passed+failed} = {passed/(passed+failed)*100:.1f}%')
```

### 修改参数的方式
- `configs/base_config.json` 中 `token_channel.delta`：基础 delta 值
- `wfcllm/watermark/orchestrator.py` 中 `max_delta = delta * 2.0`：cap 倍数（改这个数字）
- `configs/base_config.json` 中 `temperature`：生成温度（当前 0.2）
- `configs/base_config.json` 中 `max_retries`：语义通道最大重试次数（当前 15）

### 监控频率
每半小时检查一次实验进度。不要频繁 poll。用 `grep "EMBED DONE\|EXTRACT DONE" /tmp/exp_*.log` 检查是否完成。

## 工作流程

1. 修改参数/代码
2. 用 screen 启动嵌入（20 样本）
3. 等嵌入完成后启动提取
4. 提取完成后跑 pass@1 评估
5. 记录结果，分析，决定下一步
6. 如果 TPR 和 Pass@1 都达标，提交代码

## 重要注意事项
- 环境变量必须加 `TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`（离线环境）
- 用 screen 跑长任务，不要用后台 & 或 nohup
- 提取输出文件名在嵌入日志最后一行 `[完成] 水印数据集已保存至 data/watermarked/humaneval_YYYYMMDD_HHMMSS.jsonl`
- 提取结果在 `data/results/humaneval_YYYYMMDD_HHMMSS_summary.json`
- 详细结果在 `data/results/humaneval_YYYYMMDD_HHMMSS_details.jsonl`
