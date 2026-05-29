#!/bin/bash
source activate WFCLLM
cd /root/autodl-tmp/WFCLLM
HF_HUB_OFFLINE=1 python run.py --config configs/base_config.json --phase token-channel-train --force
