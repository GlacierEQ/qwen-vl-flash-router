# Qwen VL Flash Router

> **Production Solution for Qwen2.5-VL Visual Token Packing & FlashAttention-3 MoE Routing**

## Overview
Dynamic visual token packer and FlashAttention-3 MoE router for Alibaba Cloud Qwen2.5-VL and Qwen2.5-Coder architectures.

## Verification
```bash
PYTHONPATH=src python3 tests/test_qwen.py
python3 mastermind_sidecar.py
```
