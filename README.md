# Qwen VL Flash Router

> **Production Solution for Qwen2.5-VL Visual Token Packing & FlashAttention-3 MoE Routing**

## Overview
Dynamic visual token packer and FlashAttention-3 MoE router for Alibaba Cloud Qwen2.5-VL and Qwen2.5-Coder architectures.

## Verification
```bash
PYTHONPATH=src python3 tests/test_qwen.py
python3 mastermind_sidecar.py
```

---

## Fleet ops (transparent)

This repo may include **`.integrity/`** (SHA-256 baselines / watchdog) and/or a health sidecar.
These are **documented multi-repo fleet operations**, not covert implants.

See [SECURITY_AND_FLEET_OPS.md](SECURITY_AND_FLEET_OPS.md) and
`~/GlacierEQ_Swarm/state/PORTFOLIO_SHADOW_AND_GAUNTLET.md`.

## Helix strand

See [HELIX_STRAND.md](HELIX_STRAND.md) — piston/spiral role in the portfolio double helix.
