# Qwen VL Flash Router — Multi-Modal Router & CUDA Vision Tokenizer 👁️

> **TypeScript multi-modal flash attention router & CUDA kernel for vision-language image patch tokenization.**

[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Domain](https://img.shields.io/badge/Domain-Vision--Language-blue)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements the **Qwen VL Flash Router & Vision Tokenizer** — routing vision-language prompts and tokenizing image patches via CUDA. It demonstrates:

- **TypeScript Flash Attention Budgeting** allocating token budgets dynamically across image/text modalities
- **CUDA image patch tokenization kernel** converting 14x14 RGB patches to 1152-dim vision embeddings
- **Dynamic resolution merging** compressing vision tokens when sequence budgets are exceeded
- **Multi-GPU load balancing** routing multi-modal streams to optimal GPU nodes

**Why this matters**: Vision-Language-Models (VLMs) process thousands of image tokens per frame. Combining CUDA vision tokenization with TypeScript flash attention routing enables real-time VLM inference.

---

## 🔬 For Engineers & Technical Reviewers

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `src/flash_router.ts` | TypeScript | Multi-modal request router & attention budget manager |
| `src/vision_pipeline.cu` | CUDA | CUDA kernel for image patch embedding & 2D position encoding |
| `tests/` | Python | End-to-end vision-language routing simulation |

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `route_vl_request()` — router endpoint queryable by multi-modal agents
- **Mastermind Sidecar**: Fully connected to APEX Highway mesh
- **SHA-256 Integrity**: Tracked in `.integrity/file_hashes.json`

---

## ⚡ Quick Start

```bash
python3 tests/test_flash_router.py
```
