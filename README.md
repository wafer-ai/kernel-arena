# Kernel Arena — Benchmark Results

Public benchmark results from [Kernel Arena](https://kernelarena.ai/eval), a leaderboard for LLM-generated AI accelerator kernels.

## Benchmark Suites

| Suite | Hardware | Tasks | Models | Reference |
| --- | --- | --- | --- | --- |
| [WaferBench NVFP4](waferbench-nvfp4-b200/) | NVIDIA B200 (CUDA 12.8) | 6 fused NVFP4 inference kernels | GPT-5.4, Claude-4.6-Opus, Composer-1.5, Gemini-3.1-Pro | FlashInfer 0.2.6.post1 |
| [KernelBench HIP](kernelbench-hip-mi300x/) | AMD MI300X (ROCm 7.0) | 41 kernels across 4 difficulty levels | 11 models from Anthropic, OpenAI, Google, xAI, Moonshot, Z.AI | PyTorch (torch.allclose) |

## Links

- **Leaderboard:** [kernelarena.ai/eval](https://kernelarena.ai/eval)
- **Methodology:** [kernelarena.ai/methodology](https://kernelarena.ai/methodology)
- **Reward Hacking Catalog:** [kernelarena.ai/resources](https://kernelarena.ai/resources)
