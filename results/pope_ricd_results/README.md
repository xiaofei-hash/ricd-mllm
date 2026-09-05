# Selected POPE RiCD evidence package

This package contains the exact repository artifacts matching the requested LLaVA, InstructBLIP, and Qwen result rows.

## Parameters
- LLaVA-1.5-7B: lambda_l=1.0, lambda_s=0.5, tau_l=tau_v=1.0, beta=0.3, top_k_proxy=10.
- InstructBLIP-7B (J3): lambda_l=0.5, lambda_s=1.0, tau_l=1.0, tau_v=1.0, beta=0.3, top_k_proxy=10.
- Qwen2.5-VL-7B-Instruct: lambda_l=1.5, lambda_s=1.0, tau_l=tau_v=1.0, beta=0.3, top_k_proxy=10.

All released POPE metrics record seed 1 and `do_sample=false`. The summary reports macro averages over COCO, A-OKVQA, and GQA.

## Contents
- results/<model>/<type>/<dataset>/predictions.jsonl
- results/<model>/<type>/<dataset>/metrics.json
- results/<model>/<type>/<dataset>/rng_state.pt
- configs/: available protocol configuration records; not every released result has a standalone configuration file
- logs/: available original run logs
- summary_metrics.csv: dataset-level and macro-average metrics
- source_manifest.json: original namespace and path provenance
- checksums.sha256: SHA-256 integrity hashes
