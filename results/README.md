# Reviewer evidence files

This directory collects the source artifacts supplied for reviewer inspection.
Prediction text and reported metric values are retained. Source hashes in the
manifests match the released code.

## Contents

- `amber_ricd_results/`: LLaVA and InstructBLIP AMBER predictions, metrics,
  and source manifests.
- `chair_ricd_results/`: InstructBLIP and LLaVA CHAIR predictions,
  metrics, evaluator logs, and evaluator artifacts.
- `pope_ricd_results/`: selected LLaVA, InstructBLIP, and Qwen POPE evidence
  across Random, Popular, and Adversarial settings for COCO, A-OKVQA, and GQA.
- `llava_amber/`: reviewer-facing LLaVA AMBER configuration and the original
  source-code manifest retained from the code release.

For the RiCD method, `beta` is fixed at 0.3 and the language-proxy candidate
count `k` (`top_k_proxy` in code) is fixed at 10. The separate
`generation.top_k` field controls token sampling and is not this method
parameter. Other settings are recorded in the per-experiment configuration
files and accompanying metrics where available.

Some JSON and log files contain absolute paths from the original experiment
server. These paths are retained as provenance metadata and are not expected to
exist on a reviewer's machine.

Files ending in `.pt` or `.pkl` are binary Python/PyTorch artifacts. Load them
only from a trusted copy of this repository.
