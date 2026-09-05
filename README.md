# RiCD: Risk-Calibrated Decoding

RiCD is a training-free, inference-time decoding method for reducing
hallucination in multimodal large language models. At each decoding step, it
combines deep, shallow, and text-only model paths to estimate language-driven
risk and visually supported recovery opportunities.

This repository is a cleaned code release assembled from the source snapshot
used for the reported LLaVA-1.5-7B / AMBER experiment. The historical Python
class names `RCCRDecoder` and `RCCRPipeline` are retained for compatibility.

## Method overview

For each candidate token, RiCD:

1. computes language and visual conflict proxies from deep, shallow, and
   blind logits;
2. converts those proxies into language-risk and suppression-risk surrogates;
3. applies candidate-set validation and adaptive plausibility constraints;
4. calibrates the deep-path token score with the two risk terms.

The core implementation is in
[`src/decoder/rccr_decoder.py`](src/decoder/rccr_decoder.py).



## Repository layout

```text
configs/                 portable runtime paths
results/                 reviewer-facing predictions, metrics, and manifests
scripts/                 AMBER experiment runner
src/decoder/             RiCD core decoder
src/models/              model adapters and KV-cache utilities
src/proxy/               cross-modal conflict proxies
src/risk/                risk surrogate functions
```

## Installation

The recorded LLaVA experiment used Linux, Python 3.10, and a CUDA-capable GPU.
Use a separate environment for LLaVA because the upstream LLaVA package pins
older `transformers` and `accelerate` releases than the Hugging Face adapters
listed in this repository's `requirements.txt`.

```bash
python3.10 -m venv .venv-llava
source .venv-llava/bin/activate
python -m pip install --upgrade pip

mkdir -p third_party
git clone https://github.com/haotian-liu/LLaVA.git third_party/LLaVA
python -m pip install -e third_party/LLaVA
python -m pip install "Pillow>=9" "tqdm>=4.60"
```

The `requirements.txt` file records shared dependencies for the Hugging Face
adapters, but it is not a complete model-specific lockfile. Do not install it
into the LLaVA environment above.

The released manifests do not record the exact upstream LLaVA revision used by
the historical run. The commands above therefore prepare a functional rerun,
not a bit-for-bit reconstruction of that upstream environment.

Clone the official [AMBER benchmark](https://github.com/junyangwang0410/AMBER)
and check out the revision recorded by the released experiment configuration:

```bash
git clone https://github.com/junyangwang0410/AMBER.git third_party/AMBER
git -C third_party/AMBER checkout 534babf6bbfcce2e735c26289dedfb21cef3c939
```

Download the AMBER images according to the official instructions. The runner
expects the query file at
`third_party/AMBER/data/query/query_generative.json`.

Set local paths without editing tracked files:

```bash
export LLAVA_MODEL_PATH=/absolute/path/to/llava-v1.5-7b
export AMBER_IMAGE_DIR=/absolute/path/to/amber/images
```

## Validation and inference

Check the registered seed, the ordered 1,004-query AMBER generative split, and
the presence of every referenced image without loading the model:

```bash
python scripts/run_amber_protocol.py \
  --model llava-1.5-7b \
  --method ricd \
  --config results/llava_amber/config.json \
  --validate_only \
  --require_images
```

This command does not run inference or the AMBER evaluator.

Verify the six RiCD source files listed in the released LLaVA AMBER manifest:

```bash
python scripts/verify_snapshot.py
```

This hash check does not validate third-party packages, predictions, metrics,
data files, or the runtime environment.

With the complete 1,004-image split installed, run inference on two samples:

```bash
python scripts/run_amber_protocol.py \
  --model llava-1.5-7b \
  --method ricd \
  --config results/llava_amber/config.json \
  --max_samples 2
```

The runner checks all 1,004 image paths before applying `--max_samples`; this
option limits inference time but does not reduce the data-installation
requirement. Run the full prediction generation by omitting `--max_samples`:

```bash
python scripts/run_amber_protocol.py \
  --model llava-1.5-7b \
  --method ricd \
  --config results/llava_amber/config.json \
  --namespace amber_cover_neighbor_greedy128_full_v1_llava_ln4
```

Full inference requires the model weights, the complete AMBER generative split,
and a CUDA environment. Set up the official AMBER evaluator in a separate
environment:

```bash
python3.10 -m venv .venv-amber-eval
source .venv-amber-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install spacy nltk tqdm
python -m spacy download en_core_web_lg
python -m nltk.downloader \
  punkt punkt_tab averaged_perceptron_tagger \
  averaged_perceptron_tagger_eng wordnet omw-1.4
```

With the default output root, evaluate the full responses from the AMBER
repository directory so that the evaluator can resolve its data files. The
official evaluator prints its metrics to standard output, so the command also
saves that output for inspection:

```bash
(
  cd third_party/AMBER
  python inference.py \
    --inference_data ../../outputs/amber_cover_neighbor_greedy128_full_v1_llava_ln4/llava-1.5-7b/amber/ricd/seed1/responses.json \
    --evaluation_type g \
    | tee ../../outputs/amber_cover_neighbor_greedy128_full_v1_llava_ln4/llava-1.5-7b/amber/ricd/seed1/official_amber_stdout.txt
)
```
