"""Stage 06: activation extraction (E4). Skips gracefully without GPU."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils.io import Manifest
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="06: extract activations E4")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("06_extract_activations", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    try:
        import torch

        from src.features.activations import needs_gpu

        gpu = needs_gpu()
    except ImportError:
        gpu = False
    if cfg.data.synthetic or not gpu:
        reason = ("synthetic smoke mode" if cfg.data.synthetic
                  else "CUDA unavailable (no GPU)")
        msg = (f"SKIPPED: activation extraction requires a GPU runtime. {reason}. "
               "Downstream stages continue on non-activation representations; "
               "E4 rows will read NOT_RUN in RESULTS.md.")
        log.info(msg)
        manifest.log("06_extract_activations", args.config,
                     "SKIPPED_GPU_ABSENT", {"reason": reason})
        print_summary_block("06_extract_activations", [msg])
        return

    # GPU path: generate GSM8K trajectories, one forward pass per trace.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data.loaders import load_gsm8k
    from src.data.segmentation import split_steps
    from src.features.activations import extract_step_states

    model_name = cfg.probes.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=__import__("torch").float16,
        device_map="auto", output_hidden_states=True,
    )
    model.eval()
    questions = load_gsm8k(limit=cfg.probes.n_problems)
    layers = list(cfg.probes.layer_sweep)

    def gold(answer: str):
        match = re.search(r"####\s*(-?[\d,\.]+)", answer or "")
        return match.group(1).replace(",", "").strip() if match else None

    records, per_layer = [], {layer: [] for layer in layers}
    import torch

    for qi, trace in enumerate(questions):
        prompt = trace.question + "\nReason step by step."
        messages = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer(chat, return_tensors="pt").to(model.device)
        for s in range(cfg.probes.samples_per_question):
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=cfg.probes.max_new_tokens,
                                     do_sample=True,
                                     temperature=cfg.probes.generation_temperature,
                                     top_p=0.95,
                                     pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                                    skip_special_tokens=True)
            steps = split_steps(text)
            if len(steps) < 2:
                continue
            states = extract_step_states(model, tokenizer, chat, text, steps, layers)
            for si in range(len(steps)):
                records.append({"question_id": f"gsm8k-gen:{qi}",
                                "sample": s, "step": si, "n_steps": len(steps),
                                "rel_pos": (si + 1) / len(steps),
                                "text": steps[si]})
                for layer in layers:
                    per_layer[layer].append(states[layer][si])
        if qi % 20 == 0:
            log.info("%d/%d problems | %d step-states", qi, len(questions), len(records))
    meta = pd.DataFrame(records)
    meta.to_csv(Path(cfg.data.processed_dir) / "activation_metadata.csv", index=False)
    import hashlib

    for layer, arr in per_layer.items():
        arr_np = np.array(arr, dtype=np.float32)
        path = Path(cfg.data.processed_dir) / f"hidden_states_L{layer}.npy"
        np.save(path, arr_np)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        log.info("layer %d: %s checksum=%s", layer, arr_np.shape, checksum)
    summary = [f"Collected {len(records)} step-states across layers {layers}.",
               "Activation tensors are NOT committed (see .gitignore + manifest)."]
    manifest.log("06_extract_activations", args.config, "OK",
                 {"n_states": len(records), "layers": layers})
    print_summary_block("06_extract_activations", summary)


if __name__ == "__main__":
    main()
