"""Stage 06: generate GSM8K trajectories + extract step-boundary activations (E4). Skips without GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.loaders import Trace, extract_last_number, numbers_match
from src.data.segmentation import split_steps
from src.features.activations import extract_step_states, needs_gpu
from src.utils.io import Manifest, atomic_write_dataframe, atomic_write_text
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

    if cfg.data.synthetic or not needs_gpu():
        reason = "synthetic smoke mode" if cfg.data.synthetic else "CUDA unavailable (no GPU)"
        msg = (f"SKIPPED: activation extraction requires a GPU runtime ({reason}). "
               "Downstream stages continue on non-activation representations; "
               "E4 rows read NOT_RUN in RESULTS.md.")
        log.info(msg)
        manifest.log("06_extract_activations", args.config, "SKIPPED_GPU_ABSENT", {"reason": reason})
        print_summary_block("06_extract_activations", [msg])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions_path = Path(cfg.data.raw_dir) / "gsm8k_questions.jsonl"
    questions = [Trace(**json.loads(line)) for line in
                 questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(cfg.probes.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.probes.model, torch_dtype=torch.float16,
                                                 device_map="auto")
    model.eval()
    layers = list(cfg.probes.layer_sweep)
    if max(layers) > model.config.num_hidden_layers:
        raise ValueError(f"layer_sweep {layers} exceeds the model's "
                         f"{model.config.num_hidden_layers} layers.")

    records, per_layer, n_trajectories, n_correct = [], {layer: [] for layer in layers}, 0, 0
    for qi, question in enumerate(questions):
        if question.gold_answer is None:
            continue
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": question.question + "\nReason step by step."}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer(chat, return_tensors="pt").to(model.device)
        for sample in range(cfg.probes.samples_per_question):
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=cfg.probes.max_new_tokens, do_sample=True,
                                     temperature=cfg.probes.generation_temperature,
                                     top_p=cfg.probes.top_p, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            steps = split_steps(text)
            if len(steps) < cfg.preprocessing.min_steps:  # P5 applies to generated traces too
                continue
            # Grade the WHOLE trajectory once: last number vs GSM8K gold answer.
            final_correct = numbers_match(extract_last_number(text), question.gold_answer)
            n_trajectories += 1
            n_correct += int(final_correct)
            states = extract_step_states(model, tokenizer, chat, text, steps, layers)
            for step_index in range(len(steps)):
                records.append({"question_id": question.question_id, "sample": sample,
                                "step": step_index + 1, "n_steps": len(steps),
                                "rel_pos": (step_index + 1) / len(steps),
                                "final_correct": bool(final_correct)})
                for layer in layers:
                    per_layer[layer].append(states[layer][step_index])
        if qi % 20 == 0:
            log.info("%d/%d problems | %d trajectories | %d step-states",
                     qi, len(questions), n_trajectories, len(records))

    processed = Path(cfg.data.processed_dir)
    atomic_write_dataframe(pd.DataFrame(records), processed / "activation_metadata.csv")
    checksums = {}
    for layer, rows in per_layer.items():
        path = processed / f"hidden_states_L{layer}.npy"
        np.save(path, np.asarray(rows, dtype=np.float32))
        checksums[f"L{layer}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Tensors are git-ignored; the checksums are what gets committed.
    atomic_write_text(Path(cfg.outputs.tables_dir) / "activation_checksums.json",
                      json.dumps(checksums, indent=2))
    summary = [f"Trajectories kept: {n_trajectories} (accuracy {n_correct / max(1, n_trajectories):.3f})",
               f"Step-states: {len(records)} across layers {layers}",
               "Activation tensors are NOT committed; SHA-256 checksums in activation_checksums.json."]
    manifest.log("06_extract_activations", args.config, "OK",
                 {"n_states": len(records), "n_trajectories": n_trajectories, "layers": layers})
    print_summary_block("06_extract_activations", summary)


if __name__ == "__main__":
    main()
