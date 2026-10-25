"""Stage 37 (GPU, smoke test): is our LettuceDetect inference the detector its authors released?

Scores released RAGTruth test responses with src/recovery/engine.LettuceDetector (direct
transformers inference) and, when the `lettucedetect` package is importable, with the package's
own predict_prompt(); prints span agreement between the two and response-level agreement of ours
with the human labels. Run before trusting D and E2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import engine as E  # noqa: E402
from src.recovery import ragtruth as rt  # noqa: E402
from src.recovery import stats as S  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragtruth", default="data/external/RAGTruth")
    ap.add_argument("--model", default="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="results/recovery/run/detector_check.json")
    args = ap.parse_args()
    resp, src = rt.load(args.ragtruth)
    rows = [r for r in resp if r["split"] == "test" and r["quality"] == "good"][: args.n]
    ours = E.LettuceDetector(args.model, device="cuda")
    res = ours.score_batch([{"prompt": src[r["source_id"]]["prompt"], "text": r["response"]} for r in rows])
    pred = [float(max(x["sent_scores"] or [0]) >= 0.5) for x in res]
    truth = [float(len(r["labels"]) > 0) for r in rows]
    report = {"ours_vs_human": S.binary_agreement(pred, truth), "truncated": sum(
        x.get("covered_chars", 0) < len(r["response"].rstrip()) - 2 for x, r in zip(res, rows))}
    try:
        from lettucedetect.models.inference import HallucinationDetector

        pkg = HallucinationDetector(method="transformer", model_path=args.model)
        agree, n = 0, 0
        for r, x in zip(rows, res):
            spans = pkg.predict_prompt(prompt=src[r["source_id"]]["prompt"], answer=r["response"], output_format="spans")
            a = float(len(spans) > 0)
            b = float(len(x["spans"]) > 0)
            agree += a == b
            n += 1
        report["package_response_level_agreement"] = agree / n
    except Exception as exc:  # noqa: BLE001
        report["package"] = f"not available: {exc!r}"[:200]
    print(json.dumps(report, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
