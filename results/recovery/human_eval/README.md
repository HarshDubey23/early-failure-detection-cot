# Blinded human comparison kit (Run R)

`sheet.html` and `sheet_blank.csv` show 80 RAGTruth items with two outputs each, CARE and
restart-until-D-passes in random A/B order. Two raters mark unsupported sentences and a preference;
`python scripts/38_human_eval_kit.py score` computes each policy's human hallucination rate, the
paired difference and inter-rater kappa.

**Blinding.** The A/B key of this sheet was committed in `09dafdf` and is therefore public in the
history. Before handing sheets to raters, regenerate the kit with a new seed so that the public key
no longer applies:

    python scripts/38_human_eval_kit.py make --ref restart_fb --n 80 --seed <new seed> --out results/recovery/human_eval

`make` now also writes `KEY.sha256`. Commit the sheets and `KEY.sha256`, never the key
(`.gitignore` excludes it). Publish the key after rating; anyone can check it against the hash.
