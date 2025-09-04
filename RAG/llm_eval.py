#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Evaluation (Single Model vs. Gold, Full Version mit Per-Example)
-------------------------------------------------------------------
Berechnet eine große Sammlung an Metriken:

🔹 Overlap / Semantik:
- ROUGE-1/2/L/Lsum
- BERTScore
- MoverScore
- BLEURT (optional, braucht TF)
- METEOR
- chrF (Character n-gram F-score)

🔹 Fakten / QA:
- QuestEval (wenn installiert) -> Faktentreue (Source nötig)

🔹 Lesbarkeit:
- Flesch Reading Ease
- Wiener Sachtextformel

🔹 Stil / Länge:
- Durchschnittliche Länge
- Distinct-n (1/2)
- Compression Ratio (pred/src)

CSV (UTF-8) mit Spalten:
    id, source_text, gold, pred
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

# ---------- Utils ----------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in mapping:
                return mapping[n]
        return None
    col_id   = pick('id')
    col_src  = pick('source_text','source','document','input')
    col_gold = pick('gold','reference','goldstandard','target')
    col_pred = pick('pred','prediction','modeloutput','system')
    missing = [n for n,v in [('id',col_id),('gold',col_gold),('pred',col_pred)] if v is None]
    if missing:
        raise ValueError(f"Fehlende Spalten in CSV: {missing}. Vorhanden: {list(df.columns)}")
    rename_map = {col_id:'id', col_gold:'gold', col_pred:'pred'}
    if col_src: rename_map[col_src] = 'source_text'
    return df.rename(columns=rename_map)


def summarize_lengths(texts):
    lengths = [len(str(t).split()) for t in texts]
    return {
        'mean_len': float(np.mean(lengths)),
        'median_len': float(np.median(lengths)),
        'min_len': float(np.min(lengths)),
        'max_len': float(np.max(lengths)),
    }

def distinct_ngrams(texts, n=1):
    total, uniq = 0, set()
    for t in texts:
        toks = str(t).split()
        total += max(0, len(toks) - n + 1)
        for i in range(len(toks) - n + 1):
            uniq.add(tuple(toks[i:i+n]))
    return (len(uniq) / total) if total > 0 else float("nan")

# ---------- Metrics ----------

def compute_rouge(preds, refs):
    from evaluate import load
    rouge = load("rouge")
    return rouge.compute(predictions=preds, references=refs, use_stemmer=True)

def compute_bertscore(preds, refs, model="xlm-roberta-large"):
    from evaluate import load
    bertscore = load("bertscore")
    res = bertscore.compute(predictions=preds, references=refs, model_type=model, lang="de", rescale_with_baseline=True)
    return {
        'precision': float(np.mean(res['precision'])),
        'recall': float(np.mean(res['recall'])),
        'f1': float(np.mean(res['f1'])),
    }

def compute_moverscore(preds, refs):
    try:
        from evaluate import load
        moverscore = load("moverscore")
        res = moverscore.compute(predictions=preds, references=refs)
        return {'moverscore': float(np.mean(res['score']))}
    except Exception as e:
        return {'moverscore': f"Error: {e}"}

def compute_bleurt(preds, refs, checkpoint="bleurt-20"):
    try:
        from evaluate import load
        bleurt = load("bleurt")
        res = bleurt.compute(predictions=preds, references=refs, checkpoint=checkpoint)
        return {'bleurt': float(np.mean(res["scores"]))}
    except Exception as e:
        return {'bleurt': f"Error: {e}"}

def compute_meteor(preds, refs):
    try:
        from evaluate import load
        meteor = load("meteor")
        res = meteor.compute(predictions=preds, references=refs)
        return {'meteor': float(np.mean(res['meteor']))}
    except Exception as e:
        return {'meteor': f"Error: {e}"}

def compute_chrf(preds, refs):
    try:
        from evaluate import load
        chrf = load("chrf")
        res = chrf.compute(predictions=preds, references=refs)
        return {'chrf': float(res['score'])}
    except Exception as e:
        return {'chrf': f"Error: {e}"}

def compute_questeval(sources, preds):
    try:
        from evaluate import load
        questeval = load("questeval")
        res = questeval.compute(sources=sources, predictions=preds)
        return {'questeval': float(np.mean(res['scores']))}
    except Exception as e:
        return {'questeval': f"Error: {e}"}

def compute_readability(texts):
    try:
        import textstat
        scores = {
            'flesch_reading_ease': float(np.mean([textstat.flesch_reading_ease(t) for t in texts])),
            'wiener_sachtextformel': float(np.mean([textstat.wiener_sachtextformel(t, variant=1) for t in texts]))
        }
        return scores
    except Exception as e:
        return {'readability_error': str(e)}

def compute_compression_ratio(sources, preds):
    if not sources or all(str(s).strip()=="" for s in sources):
        return None
    ratios = []
    for s,p in zip(sources, preds):
        ls, lp = len(str(s).split()), len(str(p).split())
        if ls > 0:
            ratios.append(lp/ls)
    return float(np.mean(ratios)) if ratios else None

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluiert EIN Modell gegen Goldstandard (Full, mit Per-Example).")
    parser.add_argument("--csv", required=True, help="Pfad zur Eingabe-CSV")
    parser.add_argument("--outdir", default="eval_results_single_full", help="Ausgabeverzeichnis")
    parser.add_argument("--with-bleurt", action="store_true", help="BLEURT zusätzlich berechnen")
    parser.add_argument("--bleurt-ckpt", default="bleurt-20")
    parser.add_argument("--bertscore-model", default="xlm-roberta-large")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = standardize_columns(df)
    df = df.fillna("")

    refs = df['gold'].tolist()
    preds = df['pred'].tolist()
    sources = df['source_text'].tolist() if 'source_text' in df.columns else None

    # ---------- Gesamt-Scores ----------
    print("Berechne Gesamtscores ...")
    rouge = compute_rouge(preds, refs)
    bert = compute_bertscore(preds, refs, model=args.bertscore_model)
    mover = compute_moverscore(preds, refs)
    meteor = compute_meteor(preds, refs)
    chrf = compute_chrf(preds, refs)
    bleurt = compute_bleurt(preds, refs, checkpoint=args.bleurt_ckpt) if args.with_bleurt else None
    questeval = compute_questeval(sources, preds) if sources is not None else None
    readability = compute_readability(preds)
    lengths = summarize_lengths(preds)
    d1, d2 = distinct_ngrams(preds,1), distinct_ngrams(preds,2)
    compression = compute_compression_ratio(sources, preds) if sources is not None else None

    summary = {
        'n_examples': len(df),
        'rouge': rouge,
        'bertscore': bert,
        'moverscore': mover,
        'meteor': meteor,
        'chrf': chrf,
        'bleurt': bleurt,
        'questeval': questeval,
        'readability': readability,
        'lengths': lengths,
        'distinct1': d1,
        'distinct2': d2,
        'compression_ratio': compression,
    }

    # ---------- Per-Example-Scores ----------
    print("Berechne Per-Example Scores ...")
    per_example = []
    for i, (ref, pred) in enumerate(zip(refs, preds)):
        entry = {"id": df.iloc[i]["id"], "gold": ref, "pred": pred}
        rouge_i = compute_rouge([pred], [ref])
        entry.update({f"rouge_{k}": v for k,v in rouge_i.items()})
        bert_i = compute_bertscore([pred], [ref], model=args.bertscore_model)
        entry.update({f"bertscore_{k}": v for k,v in bert_i.items()})
        mover_i = compute_moverscore([pred], [ref])
        entry.update(mover_i)
        meteor_i = compute_meteor([pred], [ref])
        entry.update(meteor_i)
        chrf_i = compute_chrf([pred], [ref])
        entry.update(chrf_i)
        if args.with_bleurt:
            bleurt_i = compute_bleurt([pred], [ref], checkpoint=args.bleurt_ckpt)
            entry.update(bleurt_i)
        if sources is not None:
            questeval_i = compute_questeval([sources[i]], [pred])
            entry.update(questeval_i)
        per_example.append(entry)

    # ---------- Speichern ----------
    out_json = os.path.join(args.outdir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"JSON gespeichert: {out_json}")

    out_csv = os.path.join(args.outdir, "per_example_scores.csv")
    pd.DataFrame(per_example).to_csv(out_csv, index=False)
    print(f"Per-Example Scores gespeichert: {out_csv}")

    # TXT Report
    lines = []
    lines.append(f"N = {len(df)} Beispiele\n")
    lines.append("ROUGE:")
    for k,v in rouge.items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("\nBERTScore:")
    for k,v in bert.items():
        lines.append(f"  {k}: {v:.4f}")
    # lines.append(f"\nMoverScore: {mover}")
    lines.append(f"\nMETEOR: {meteor}")
    lines.append(f"\nchrF: {chrf}")
    if bleurt:
        lines.append(f"\nBLEURT: {bleurt}")
    if questeval:
        lines.append(f"\nQuestEval: {questeval}")
    lines.append("\nReadability:")
    for k,v in readability.items():
        lines.append(f"  {k}: {v}")
    lines.append("\nLängen & Diversität:")
    lines.append(json.dumps({**lengths, 'distinct1': d1, 'distinct2': d2}, ensure_ascii=False, indent=2))
    if compression is not None:
        lines.append(f"\nCompression Ratio (pred/src): {compression:.4f}")

    # Per-Example in TXT
    lines.append("\n\n--- Per-Example Scores ---")
    for entry in per_example:
        lines.append(f"\nID {entry['id']}:")
        for k, v in entry.items():
            if k in ["id","gold","pred"]:
                continue
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

    out_txt = os.path.join(args.outdir, "report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report gespeichert: {out_txt}")

if __name__ == "__main__":
    main()
