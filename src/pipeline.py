# pipeline.py
# Unified pipeline: extract / normalize / evaluate
# Safe creds via .env; no Phase labels anywhere.

import os, re, sys, glob, argparse
import pandas as pd
from dotenv import load_dotenv

# ===== Optional deps for extract =====
try:
    from hugchat import hugchat
    from hugchat.login import Login
    HAS_HUGCHAT = True
except Exception:
    HAS_HUGCHAT = False

# ===== Optional deps for evaluate/normalize =====
from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from metaphone import doublemetaphone

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def load_env():
    load_dotenv()
    return os.getenv("HUG_EMAIL"), os.getenv("HUG_PASSWORD")

# --------- EXTRACT ----------
PROMPT = (
    "Summarize NCT ID: {nct} and its study title: {title} "
    "in a structured format with headings like 'study title', "
    "'interventions (list the control or experimental arms and drug names concisely)', "
    "'condition', 'study status', 'phases', 'primary objective', and "
    "'specific conclusions for each arm' of the Results. "
    "Please search https://classic.clinicaltrials.gov/ct2/history/{nct}"
)

def do_extract(args):
    if not HAS_HUGCHAT:
        raise RuntimeError("hugchat not installed. `pip install hugchat python-dotenv`")
    email, pw = load_env()
    if not (email and pw):
        raise RuntimeError("Missing HUG_EMAIL/HUG_PASSWORD in .env")

    df = pd.read_csv(args.input)
    for col in ["NCT Number", "Study Title"]:
        if col not in df.columns:
            raise ValueError(f"Input must contain column '{col}'")

    ensure_dir(args.outdir); ensure_dir("cookies")
    from datetime import datetime
    from hugchat.login import Login

    login = Login(email, pw)
    cookies = login.login(cookie_dir_path="cookies", save_cookies=True)
    bot = hugchat.ChatBot(cookies=cookies.get_dict())

    index_rows = []
    for _, row in df.iterrows():
        nct = str(row["NCT Number"]); title = str(row["Study Title"])
        outfile = os.path.join(args.outdir, f"{nct}.txt")
        if os.path.exists(outfile) and not args.overwrite:
            index_rows.append({"NCT Number": nct, "Study Title": title, "Output": outfile})
            continue
        bot.new_conversation(4, switch_to=True)
        res = bot.query(PROMPT.format(nct=nct, title=title), web_search=True)
        sources = []
        try:
            for s in res.web_search_sources:
                sources.append(f"- {getattr(s, 'title', '')} | {getattr(s, 'link', '')}")
        except Exception:
            pass
        text = "\n".join([
            f"NCT: {nct}",
            f"Study Title: {title}",
            "=== Summary ===",
            str(res),
            "=== Sources ===",
            *sources if sources else ["(none)"],
            "",
        ])
        with open(outfile, "w", encoding="utf-8") as f: f.write(text)
        index_rows.append({"NCT Number": nct, "Study Title": title, "Output": outfile})
        print(f"[extract] wrote {outfile}")

    if args.index:
        if os.path.exists(args.index):
            old = pd.read_csv(args.index)
            merged = pd.concat([old, pd.DataFrame(index_rows)], ignore_index=True)\
                     .drop_duplicates(subset=["NCT Number"])
            merged.to_csv(args.index, index=False)
        else:
            pd.DataFrame(index_rows).to_csv(args.index, index=False)
        print(f"[extract] updated index: {args.index}")

# --------- NORMALIZE ----------
SYNONYMS = {
    "interferon alfa-2b": {"recombinant alpha interferon", "human interferon alpha2b"},
    "interferon alfa-2a": {"human interferon alpha2a", "interferon alpha2a"},
    "interferon alfa-n1": {"interferon alpha", "human interferon alpha"},
    "zidovudine": {"azidothymidine"},
    "sargramostim": {"rhugmcsf", "gmcsf"},
    "fenretinide": {"4hpr"},
    "ivig": {"intravenous immunoglobulin"},
    "y-90 humanized anti-tac": {"90yantitac"},
    "tumor necrosis factor": {"rtnf"},
    "aldesleukin": {"interleukin2"},
    "trabectedin": {"ecteinascidin 743"},
    "il2": {"interleukin 2"},
    "ad5cmv-p53 gene": {"adp53", "adwtp53"},
    "recombinant interleukin-12": {"il-12"},
    "mkc-1": {"mkc 1"},
}

def clean_txt(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r"\b(drug|biological):\b", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t

def canon(t: str) -> str:
    c = clean_txt(t)
    for k, syn in SYNONYMS.items():
        if c == k or c in syn: return k
    return c

def do_normalize(args):
    df = pd.read_csv(args.input)
    for col in ["NCT ID", "original intervention"]:
        if col not in df.columns:
            raise ValueError(f"Input must contain column '{col}'")

    def normalize_row(s):
        parts = [p.strip() for p in str(s).split("|") if p.strip()]
        mapped = sorted(set(canon(p) for p in parts))
        return ",".join(mapped)

    df["std Treatment"] = df["original intervention"].apply(normalize_row)
    df.to_csv(args.output, index=False)
    print(f"[normalize] wrote {args.output}")

# --------- EVALUATE ----------
TFIDF = TfidfVectorizer()
def has_overlap(a, b): return bool(set(canon(a).split()) & set(canon(b).split()))
def tfidf_sim(a, b):
    m = TFIDF.fit_transform([canon(a), canon(b)])
    return float(cosine_similarity(m[0:1], m[1:2])[0][0])
def phonic_eq(a, b):
    return doublemetaphone(canon(a))[0] == doublemetaphone(canon(b))[0]

def build_w2v(tokens_list):
    return Word2Vec(tokens_list, vector_size=100, window=5, min_count=1, workers=4)

def w2v_sim(a, b, model):
    t1, t2 = canon(a).split(), canon(b).split()
    s, c = 0.0, 0
    for x in t1:
        for y in t2:
            if x in model.wv and y in model.wv:
                s += model.wv.similarity(x, y); c += 1
    return s / c if c else 0.0

def similar(a, b, model, fuzzy=80, lev=5, w2v_th=0.7, tfidf_th=0.7):
    A, B = canon(a), canon(b)
    if fuzz.partial_ratio(A, B) >= fuzzy: return True
    if levenshtein_distance(A, B) <= lev: return True
    if has_overlap(a, b): return True
    if tfidf_sim(a, b) >= tfidf_th: return True
    if phonic_eq(a, b): return True
    if w2v_sim(a, b, model) >= w2v_th: return True
    return False

def do_evaluate(args):
    df = pd.read_csv(args.input)
    for col in ["NCT ID", "original intervention", "std Treatment"]:
        if col not in df.columns:
            raise ValueError(f"Input must contain columns: {col}")

    sentences = [canon(x).split() for x in df["std Treatment"].dropna().astype(str)]
    w2v = build_w2v(sentences)
    comb = df.groupby("NCT ID", as_index=False).agg({
        "original intervention": "first",
        "std Treatment": lambda s: ",".join(pd.Series(list(
            dict.fromkeys(",".join(map(str, s)).split(",")))).tolist())
    })

    rows = []
    for _, r in comb.iterrows():
        nct = r["NCT ID"]
        original = [t.strip() for t in str(r["original intervention"]).split("|") if t.strip()]
        stds = [canon(x) for x in str(r["std Treatment"]).split(",") if x.strip()]
        missing = []
        for t in original:
            if not any(similar(t, s, w2v) for s in stds):
                missing.append(t)
        acc = (len(original) - len(missing)) / len(original) if original else 0.0
        rows.append({
            "NCT ID": nct,
            "original intervention": r["original intervention"],
            "std Treatment (combined)": ",".join(stds),
            "missing_treatments": ",".join(missing),
            "accuracy": acc
        })

    rep = pd.DataFrame(rows)
    rep["accuracy_percent"] = rep["accuracy"].apply(lambda x: f"{x:.2%}")
    rep.to_csv(args.report, index=False)
    overall = rep["accuracy"].mean()
    print(f"[evaluate] wrote {args.report} | Overall: {overall:.4f} ({overall:.2%})")

# --------- CLI ----------
def build_cli():
    p = argparse.ArgumentParser(description="Unified LLM Clinical Trials Pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract NCT summaries via HuggingChat web_search")
    pe.add_argument("--input", required=True)
    pe.add_argument("--outdir", default="outputs")
    pe.add_argument("--index", default="results.csv")
    pe.add_argument("--overwrite", action="store_true")
    pe.set_defaults(func=do_extract)

    pn = sub.add_parser("normalize", help="Normalize interventions to std Treatment")
    pn.add_argument("--input", required=True)
    pn.add_argument("--output", required=True)
    pn.set_defaults(func=do_normalize)

    pv = sub.add_parser("evaluate", help="Evaluate matching accuracy")
    pv.add_argument("--input", required=True)
    pv.add_argument("--report", default="accuracy_report.csv")
    pv.set_defaults(func=do_evaluate)
    return p

def main():
    args = build_cli().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
