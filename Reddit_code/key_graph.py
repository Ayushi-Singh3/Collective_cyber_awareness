import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

CORPUS_FILE   = r"D:\For Rajesh'sir\Result of code\corpus_50k_FINAL.csv"
RQ3_FOLDER    = r"D:\For Rajesh'sir\Result of code\RQ3_Analysis"
OUTPUT_FOLDER = r"D:\For Rajesh'sir\Result of code\Topic_Sentiment"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

corpus = pd.read_csv(CORPUS_FILE)
corpus["full_text"] = corpus["full_text"].fillna("").str.lower()

date_col = None
for col in corpus.columns:
    if any(x in col.lower() for x in ['date','time','utc','created']):
        date_col = col
        break

try:
    corpus["parsed_date"] = pd.to_datetime(corpus[date_col], unit="s", errors="coerce")
    if corpus["parsed_date"].isna().all():
        corpus["parsed_date"] = pd.to_datetime(corpus[date_col], errors="coerce")
except:
    corpus["parsed_date"] = pd.to_datetime(corpus[date_col], errors="coerce")

corpus["year"] = corpus["parsed_date"].dt.year

# Row 1: Prompt Injection, Ransomware, Privacy
# Row 2: Deepfake, Data Breach, Phishing
KEYWORDS = {
    "Prompt Injection"          : [r"\bprompt injection\b", r"\bjailbreak\b"],
    "Ransomware"                : [r"\bransomware\b"],
    "Privacy"                   : [r"\bprivacy\b"],
    "Deepfake"                  : [r"\bdeepfake\b", r"\bdeep fake\b",
                                    r"\bai voice\b", r"\bvoice cloning\b"],
    "Data Breach"               : [r"\bdata breach\b", r"\bdata leak\b"],
    "Phishing incl. AI Phishing": [r"\bphishing\b", r"\bspear phishing\b",
                                    r"\bai phishing\b", r"\bai.generated phishing\b",
                                    r"\bllm phishing\b"],
}

years = list(range(2019, 2026))

results = {}
for name, patterns in KEYWORDS.items():
    combined = "|".join(patterns)
    yearly   = []
    for y in years:
        count = corpus[corpus["year"]==y]["full_text"].str.contains(
            combined, regex=True, na=False
        ).sum()
        yearly.append(int(count))
    results[name] = yearly
    print(f"{name}: {yearly} | Total={sum(yearly)}")

subplot_labels = [
    "(a) Prompt Injection",
    "(b) Ransomware",
    "(c) Privacy",
    "(d) Deepfake",
    "(e) Data Breach",
    "(f) Phishing incl. AI Phishing",
]

# 2 rows x 3 columns
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, name in enumerate(KEYWORDS.keys()):
    ax     = axes[idx]
    values = results[name]

    ax.plot(years, values,
            color="black",
            linewidth=1.8,
            marker="o",
            markersize=5,
            markerfacecolor="black")

    ax.axvline(x=2022, color="red",
               linestyle="dotted",
               linewidth=1.5)

    ax.set_title(subplot_labels[idx],
                 fontsize=12,
                 fontweight="bold",
                 loc="left")

    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years],
                       fontsize=9, rotation=45)
    ax.set_ylabel("Number of Posts", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

fig.suptitle(
    "Year-wise evolution of key cybersecurity terms in Reddit communities.\n"
    "The year 2022 marks ChatGPT launch (shown by vertical dotted red line).",
    fontsize=12, y=1.01
)

fig.patch.set_facecolor("white")
plt.subplots_adjust(hspace=0.6, wspace=0.3)

out = rf"{OUTPUT_FOLDER}\keyword_pic.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out}")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"{'Keyword':<30} {'Pre':>8} {'Post':>8} {'Change':>10}")
print("-"*60)
for name in KEYWORDS.keys():
    pre  = sum(results[name][:4])
    post = sum(results[name][4:])
    if pre > 0:
        pct    = (post-pre)/pre*100
        change = f"{pct:+.1f}%"
    else:
        change = "NEW"
    print(f"{name:<30} {pre:>8} {post:>8} {change:>10}")

print("\nDone!")