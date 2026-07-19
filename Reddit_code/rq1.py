

import pandas as pd
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from scipy import stats
import statsmodels.api as sm
import torch



DATA_FOLDER    = r"D:\For Rajesh'sir\Result of code"
GRAPHS_FOLDER  = r"D:\For Rajesh'sir\Result of code\RQ1_Graphs"
ANALYSIS_FOLDER= r"D:\For Rajesh'sir\Result of code\RQ1_Analysis"

os.makedirs(GRAPHS_FOLDER,   exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

INPUT_FILE  = os.path.join(DATA_FOLDER,
                            "corpus_50k_FINAL.csv")
OUTPUT_FILE = os.path.join(ANALYSIS_FOLDER,
                            "RQ1_sentiment_complete.csv")



print("=" * 55)
print("RQ1 ANALYSIS")
print("=" * 55)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ")
    DEVICE = 0
else:
    print("GPU not found — using CPU")
    DEVICE = -1



# Resume support
vader_file    = os.path.join(ANALYSIS_FOLDER,
                              "step1_vader.csv")
roberta_file  = os.path.join(ANALYSIS_FOLDER,
                              "step2_roberta.csv")
complete_file = os.path.join(ANALYSIS_FOLDER,
                              "RQ1_sentiment_complete.csv")

if os.path.exists(complete_file):
    print("\nAll steps done! Loading final file...")
    df = pd.read_csv(complete_file)
elif os.path.exists(roberta_file):
    print("\nRoBERTa done! Loading...")
    df = pd.read_csv(roberta_file)
elif os.path.exists(vader_file):
    print("\nVADER done! Loading...")
    df = pd.read_csv(vader_file)
else:
    print(f"\nLoading data: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

df["text"]      = df["text"].fillna("")
df["full_text"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
).str.strip()

print(f"Total posts: {len(df):,}")
print(df["era"].value_counts().to_string())



if "vader_label" not in df.columns:
    print("\n" + "─"*45)
    print("STEP 1: VADER SENTIMENT")
    print("Expected: ~10 minutes")
    print("─"*45)

    vader = SentimentIntensityAnalyzer()

    def get_vader(text):
        s = vader.polarity_scores(str(text))
        c = s["compound"]
        label = ("positive" if c >= 0.05
                 else "negative" if c <= -0.05
                 else "neutral")
        return label, round(c, 4)

    start = time.time()
    df[["vader_label", "vader_score"]] = (
        df["full_text"].apply(
            lambda t: pd.Series(get_vader(t))
        )
    )
    elapsed = round((time.time()-start)/60, 1)
    print(f"VADER done in {elapsed} min!")
    print(df["vader_label"].value_counts().to_string())
    df.to_csv(vader_file, index=False)
    print(f"Saved: step1_vader.csv")
else:
    print("\nVADER: already done ")



if "roberta_label" not in df.columns:
    print("\n" + "─"*45)
    print("STEP 2: RoBERTa SENTIMENT")
    print("Expected: ~10-12 min (GPU)")
    print("─"*45)

    roberta = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
        device=DEVICE
    )

    start   = time.time()
    total   = len(df)
    results = []

    for i, text in enumerate(df["full_text"]):
        try:
            r = roberta(str(text)[:512])[0]
            results.append((r["label"],
                            round(r["score"], 4)))
        except:
            results.append(("neutral", 0.0))

        if (i+1) % 1000 == 0:
            elapsed   = (time.time()-start)/60
            remaining = (elapsed/(i+1))*(total-i-1)
            print(f"  {i+1:,}/{total:,} | "
                  f"Elapsed: {elapsed:.1f}min | "
                  f"Remaining: {remaining:.1f}min")

    df["roberta_label"] = [r[0] for r in results]
    df["roberta_score"] = [r[1] for r in results]

    elapsed = round((time.time()-start)/60, 1)
    print(f"RoBERTa done in {elapsed} min!")
    print(df["roberta_label"].value_counts().to_string())
    df.to_csv(roberta_file, index=False)
    print("Saved: step2_roberta.csv")
else:
    print("\nRoBERTa: already done ")



if "emotion_label" not in df.columns:
    print("\n" + "─"*45)
    print("STEP 3: HARTMANN EMOTION")
    print("Expected: ~10-12 min (GPU)")
    print("─"*45)

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        truncation=True,
        max_length=512,
        device=DEVICE
    )

    start   = time.time()
    total   = len(df)
    results = []

    for i, text in enumerate(df["full_text"]):
        try:
            r = emotion_model(str(text)[:512])[0]
            results.append((r["label"],
                            round(r["score"], 4)))
        except:
            results.append(("neutral", 0.0))

        if (i+1) % 1000 == 0:
            elapsed   = (time.time()-start)/60
            remaining = (elapsed/(i+1))*(total-i-1)
            print(f"  {i+1:,}/{total:,} | "
                  f"Elapsed: {elapsed:.1f}min | "
                  f"Remaining: {remaining:.1f}min")

    df["emotion_label"] = [r[0] for r in results]
    df["emotion_score"] = [r[1] for r in results]

    elapsed = round((time.time()-start)/60, 1)
    print(f"Hartmann done in {elapsed} min!")
    print(df["emotion_label"].value_counts().to_string())
    df.to_csv(complete_file, index=False)
    print("Saved: RQ1_sentiment_complete.csv")
else:
    print("\nHartmann: already done ")


print("\n" + "─"*45)
print("STEP 4: HYPOTHESIS TEST + ITS")
print("─"*45)

df["date"] = pd.to_datetime(
    df["date"], dayfirst=True, errors="coerce")

pre_v  = df[df["era"]=="pre_genAI"][
    "vader_score"].dropna()
post_v = df[df["era"]=="post_genAI"][
    "vader_score"].dropna()

stat_v, p_v = stats.mannwhitneyu(
    pre_v, post_v, alternative="two-sided")

# Effect size
n1, n2 = len(pre_v), len(post_v)
r_effect = 1 - (2*stat_v)/(n1*n2)

print(f"\nMann-Whitney U Test:")
print(f"  Pre  mean : {pre_v.mean():.4f} (n={n1:,})")
print(f"  Post mean : {post_v.mean():.4f} (n={n2:,})")
print(f"  U statistic: {stat_v:.0f}")
print(f"  p-value   : {p_v:.4f}")
print(f"  Effect size (r): {r_effect:.4f}")
print(f"  Result    : " +
      ("H0 REJECTED" if p_v < 0.05
       else "H0 RETAINED"))

# ITS
df_its = df.dropna(
    subset=["date","vader_score"]
).sort_values("date").reset_index(drop=True)

df_its["time"]         = np.arange(len(df_its))
df_its["intervention"] = (
    df_its["date"] >= pd.Timestamp("2022-11-01")
).astype(int)
df_its["time_after"]   = (
    df_its["time"] * df_its["intervention"])

X = sm.add_constant(
    df_its[["time","intervention","time_after"]])
y = df_its["vader_score"].fillna(0)
its_model = sm.OLS(y, X).fit()

print(f"\nITS Regression:")
print(f"  β0 (intercept)   : "
      f"{its_model.params['const']:.4f}")
print(f"  β1 (time)        : "
      f"{its_model.params['time']:.6f}")
print(f"  β2 (intervention): "
      f"{its_model.params['intervention']:.4f} "
      f"(p={its_model.pvalues['intervention']:.4f})")
print(f"  β3 (time_after)  : "
      f"{its_model.params['time_after']:.6f} "
      f"(p={its_model.pvalues['time_after']:.4f})")

# Save stats
pd.DataFrame({
    "metric":  ["pre_mean","post_mean","U_stat",
                "p_value","effect_size_r",
                "its_intervention","its_time_after"],
    "value":   [pre_v.mean(), post_v.mean(),
                stat_v, p_v, r_effect,
                its_model.params["intervention"],
                its_model.params["time_after"]],
    "p_value": ["","","",p_v,"",
                its_model.pvalues["intervention"],
                its_model.pvalues["time_after"]]
}).to_csv(
    os.path.join(ANALYSIS_FOLDER,
                 "RQ1_stats_results.csv"),
    index=False
)
print("\nStats saved!")



print("\n" + "─"*45)
print("STEP 5: GENERATING GRAPHS (alag alag)")
print("─"*45)


fig, ax = plt.subplots(figsize=(8, 6))
counts  = df["vader_label"].value_counts()
colors  = ["#2ecc71","#e74c3c","#95a5a6"]
ax.pie(counts.values,
       labels=counts.index,
       colors=colors[:len(counts)],
       autopct="%1.1f%%",
       startangle=90,
       textprops={"fontsize":12})
ax.set_title("VADER: Overall Sentiment Distribution",
             fontweight="bold", fontsize=14,
             color="#2c3e50")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "01_VADER_overall_pie.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 1 saved: 01_VADER_overall_pie.png")


fig, ax = plt.subplots(figsize=(8, 6))
if "roberta_label" in df.columns:
    counts_r = df["roberta_label"].value_counts()
    colors_r = ["#3498db","#e67e22","#9b59b6"]
    ax.pie(counts_r.values,
           labels=counts_r.index,
           colors=colors_r[:len(counts_r)],
           autopct="%1.1f%%",
           startangle=90,
           textprops={"fontsize":12})
    ax.set_title("RoBERTa: Overall Sentiment Distribution",
                 fontweight="bold", fontsize=14,
                 color="#2c3e50")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "02_RoBERTa_overall_pie.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 2 saved: 02_RoBERTa_overall_pie.png")


fig, ax = plt.subplots(figsize=(10, 6))
if "emotion_label" in df.columns:
    ec = df["emotion_label"].value_counts()
    colors_e = {
        "anger":"#e74c3c","disgust":"#8e44ad",
        "fear":"#e67e22","joy":"#2ecc71",
        "sadness":"#3498db","surprise":"#f1c40f",
        "neutral":"#95a5a6","unknown":"#bdc3c7"
    }
    bc = [colors_e.get(e,"#bdc3c7")
          for e in ec.index]
    bars = ax.bar(ec.index, ec.values,
                  color=bc, edgecolor="white",
                  linewidth=1.2)
    ax.set_title("Hartmann: Emotion Distribution",
                 fontweight="bold", fontsize=14,
                 color="#2c3e50")
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_facecolor("#ffffff")
    for bar, val in zip(bars, ec.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                val+50, f"{val:,}",
                ha="center", fontsize=9,
                fontweight="bold")
    plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "03_Hartmann_emotion_bar.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 3 saved: 03_Hartmann_emotion_bar.png")

fig, ax = plt.subplots(figsize=(10, 6))
era_data = df.groupby(
    ["era","vader_label"]).size().unstack(fill_value=0)
for col in ["positive","negative","neutral"]:
    if col not in era_data.columns:
        era_data[col] = 0
era_data = era_data[["positive","negative","neutral"]]
x     = np.arange(len(era_data))
width = 0.25
b1 = ax.bar(x-width, era_data["positive"], width,
            label="Positive", color="#2ecc71",
            alpha=0.85)
b2 = ax.bar(x, era_data["negative"], width,
            label="Negative", color="#e74c3c",
            alpha=0.85)
b3 = ax.bar(x+width, era_data["neutral"], width,
            label="Neutral",  color="#95a5a6",
            alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(
    ["Pre-GenAI\n(Jan 2019–Oct 2022)",
     "Post-GenAI\n(Nov 2022–Dec 2025)"],
    fontsize=11)
ax.set_title("VADER: Sentiment Pre vs Post GenAI",
             fontweight="bold", fontsize=14,
             color="#2c3e50")
ax.set_ylabel("Number of Posts", fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor("#ffffff")
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x()+bar.get_width()/2,
                h+50, f"{int(h):,}",
                ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "04_VADER_pre_vs_post.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 4 saved: 04_VADER_pre_vs_post.png")


fig, ax = plt.subplots(figsize=(14, 6))
df_q = df.dropna(subset=["date"]).copy()
df_q["quarter"] = pd.PeriodIndex(
    df_q["date"], freq="Q")
quarterly = df_q.groupby(
    "quarter")["vader_score"].mean()
q_str = [str(q) for q in quarterly.index]

ax.plot(q_str, quarterly.values,
        marker="o", linewidth=2.5,
        color="#8e44ad", markersize=6,
        markerfacecolor="white",
        markeredgewidth=2)
ax.fill_between(q_str, quarterly.values,
                alpha=0.15, color="#8e44ad")
ax.axhline(y=0, color="gray",
           linestyle="--", alpha=0.4)

if "2022Q4" in q_str:
    idx = q_str.index("2022Q4")
    ax.axvline(x=idx, color="#e74c3c",
               linestyle="--", linewidth=2)
    ax.text(idx+0.2,
            min(quarterly.values)+0.02,
            "ChatGPT\nLaunch",
            fontsize=9, color="#e74c3c",
            fontweight="bold")

ax.set_title(
    "ITS: Sentiment Trend Over Time (Quarterly)",
    fontweight="bold", fontsize=14,
    color="#2c3e50")
ax.set_ylabel("Mean VADER Score", fontsize=12)
ax.set_xlabel("Quarter", fontsize=12)
ax.set_facecolor("#ffffff")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "05_ITS_sentiment_trend.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 5 saved: 05_ITS_sentiment_trend.png")


fig, ax = plt.subplots(figsize=(12, 6))
if "emotion_label" in df.columns:
    emo_era = df.groupby(
        ["era","emotion_label"]
    ).size().unstack(fill_value=0)
    emo_era.T.plot(
        kind="bar", ax=ax,
        color=["#3498db","#e74c3c"],
        alpha=0.85, edgecolor="white",
        width=0.7)
    ax.set_title(
        "Hartmann: Emotions Pre vs Post GenAI",
        fontweight="bold", fontsize=14,
        color="#2c3e50")
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_facecolor("#ffffff")
    ax.legend(["Pre-GenAI","Post-GenAI"],
              fontsize=11)
    plt.xticks(rotation=30, ha="right",
               fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "06_Emotion_pre_vs_post.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 6 saved: 06_Emotion_pre_vs_post.png")

fig, ax = plt.subplots(figsize=(10, 6))
if "phase" in df.columns:
    phase_vader = df.groupby(
        "phase")["vader_score"].mean()
    phase_colors = {
        "Phase1_PreAwareness": "#3498db",
        "Phase2_Disruption":   "#e74c3c",
        "Phase3_Normalisation":"#2ecc71"
    }
    colors_p = [phase_colors.get(p,"#95a5a6")
                for p in phase_vader.index]
    bars = ax.bar(
        ["Phase 1\nPre-Awareness",
         "Phase 2\nDisruption",
         "Phase 3\nNormalisation"],
        phase_vader.values,
        color=colors_p,
        edgecolor="white",
        linewidth=1.5,
        width=0.5
    )
    ax.set_title(
        "Mean VADER Score by Phase",
        fontweight="bold", fontsize=14,
        color="#2c3e50")
    ax.set_ylabel("Mean VADER Score",
                  fontsize=12)
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, phase_vader.values):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            val+0.005,
            f"{val:.3f}",
            ha="center", fontsize=11,
            fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "07_Phase_sentiment.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 7 saved: 07_Phase_sentiment.png")


fig, ax = plt.subplots(figsize=(10, 6))
if "roberta_label" in df.columns:
    rob_era = df.groupby(
        ["era","roberta_label"]
    ).size().unstack(fill_value=0)
    rob_era.T.plot(
        kind="bar", ax=ax,
        color=["#3498db","#e74c3c"],
        alpha=0.85, edgecolor="white",
        width=0.7)
    ax.set_title(
        "RoBERTa: Sentiment Pre vs Post GenAI",
        fontweight="bold", fontsize=14,
        color="#2c3e50")
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_facecolor("#ffffff")
    ax.legend(["Pre-GenAI","Post-GenAI"],
              fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER,
            "08_RoBERTa_pre_vs_post.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 8 saved: 08_RoBERTa_pre_vs_post.png")



print("\n" + "="*55)
print("RQ1 COMPLETE!")
print("="*55)
print(f"\nTotal posts analysed: {len(df):,}")
print(f"\nVADER Results:")
print(f"  Positive: "
      f"{(df['vader_label']=='positive').sum():,} "
      f"({(df['vader_label']=='positive').mean()*100:.1f}%)")
print(f"  Negative: "
      f"{(df['vader_label']=='negative').sum():,} "
      f"({(df['vader_label']=='negative').mean()*100:.1f}%)")
print(f"  Neutral : "
      f"{(df['vader_label']=='neutral').sum():,} "
      f"({(df['vader_label']=='neutral').mean()*100:.1f}%)")

if "emotion_label" in df.columns:
    top_emo = df["emotion_label"].value_counts()
    print(f"\nTop emotion: "
          f"{top_emo.index[0]} "
          f"({top_emo.values[0]:,})")

print(f"\nHypothesis: "
      f"{'REJECTED' if p_v < 0.05 else 'RETAINED'} "
      f"(p={p_v:.4f})")

print(f"\nFiles saved in:")
print(f"  {ANALYSIS_FOLDER}")
print(f"  {GRAPHS_FOLDER}")
