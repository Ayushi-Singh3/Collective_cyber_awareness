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

DATA_FOLDER     = r"D:\For Rajesh'sir\Result of code"
GRAPHS_FOLDER   = r"D:\For Rajesh'sir\Result of code\RQ1_Graphs"
ANALYSIS_FOLDER = r"D:\For Rajesh'sir\Result of code\RQ1_Analysis"

os.makedirs(GRAPHS_FOLDER,    exist_ok=True)
os.makedirs(ANALYSIS_FOLDER,  exist_ok=True)

INPUT_FILE    = os.path.join(DATA_FOLDER,     "corpus_50k_FINAL.csv")
OUTPUT_FILE   = os.path.join(ANALYSIS_FOLDER, "RQ1_sentiment_complete.csv")
STATS_FILE    = os.path.join(ANALYSIS_FOLDER, "RQ1_stats_results.csv")
vader_file    = os.path.join(ANALYSIS_FOLDER, "step1_vader.csv")
roberta_file  = os.path.join(ANALYSIS_FOLDER, "step2_roberta.csv")
complete_file = os.path.join(ANALYSIS_FOLDER, "RQ1_sentiment_complete.csv")

print("=" * 55)
print("RQ1 ANALYSIS")
print("VADER + RoBERTa + Hartmann + ITS + LIME")
print("=" * 55)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = 0
else:
    print("GPU not found — using CPU")
    DEVICE = -1

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
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

df["text"]      = df["text"].fillna("")
df["full_text"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
).str.strip()

print(f"Total posts: {len(df):,}")
print(df["era"].value_counts().to_string())
print(df["phase"].value_counts().sort_index().to_string())

if "vader_label" not in df.columns:
    print("\n" + "─"*45)
    print("STEP 1: VADER SENTIMENT")
    print("Thresholds: positive >= 0.05, negative <= -0.05")
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
    print("Saved: step1_vader.csv")
else:
    print("\nVADER: already done")

if "roberta_label" not in df.columns:
    print("\n" + "─"*45)
    print("STEP 2: RoBERTa SENTIMENT")
    print("Model: cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("Max tokens: 512, Truncation: head")
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
            results.append((r["label"], round(r["score"], 4)))
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
    print("\nRoBERTa: already done")

if "emotion_label" not in df.columns:
    print("\n" + "─"*45)
    print("STEP 3: HARTMANN EMOTION")
    print("Model: j-hartmann/emotion-english-distilroberta-base")
    print("7 classes: anger, disgust, fear, joy, neutral, sadness, surprise")
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
            results.append((r["label"], round(r["score"], 4)))
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
    print("\nHartmann: already done")

print("\n" + "─"*45)
print("STEP 4: HYPOTHESIS TEST + ITS")
print("─"*45)

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

pre_v  = df[df["era"]=="pre_genAI"]["vader_score"].dropna()
post_v = df[df["era"]=="post_genAI"]["vader_score"].dropna()

stat_v, p_v = stats.mannwhitneyu(pre_v, post_v, alternative="two-sided")
n1, n2      = len(pre_v), len(post_v)
r_effect    = 1 - (2*stat_v)/(n1*n2)

print(f"\nMann-Whitney U Test:")
print(f"  Pre  mean : {pre_v.mean():.4f} (n={n1:,})")
print(f"  Post mean : {post_v.mean():.4f} (n={n2:,})")
print(f"  U statistic: {stat_v:.0f}")
print(f"  p-value   : {p_v:.4f}")
print(f"  Effect size (r): {r_effect:.4f}")
print(f"  Result    : " +
      ("H0 REJECTED" if p_v < 0.05 else "H0 RETAINED"))

df_its = df.dropna(
    subset=["date", "vader_score"]
).sort_values("date").reset_index(drop=True)

df_its["time"]         = np.arange(len(df_its))
df_its["intervention"] = (
    df_its["date"] >= pd.Timestamp("2022-11-01")
).astype(int)
df_its["time_after"]   = df_its["time"] * df_its["intervention"]

X = sm.add_constant(df_its[["time", "intervention", "time_after"]])
y = df_its["vader_score"].fillna(0)
its_model = sm.OLS(y, X).fit()

print(f"\nITS Regression:")
print(f"  β0 (intercept)   : {its_model.params['const']:.4f}")
print(f"  β1 (time)        : {its_model.params['time']:.6f}")
print(f"  β2 (intervention): {its_model.params['intervention']:.4f} "
      f"(p={its_model.pvalues['intervention']:.4f})")
print(f"  β3 (time_after)  : {its_model.params['time_after']:.6f} "
      f"(p={its_model.pvalues['time_after']:.4f})")

print("\n" + "─"*45)
print("STEP 5: PHASE-WISE FEAR vs OPTIMISM")
print("─"*45)

if "emotion_label" in df.columns:
    phase_emotion = pd.crosstab(
        df["phase"], df["emotion_label"],
        normalize="index"
    ).mul(100).round(2)

    print("\nPhase-wise emotion %:")
    print(phase_emotion.to_string())

    phase_order = ["Phase1_PreAwareness",
                   "Phase2_Disruption",
                   "Phase3_Normalisation"]

    print("\n=== FEAR vs JOY ACROSS PHASES ===")
    for phase in phase_order:
        ph_df    = df[df["phase"] == phase]
        fear_pct = (ph_df["emotion_label"] == "fear").mean() * 100
        joy_pct  = (ph_df["emotion_label"] == "joy").mean() * 100
        print(f"  {phase}:")
        print(f"    Fear: {fear_pct:.1f}%")
        print(f"    Joy : {joy_pct:.1f}%")

    p1_fear = (df[df["phase"]=="Phase1_PreAwareness"]["emotion_label"]=="fear").mean()*100
    p2_fear = (df[df["phase"]=="Phase2_Disruption"]["emotion_label"]=="fear").mean()*100
    p1_joy  = (df[df["phase"]=="Phase1_PreAwareness"]["emotion_label"]=="joy").mean()*100
    p2_joy  = (df[df["phase"]=="Phase2_Disruption"]["emotion_label"]=="joy").mean()*100

    print(f"\n=== AFFECTIVE POLARISATION CHECK ===")
    print(f"Fear Phase1 to Phase2: {p1_fear:.1f}% to {p2_fear:.1f}% "
          f"({'UP' if p2_fear > p1_fear else 'DOWN'})")
    print(f"Joy  Phase1 to Phase2: {p1_joy:.1f}% to {p2_joy:.1f}% "
          f"({'UP' if p2_joy > p1_joy else 'DOWN'})")

    if p2_fear > p1_fear and p2_joy > p1_joy:
        print("AFFECTIVE POLARISATION CONFIRMED!")
    elif p2_fear > p1_fear:
        print("Partial polarisation: Fear rose, Joy did not")
    else:
        print("No clear polarisation pattern")

print("\n" + "─"*45)
print("STEP 6: LIME EXPLAINABILITY")
print("100 posts: 50 pre-GenAI + 50 post-GenAI")
print("200 perturbations, 10 features each")
print("─"*45)

try:
    from lime.lime_text import LimeTextExplainer

    lime_sample = pd.concat([
        df[df["era"]=="pre_genAI"].sample(
            min(50, len(df[df["era"]=="pre_genAI"])),
            random_state=42
        ),
        df[df["era"]=="post_genAI"].sample(
            min(50, len(df[df["era"]=="post_genAI"])),
            random_state=42
        )
    ])

    roberta_lime = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
        device=DEVICE,
        top_k=None
    )

    class_names = ["negative", "neutral", "positive"]

    def predict_proba(texts):
        results = []
        for text in texts:
            try:
                output = roberta_lime(str(text)[:512])[0]
                probs  = {item["label"].lower(): item["score"]
                          for item in output}
                results.append([
                    probs.get("negative", 0.0),
                    probs.get("neutral",  0.0),
                    probs.get("positive", 0.0)
                ])
            except:
                results.append([0.33, 0.34, 0.33])
        return np.array(results)

    explainer    = LimeTextExplainer(class_names=class_names)
    lime_results = []

    for i, (idx, row) in enumerate(lime_sample.iterrows()):
        try:
            exp = explainer.explain_instance(
                str(row["full_text"])[:512],
                predict_proba,
                num_features=10,
                num_samples=200
            )
            lime_results.append({
                "post_id":      row.get("id", i),
                "era":          row["era"],
                "phase":        row["phase"],
                "vader_label":  row.get("vader_label", ""),
                "top_features": str(exp.as_list()[:5])
            })
        except:
            continue

        if (i+1) % 10 == 0:
            print(f"  LIME: {i+1}/{len(lime_sample)} posts done")

    lime_df = pd.DataFrame(lime_results)
    lime_file = os.path.join(ANALYSIS_FOLDER, "RQ1_LIME_results.csv")
    lime_df.to_csv(lime_file, index=False)
    print(f"LIME done! {len(lime_df)} posts explained")

except ImportError:
    print("LIME not installed — run: pip install lime")

pd.DataFrame([{
    "total_posts":        len(df),
    "pre_posts":          n1,
    "post_posts":         n2,
    "pre_vader_mean":     round(pre_v.mean(), 4),
    "post_vader_mean":    round(post_v.mean(), 4),
    "U_statistic":        round(stat_v, 0),
    "p_value":            round(p_v, 4),
    "effect_size_r":      round(r_effect, 4),
    "beta0":              round(its_model.params["const"], 4),
    "beta1_time":         round(its_model.params["time"], 6),
    "beta2_intervention": round(its_model.params["intervention"], 4),
    "beta2_p":            round(its_model.pvalues["intervention"], 4),
    "beta3_time_after":   round(its_model.params["time_after"], 6),
    "beta3_p":            round(its_model.pvalues["time_after"], 4),
}]).to_csv(STATS_FILE, index=False)
print(f"\nStats saved: RQ1_stats_results.csv")

print("\n" + "─"*45)
print("STEP 7: GRAPHS")
print("─"*45)

fig, ax = plt.subplots(figsize=(8, 6))
counts  = df["vader_label"].value_counts()
colors  = ["#2ecc71", "#e74c3c", "#95a5a6"]
ax.pie(counts.values,
       labels=counts.index,
       colors=colors[:len(counts)],
       autopct="%1.1f%%",
       startangle=90,
       textprops={"fontsize": 12})
ax.set_title("VADER: Overall Sentiment Distribution",
             fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "01_VADER_overall_pie.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 1 saved")

if "roberta_label" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    counts_r = df["roberta_label"].value_counts()
    ax.pie(counts_r.values,
           labels=counts_r.index,
           colors=["#3498db", "#e67e22", "#9b59b6"][:len(counts_r)],
           autopct="%1.1f%%",
           startangle=90,
           textprops={"fontsize": 12})
    ax.set_title("RoBERTa: Overall Sentiment Distribution",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "02_RoBERTa_overall_pie.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph 2 saved")

if "emotion_label" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ec = df["emotion_label"].value_counts()
    colors_e = {"anger": "#e74c3c", "disgust": "#8e44ad",
                "fear": "#e67e22", "joy": "#2ecc71",
                "sadness": "#3498db", "surprise": "#f1c40f",
                "neutral": "#95a5a6"}
    bars = ax.bar(ec.index, ec.values,
                  color=[colors_e.get(e, "#bdc3c7") for e in ec.index],
                  edgecolor="white", linewidth=1.2)
    ax.set_title("Hartmann: Emotion Distribution",
                 fontweight="bold", fontsize=14)
    ax.set_ylabel("Number of Posts", fontsize=12)
    for bar, val in zip(bars, ec.values):
        ax.text(bar.get_x()+bar.get_width()/2,
                val+50, f"{val:,}",
                ha="center", fontsize=9, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "03_Hartmann_emotion_bar.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph 3 saved")

fig, ax = plt.subplots(figsize=(10, 6))
era_data = df.groupby(["era", "vader_label"]).size().unstack(fill_value=0)
for col in ["positive", "negative", "neutral"]:
    if col not in era_data.columns:
        era_data[col] = 0
era_data = era_data[["positive", "negative", "neutral"]]
x     = np.arange(len(era_data))
width = 0.25
b1 = ax.bar(x-width, era_data["positive"], width, label="Positive",
            color="#2ecc71", alpha=0.85)
b2 = ax.bar(x,        era_data["negative"], width, label="Negative",
            color="#e74c3c", alpha=0.85)
b3 = ax.bar(x+width,  era_data["neutral"],  width, label="Neutral",
            color="#95a5a6", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(["Pre-GenAI\n(Jan 2019-Oct 2022)",
                     "Post-GenAI\n(Nov 2022-Dec 2025)"],
                   fontsize=11)
ax.set_title("VADER: Sentiment Pre vs Post GenAI",
             fontweight="bold", fontsize=14)
ax.set_ylabel("Number of Posts", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "04_VADER_pre_vs_post.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 4 saved")

fig, ax = plt.subplots(figsize=(14, 6))
df_q = df.dropna(subset=["date"]).copy()
df_q["quarter"] = pd.PeriodIndex(df_q["date"], freq="Q")
quarterly = df_q.groupby("quarter")["vader_score"].mean()
q_str     = [str(q) for q in quarterly.index]
ax.plot(q_str, quarterly.values, marker="o", linewidth=2.5,
        color="#8e44ad", markersize=6,
        markerfacecolor="white", markeredgewidth=2)
ax.fill_between(q_str, quarterly.values, alpha=0.15, color="#8e44ad")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
if "2022Q4" in q_str:
    idx = q_str.index("2022Q4")
    ax.axvline(x=idx, color="#e74c3c", linestyle="--", linewidth=2)
    ax.text(idx+0.2, min(quarterly.values)+0.02,
            "ChatGPT\nLaunch", fontsize=9,
            color="#e74c3c", fontweight="bold")
ax.set_title("ITS: Sentiment Trend Over Time (Quarterly)",
             fontweight="bold", fontsize=14)
ax.set_ylabel("Mean VADER Score", fontsize=12)
ax.set_xlabel("Quarter", fontsize=12)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "05_ITS_sentiment_trend.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 5 saved")

if "emotion_label" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    emo_era = df.groupby(["era", "emotion_label"]).size().unstack(fill_value=0)
    emo_era.T.plot(kind="bar", ax=ax,
                   color=["#3498db", "#e74c3c"],
                   alpha=0.85, edgecolor="white", width=0.7)
    ax.set_title("Hartmann: Emotions Pre vs Post GenAI",
                 fontweight="bold", fontsize=14)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.legend(["Pre-GenAI", "Post-GenAI"], fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "06_Emotion_pre_vs_post.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph 6 saved")

if "phase" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    phase_vader = df.groupby("phase")["vader_score"].mean()
    phase_order_g = ["Phase1_PreAwareness",
                     "Phase2_Disruption",
                     "Phase3_Normalisation"]
    phase_labels_g = ["Phase 1\nAnticipation",
                      "Phase 2\nDisruption",
                      "Phase 3\nAdaptation"]
    phase_colors_g = ["#3498db", "#e74c3c", "#2ecc71"]
    vals  = [phase_vader.get(p, 0) for p in phase_order_g]
    bars  = ax.bar(phase_labels_g, vals,
                   color=phase_colors_g,
                   edgecolor="white", linewidth=1.5, width=0.5)
    ax.set_title("Mean VADER Score by Phase",
                 fontweight="bold", fontsize=14)
    ax.set_ylabel("Mean VADER Score", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                val+0.005, f"{val:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "07_Phase_sentiment.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph 7 saved")

if "emotion_label" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    phase_emo = pd.crosstab(
        df["phase"], df["emotion_label"], normalize="index"
    ).mul(100)
    phase_order_g  = ["Phase1_PreAwareness",
                      "Phase2_Disruption",
                      "Phase3_Normalisation"]
    phase_labels_g = ["Phase 1\nAnticipation",
                      "Phase 2\nDisruption",
                      "Phase 3\nAdaptation"]
    x     = np.arange(3)
    w     = 0.35
    fear_vals = [phase_emo.loc[p, "fear"]
                 if p in phase_emo.index and "fear" in phase_emo.columns
                 else 0 for p in phase_order_g]
    joy_vals  = [phase_emo.loc[p, "joy"]
                 if p in phase_emo.index and "joy" in phase_emo.columns
                 else 0 for p in phase_order_g]
    b1 = ax.bar(x - w/2, fear_vals, w, label="Fear",
                color="#e74c3c", alpha=0.85)
    b2 = ax.bar(x + w/2, joy_vals,  w, label="Joy/Optimism",
                color="#f1c40f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels_g, fontsize=11)
    ax.set_title("Fear vs Joy Across Phases",
                 fontweight="bold", fontsize=14)
    ax.set_ylabel("% of Posts in Phase", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2,
                    h+0.3, f"{h:.1f}%",
                    ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "08_Fear_vs_Joy_phases.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph 8 saved")

if "roberta_label" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    rob_era = df.groupby(["era", "roberta_label"]).size().unstack(fill_value=0)
    rob_era.T.plot(kind="bar", ax=ax,
                   color=["#3498db", "#e74c3c"],
                   alpha=0.85, edgecolor="white", width=0.7)
    ax.set_title("RoBERTa: Sentiment Pre vs Post GenAI",
                 fontweight="bold", fontsize=14)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.legend(["Pre-GenAI", "Post-GenAI"], fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "09_RoBERTa_pre_vs_post.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph 9 saved")

print("\n" + "="*55)
print("RQ1 COMPLETE!")
print("="*55)
print(f"Total posts: {len(df):,}")
print(f"\nVADER:")
print(f"  Positive: {(df['vader_label']=='positive').sum():,} "
      f"({(df['vader_label']=='positive').mean()*100:.1f}%)")
print(f"  Negative: {(df['vader_label']=='negative').sum():,} "
      f"({(df['vader_label']=='negative').mean()*100:.1f}%)")
print(f"  Neutral : {(df['vader_label']=='neutral').sum():,} "
      f"({(df['vader_label']=='neutral').mean()*100:.1f}%)")
if "emotion_label" in df.columns:
    print(f"\nTop emotion: {df['emotion_label'].value_counts().index[0]}")
print(f"\nMann-Whitney: p={p_v:.4f} "
      f"({'REJECTED' if p_v < 0.05 else 'RETAINED'})")
print(f"Effect size r: {r_effect:.4f}")
print(f"\nFiles: {ANALYSIS_FOLDER}")
print(f"Graphs: {GRAPHS_FOLDER}")