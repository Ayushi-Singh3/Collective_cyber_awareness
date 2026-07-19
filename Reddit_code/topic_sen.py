import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import warnings
warnings.filterwarnings("ignore")

RQ1_FOLDER    = r"D:\For Rajesh'sir\Result of code\RQ1_Analysis"
CORPUS_FILE   = r"D:\For Rajesh'sir\Result of code\corpus_50k_FINAL.csv"
OUTPUT_FOLDER = r"D:\For Rajesh'sir\Result of code\Topic_Sentiment"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

rq1    = pd.read_csv(rf"{RQ1_FOLDER}\RQ1_sentiment_complete.csv")
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


df = rq1[["id", "vader_score"]].merge(
    corpus[["id", "full_text", "year"]],
    on="id", how="inner"
)
print(f"Merged: {len(df):,}")


TOPICS = {
    "Ransomware"        : r"\bransomware\b|\bransomware attack\b|\blockbit\b|\brevil\b",
    "Certifications"    : r"\bcissp\b|\bccna\b|\boscp\b|\bcertification\b|\bsecurity\+\b|\bejpt\b|\bpnpt\b|\bceh\b",
    "Career and Jobs"   : r"\bcareer\b|\bjob\b|\bsoc analyst\b|\bhired\b|\bresume\b|\bsalary\b|\bentry level\b",
    "AI and Automation" : r"\bchatgpt\b|\bgpt\b|\bllm\b|\bartificial intelligence\b|\bai replace\b|\bai security\b|\bai tool\b|\bmachine learning\b",
}

def assign_topic(text):
    for topic, pattern in TOPICS.items():
        if re.search(pattern, str(text), re.IGNORECASE):
            return topic
    return None

print("Assigning topics...")
df["topic"] = df["full_text"].apply(assign_topic)

print("\nTopic distribution:")
print(df["topic"].value_counts().to_string())


years = list(range(2019, 2026))

results = {}
for topic in TOPICS.keys():
    topic_df = df[df["topic"] == topic]
    yearly   = []
    for y in years:
        year_data = topic_df[topic_df["year"] == y]["vader_score"].dropna()
        if len(year_data) >= 10:
            yearly.append(round(year_data.mean(), 4))
        else:
            yearly.append(np.nan)
    results[topic] = yearly
    print(f"{topic}: {yearly}")

# Save CSV
df_out = pd.DataFrame(results, index=years)
df_out.index.name = "Year"
df_out.to_csv(rf"{OUTPUT_FOLDER}\topic_sentiment_4lines.csv")
print("\nSaved CSV!")

# Plot — 4 lines one graph
fig, ax = plt.subplots(figsize=(12, 6))

# Simple distinct styles — no bright colors
styles = [
    {"color": "#000000", "linestyle": "-",    "marker": "o",  "label": "Ransomware"},
    {"color": "#444444", "linestyle": "--",   "marker": "s",  "label": "Certifications"},
    {"color": "#777777", "linestyle": "-.",   "marker": "^",  "label": "Career and Jobs"},
    {"color": "#aaaaaa", "linestyle": ":",    "marker": "D",  "label": "AI and Automation"},
]

for idx, (topic, values) in enumerate(results.items()):
    s = styles[idx]
    ax.plot(years, values,
            color=s["color"],
            linestyle=s["linestyle"],
            marker=s["marker"],
            linewidth=2.0,
            markersize=7,
            label=s["label"])

# ChatGPT launch vertical line
ax.axvline(x=2022, color="red",
           linestyle="dotted",
           linewidth=1.5,
           label="ChatGPT launch (2022)")

# Zero line
ax.axhline(y=0, color="#dddddd",
           linewidth=0.8, linestyle="-")

# Formatting
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], fontsize=11)
ax.set_ylabel("Mean VADER Sentiment Score", fontsize=12)
ax.set_xlabel("Year", fontsize=12)
ax.set_title(
    "Year-wise Sentiment Trends by Topic\n"
    "Reddit Cybersecurity Communities (2019–2025)",
    fontsize=12, fontweight="bold", pad=12
)

ax.legend(loc="upper center",
          bbox_to_anchor=(0.5, -0.18),
          ncol=4,
          fontsize=11,
          framealpha=0.95,
          edgecolor="#dddddd")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.grid(axis="y", alpha=0.25, color="#dddddd")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
out = rf"{OUTPUT_FOLDER}\topic_sentiment_4lines.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out}")
print("Done!")