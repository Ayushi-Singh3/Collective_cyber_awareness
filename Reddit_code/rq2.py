import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

DATA_FOLDER     = r"D:\For Rajesh'sir\Result of code"
GRAPHS_FOLDER   = r"D:\For Rajesh'sir\Result of code\RQ2_Graphs"
ANALYSIS_FOLDER = r"D:\For Rajesh'sir\Result of code\RQ2_Analysis"

os.makedirs(GRAPHS_FOLDER,    exist_ok=True)
os.makedirs(ANALYSIS_FOLDER,  exist_ok=True)

INPUT_FILE = os.path.join(DATA_FOLDER, "corpus_50k_FINAL.csv")

print("=" * 55)
print("RQ2 ANALYSIS — BERTopic")
print("Topic modelling pre vs post GenAI")
print("=" * 55)

# ── Load data 
df = pd.read_csv(INPUT_FILE)
df["full_text"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
).str.strip()

df = df[df["full_text"].str.len() >= 20].copy()

print(f"Total posts: {len(df):,}")
print(df["era"].value_counts().to_string())
print(df["phase"].value_counts().sort_index().to_string())

# ── Split pre and post GenAI 
pre_df  = df[df["era"] == "pre_genAI"].copy()
post_df = df[df["era"] == "post_genAI"].copy()

print(f"\nPre-GenAI : {len(pre_df):,} posts")
print(f"Post-GenAI: {len(post_df):,} posts")

pre_docs  = pre_df["full_text"].tolist()
post_docs = post_df["full_text"].tolist()

# ── BERTopic configuration 
print("\n" + "─"*45)
print("BERTopic Configuration:")
print("  Embedding : all-MiniLM-L6-v2")
print("  UMAP      : n_neighbors=15, n_components=5")
print("  HDBSCAN   : min_cluster_size=30")
print("  Topics    : auto")
print("─"*45)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors  = 15,
    n_components = 5,
    min_dist     = 0.0,
    metric       = "cosine",
    random_state = 42
)

hdbscan_model = HDBSCAN(
    min_cluster_size    = 30,
    metric              = "euclidean",
    cluster_selection_method = "eom",
    prediction_data     = True
)

vectorizer_model = CountVectorizer(
    stop_words = "english",
    min_df     = 2,
    ngram_range = (1, 2)
)

# ── Run BERTopic on Pre-GenAI 
print("\nRunning BERTopic on Pre-GenAI corpus...")
topic_model_pre = BERTopic(
    embedding_model  = embedding_model,
    umap_model       = umap_model,
    hdbscan_model    = hdbscan_model,
    vectorizer_model = vectorizer_model,
    nr_topics        = "auto",
    calculate_probabilities = False,
    verbose          = True
)

topics_pre, _ = topic_model_pre.fit_transform(pre_docs)

pre_info = topic_model_pre.get_topic_info()
print(f"\nPre-GenAI topics found: {len(pre_info[pre_info['Topic'] != -1])}")
print("\nTop 15 Pre-GenAI topics:")
print(pre_info[pre_info['Topic'] != -1].head(15)[
    ['Topic', 'Count', 'Name']
].to_string())

pre_info.to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ2_pre_topics.csv"),
    index=False
)

# ── Run BERTopic on Post-GenAI 
print("\nRunning BERTopic on Post-GenAI corpus...")

umap_model2 = UMAP(
    n_neighbors  = 15,
    n_components = 5,
    min_dist     = 0.0,
    metric       = "cosine",
    random_state = 42
)

hdbscan_model2 = HDBSCAN(
    min_cluster_size    = 30,
    metric              = "euclidean",
    cluster_selection_method = "eom",
    prediction_data     = True
)

topic_model_post = BERTopic(
    embedding_model  = embedding_model,
    umap_model       = umap_model2,
    hdbscan_model    = hdbscan_model2,
    vectorizer_model = vectorizer_model,
    nr_topics        = "auto",
    calculate_probabilities = False,
    verbose          = True
)

topics_post, _ = topic_model_post.fit_transform(post_docs)

post_info = topic_model_post.get_topic_info()
print(f"\nPost-GenAI topics found: {len(post_info[post_info['Topic'] != -1])}")
print("\nTop 15 Post-GenAI topics:")
print(post_info[post_info['Topic'] != -1].head(15)[
    ['Topic', 'Count', 'Name']
].to_string())

post_info.to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ2_post_topics.csv"),
    index=False
)

# ── Save top keywords per topic
print("\nSaving top keywords per topic...")

pre_keywords = []
for topic_id in pre_info[pre_info['Topic'] != -1]['Topic'].tolist():
    words = topic_model_pre.get_topic(topic_id)
    if words:
        pre_keywords.append({
            'topic_id'  : topic_id,
            'count'     : pre_info[pre_info['Topic']==topic_id]['Count'].values[0],
            'top_words' : ', '.join([w[0] for w in words[:10]])
        })

pd.DataFrame(pre_keywords).to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ2_pre_keywords.csv"),
    index=False
)

post_keywords = []
for topic_id in post_info[post_info['Topic'] != -1]['Topic'].tolist():
    words = topic_model_post.get_topic(topic_id)
    if words:
        post_keywords.append({
            'topic_id'  : topic_id,
            'count'     : post_info[post_info['Topic']==topic_id]['Count'].values[0],
            'top_words' : ', '.join([w[0] for w in words[:10]])
        })

pd.DataFrame(post_keywords).to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ2_post_keywords.csv"),
    index=False
)

# ── Topic comparison
print("\n" + "─"*45)
print("TOPIC COMPARISON: Pre vs Post GenAI")
print("─"*45)

pre_words_set  = set()
post_words_set = set()

for item in pre_keywords:
    for w in item['top_words'].split(', '):
        pre_words_set.add(w.strip())

for item in post_keywords:
    for w in item['top_words'].split(', '):
        post_words_set.add(w.strip())

emerged = post_words_set - pre_words_set
faded   = pre_words_set  - post_words_set
shared  = pre_words_set  & post_words_set

print(f"\nNew terms in Post-GenAI (emerged): {len(emerged)}")
print(f"Terms only in Pre-GenAI (faded)  : {len(faded)}")
print(f"Shared terms (persistent)        : {len(shared)}")

# GenAI specific terms
genai_terms = [
    'chatgpt', 'gpt', 'openai', 'claude', 'gemini',
    'llm', 'generative', 'ai', 'deepfake', 'prompt',
    'injection', 'phishing', 'automation'
]

print("\nGenAI-related terms in Post-GenAI topics:")
for t in genai_terms:
    if t in post_words_set:
        print(f"  ✓ {t}")

# ── Graph 1 — Pre topic distribution 
fig, ax = plt.subplots(figsize=(12, 8))
pre_plot = pre_info[pre_info['Topic'] != -1].head(15)
bars = ax.barh(
    range(len(pre_plot)),
    pre_plot['Count'].values,
    color='#3498db', alpha=0.85
)
ax.set_yticks(range(len(pre_plot)))
ax.set_yticklabels([f"Topic {r['Topic']}: {r['Name'][:40]}"
                    for _, r in pre_plot.iterrows()],
                   fontsize=9)
ax.set_title("Top 15 Topics — Pre-GenAI (Anticipation Phase)",
             fontweight="bold", fontsize=14)
ax.set_xlabel("Number of Posts", fontsize=12)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, pre_plot['Count'].values):
    ax.text(val + 10, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "01_Pre_topics.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("\nGraph 1 saved")

# ── Graph 2 — Post topic distribution
fig, ax = plt.subplots(figsize=(12, 8))
post_plot = post_info[post_info['Topic'] != -1].head(15)
bars = ax.barh(
    range(len(post_plot)),
    post_plot['Count'].values,
    color='#e74c3c', alpha=0.85
)
ax.set_yticks(range(len(post_plot)))
ax.set_yticklabels([f"Topic {r['Topic']}: {r['Name'][:40]}"
                    for _, r in post_plot.iterrows()],
                   fontsize=9)
ax.set_title("Top 15 Topics — Post-GenAI (Disruption + Adaptation)",
             fontweight="bold", fontsize=14)
ax.set_xlabel("Number of Posts", fontsize=12)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, post_plot['Count'].values):
    ax.text(val + 10, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "02_Post_topics.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 2 saved")

# ── Graph 3 — Topic count comparison
fig, ax = plt.subplots(figsize=(8, 5))
counts = [
    len(pre_info[pre_info['Topic']  != -1]),
    len(post_info[post_info['Topic'] != -1])
]
labels = ['Pre-GenAI\n(Anticipation)', 'Post-GenAI\n(Disruption + Adaptation)']
colors = ['#3498db', '#e74c3c']
bars   = ax.bar(labels, counts, color=colors,
                edgecolor='white', linewidth=1.5, width=0.5)
ax.set_title("Number of Distinct Topics: Pre vs Post GenAI",
             fontweight="bold", fontsize=14)
ax.set_ylabel("Number of Topics", fontsize=12)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.3, str(val),
            ha='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "03_Topic_count_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 3 saved")

# ── Summary 
print("\n" + "="*55)
print("RQ2 COMPLETE!")
print("="*55)
print(f"Pre-GenAI topics  : {len(pre_info[pre_info['Topic']  != -1])}")
print(f"Post-GenAI topics : {len(post_info[post_info['Topic'] != -1])}")
print(f"Emerged terms     : {len(emerged)}")
print(f"Faded terms       : {len(faded)}")
print(f"Shared terms      : {len(shared)}")
print(f"\nFiles: {ANALYSIS_FOLDER}")
print(f"Graphs: {GRAPHS_FOLDER}")
print("\nNext step:")
print("  Upload RQ2_pre_keywords.csv + RQ2_post_keywords.csv")
print("  Manually label each topic")
print("  Then write 4.4 + Section 5.2")