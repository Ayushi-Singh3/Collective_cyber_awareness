import pandas as pd
import numpy as np
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA_FOLDER     = r"D:\For Rajesh'sir\Result of code"
GRAPHS_FOLDER   = r"D:\For Rajesh'sir\Result of code\RQ3_Graphs"
ANALYSIS_FOLDER = r"D:\For Rajesh'sir\Result of code\RQ3_Analysis"

os.makedirs(GRAPHS_FOLDER,    exist_ok=True)
os.makedirs(ANALYSIS_FOLDER,  exist_ok=True)

INPUT_FILE = os.path.join(DATA_FOLDER, "corpus_50k_FINAL.csv")

print("=" * 55)
print("RQ3 ANALYSIS — NER + Keyword Extraction")
print("Certification and threat entity detection")
print("=" * 55)

# ── Load data 
df = pd.read_csv(INPUT_FILE)
df["full_text"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
).str.strip()
df["full_text_lower"] = df["full_text"].str.lower()

print(f"Total posts: {len(df):,}")
print(df["era"].value_counts().to_string())
print(df["phase"].value_counts().sort_index().to_string())


CERTIFICATIONS = {
    "CISSP":        [r"\bcissp\b"],
    "CompTIA Security+": [r"\bsecurity\+\b", r"\bsecurity plus\b", r"\bsec\+\b"],
    "CEH":          [r"\bceh\b", r"\bcertified ethical hacker\b"],
    "OSCP":         [r"\boscp\b", r"\boffensive security\b"],
    "CISM":         [r"\bcism\b"],
    "CISA":         [r"\bcisa\b"],
    "CompTIA CySA+": [r"\bcysa\+\b", r"\bcysa plus\b"],
    "CompTIA PenTest+": [r"\bpentest\+\b", r"\bpentest plus\b"],
    "CCNA":         [r"\bccna\b"],
    "CCNP":         [r"\bccnp\b"],
    "AWS Security": [r"\baws security\b", r"\baws certified security\b"],
    "Azure Security": [r"\bazure security\b", r"\bmicrosoft security\b"],
    "Google Cloud Security": [r"\bgoogle cloud security\b", r"\bgcp security\b"],
    "GIAC":         [r"\bgiac\b", r"\bgcih\b", r"\bgcfa\b"],
    "eJPT":         [r"\bejpt\b"],
    "CompTIA A+":   [r"\ba\+\b", r"\bcompTIA a\b"],
    "CompTIA Network+": [r"\bnetwork\+\b", r"\bnetwork plus\b"],
    "PNPT":         [r"\bpnpt\b"],
    "CDSA":         [r"\bcdsa\b"],
    "SOC":          [r"\bsoc analyst\b", r"\bsoc certification\b"],
}


THREATS = {
    "Ransomware":       [r"\bransomware\b"],
    "Phishing":         [r"\bphishing\b", r"\bspear phishing\b"],
    "AI Phishing":      [r"\bai phishing\b", r"\bai.generated phishing\b", r"\bllm phishing\b"],
    "Deepfake":         [r"\bdeepfake\b", r"\bdeep fake\b", r"\bai voice\b", r"\bvoice cloning\b"],
    "Prompt Injection": [r"\bprompt injection\b", r"\bjailbreak\b"],
    "Data Breach":      [r"\bdata breach\b", r"\bdata leak\b"],
    "Malware":          [r"\bmalware\b", r"\bvirus\b", r"\btrojan\b"],
    "Social Engineering": [r"\bsocial engineering\b", r"\bvishing\b", r"\bsmishing\b"],
    "Zero Day":         [r"\bzero.day\b", r"\b0day\b"],
    "Supply Chain":     [r"\bsupply chain attack\b", r"\bsoftware supply chain\b"],
    "DDoS":             [r"\bddos\b", r"\bdistributed denial\b"],
    "Insider Threat":   [r"\binsider threat\b", r"\binsider attack\b"],
    "AI Malware":       [r"\bai malware\b", r"\bai.generated malware\b", r"\bgpt malware\b"],
    "Vulnerability":    [r"\bvulnerability\b", r"\bcve\b", r"\bexploit\b"],
    "Nation State":     [r"\bnation.state\b", r"\bapt\b", r"\bstate.sponsored\b"],
}


def extract_entities(df, entity_dict, entity_type):
    results = []
    for entity_name, patterns in entity_dict.items():
        combined_pattern = "|".join(patterns)
        mask = df["full_text_lower"].str.contains(
            combined_pattern, regex=True, na=False
        )
        matched = df[mask].copy()
        matched["entity"]      = entity_name
        matched["entity_type"] = entity_type
        results.append(matched)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

print("\nExtracting certification entities...")
cert_df = extract_entities(df, CERTIFICATIONS, "certification")
print(f"Certification mentions: {len(cert_df):,}")

print("Extracting threat entities...")
threat_df = extract_entities(df, THREATS, "threat")
print(f"Threat mentions: {len(threat_df):,}")


print("\n" + "─"*45)
print("CERTIFICATION ANALYSIS")
print("─"*45)

# Overall counts
cert_counts = cert_df.groupby("entity").size().sort_values(ascending=False)
print("\nOverall certification mentions:")
print(cert_counts.to_string())

# Pre vs Post GenAI
cert_era = cert_df.groupby(["entity", "era"]).size().unstack(fill_value=0)
cert_era["total"]   = cert_era.sum(axis=1)
cert_era["pct_change"] = (
    (cert_era.get("post_genAI", 0) - cert_era.get("pre_genAI", 0)) /
    cert_era.get("pre_genAI", 1) * 100
).round(1)
cert_era = cert_era.sort_values("total", ascending=False)

print("\nCertification pre vs post GenAI:")
print(cert_era.to_string())

cert_era.to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ3_cert_era.csv")
)

# Phase wise
cert_phase = cert_df.groupby(["entity","phase"]).size().unstack(fill_value=0)
cert_phase.to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ3_cert_phase.csv")
)


print("\n" + "─"*45)
print("THREAT ANALYSIS")
print("─"*45)

threat_counts = threat_df.groupby("entity").size().sort_values(ascending=False)
print("\nOverall threat mentions:")
print(threat_counts.to_string())

threat_era = threat_df.groupby(["entity","era"]).size().unstack(fill_value=0)
threat_era["total"]     = threat_era.sum(axis=1)
threat_era["pct_change"] = (
    (threat_era.get("post_genAI", 0) - threat_era.get("pre_genAI", 0)) /
    threat_era.get("pre_genAI", 1) * 100
).round(1)
threat_era = threat_era.sort_values("total", ascending=False)

print("\nThreat pre vs post GenAI:")
print(threat_era.to_string())

threat_era.to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ3_threat_era.csv")
)

threat_phase = threat_df.groupby(["entity","phase"]).size().unstack(fill_value=0)
threat_phase.to_csv(
    os.path.join(ANALYSIS_FOLDER, "RQ3_threat_phase.csv")
)


print("\n" + "─"*45)
print("AI-SPECIFIC THREATS (Post-GenAI only)")
print("─"*45)

ai_threats = ["AI Phishing", "Deepfake", "Prompt Injection", "AI Malware"]
for t in ai_threats:
    if t in threat_era.index:
        pre  = threat_era.loc[t, "pre_genAI"]  if "pre_genAI"  in threat_era.columns else 0
        post = threat_era.loc[t, "post_genAI"] if "post_genAI" in threat_era.columns else 0
        print(f"  {t}: Pre={pre}, Post={post}, Change={threat_era.loc[t,'pct_change']}%")


print("\n" + "─"*45)
print("GENERATING GRAPHS")
print("─"*45)

# Graph 1 — Top certifications overall
fig, ax = plt.subplots(figsize=(12, 7))
top_certs = cert_counts.head(15)
colors    = ['#2ecc71' if i < 5 else '#3498db' for i in range(len(top_certs))]
bars = ax.barh(
    range(len(top_certs)),
    top_certs.values,
    color=colors, alpha=0.85
)
ax.set_yticks(range(len(top_certs)))
ax.set_yticklabels(top_certs.index, fontsize=11)
ax.set_title("Top Certifications Mentioned — Full Corpus",
             fontweight="bold", fontsize=14)
ax.set_xlabel("Number of Posts", fontsize=12)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, top_certs.values):
    ax.text(val + 5, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "01_Cert_overall.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 1 saved")

# Graph 2 — Certifications Pre vs Post GenAI
fig, ax = plt.subplots(figsize=(12, 8))
top10 = cert_era.head(10)
x     = np.arange(len(top10))
w     = 0.35
pre_vals  = top10.get("pre_genAI",  pd.Series([0]*len(top10))).values
post_vals = top10.get("post_genAI", pd.Series([0]*len(top10))).values
b1 = ax.bar(x - w/2, pre_vals,  w, label="Pre-GenAI",  color="#3498db", alpha=0.85)
b2 = ax.bar(x + w/2, post_vals, w, label="Post-GenAI", color="#e74c3c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(top10.index, rotation=30, ha='right', fontsize=10)
ax.set_title("Certification Mentions: Pre vs Post GenAI",
             fontweight="bold", fontsize=14)
ax.set_ylabel("Number of Posts", fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "02_Cert_pre_vs_post.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 2 saved")

# Graph 3 — Top threats overall
fig, ax = plt.subplots(figsize=(12, 7))
top_threats = threat_counts.head(15)
colors_t    = ['#e74c3c' if i < 5 else '#e67e22' for i in range(len(top_threats))]
bars = ax.barh(
    range(len(top_threats)),
    top_threats.values,
    color=colors_t, alpha=0.85
)
ax.set_yticks(range(len(top_threats)))
ax.set_yticklabels(top_threats.index, fontsize=11)
ax.set_title("Top Threats Mentioned — Full Corpus",
             fontweight="bold", fontsize=14)
ax.set_xlabel("Number of Posts", fontsize=12)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, top_threats.values):
    ax.text(val + 5, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "03_Threat_overall.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 3 saved")

# Graph 4 — Threats Pre vs Post GenAI
fig, ax = plt.subplots(figsize=(12, 8))
top10t    = threat_era.head(10)
x         = np.arange(len(top10t))
pre_vals  = top10t.get("pre_genAI",  pd.Series([0]*len(top10t))).values
post_vals = top10t.get("post_genAI", pd.Series([0]*len(top10t))).values
b1 = ax.bar(x - w/2, pre_vals,  w, label="Pre-GenAI",  color="#3498db", alpha=0.85)
b2 = ax.bar(x + w/2, post_vals, w, label="Post-GenAI", color="#e74c3c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(top10t.index, rotation=30, ha='right', fontsize=10)
ax.set_title("Threat Mentions: Pre vs Post GenAI",
             fontweight="bold", fontsize=14)
ax.set_ylabel("Number of Posts", fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "04_Threat_pre_vs_post.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 4 saved")

# Graph 5 — AI-specific threats phase wise
fig, ax = plt.subplots(figsize=(10, 6))
ai_threat_df = threat_df[threat_df["entity"].isin(ai_threats)]
if len(ai_threat_df) > 0:
    ai_phase = ai_threat_df.groupby(["entity","phase"]).size().unstack(fill_value=0)
    phase_order = ["Phase1_PreAwareness","Phase2_Disruption","Phase3_Normalisation"]
    phase_labels = ["Phase 1\nAnticipation","Phase 2\nDisruption","Phase 3\nAdaptation"]
    ai_phase = ai_phase.reindex(columns=[p for p in phase_order if p in ai_phase.columns])
    ai_phase.columns = phase_labels[:len(ai_phase.columns)]
    ai_phase.T.plot(kind='bar', ax=ax, alpha=0.85, edgecolor='white', width=0.7)
    ax.set_title("AI-Specific Threats Across Phases",
                 fontweight="bold", fontsize=14)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.legend(fontsize=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_FOLDER, "05_AI_threats_phases.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Graph 5 saved")

# Graph 6 — Certification phase wise
fig, ax = plt.subplots(figsize=(12, 6))
top5_certs = cert_counts.head(5).index.tolist()
cert5_phase = cert_df[cert_df["entity"].isin(top5_certs)].groupby(
    ["entity","phase"]
).size().unstack(fill_value=0)
phase_order = ["Phase1_PreAwareness","Phase2_Disruption","Phase3_Normalisation"]
phase_labels = ["Phase 1\nAnticipation","Phase 2\nDisruption","Phase 3\nAdaptation"]
cert5_phase = cert5_phase.reindex(
    columns=[p for p in phase_order if p in cert5_phase.columns]
)
cert5_phase.columns = phase_labels[:len(cert5_phase.columns)]
cert5_phase.T.plot(kind='bar', ax=ax, alpha=0.85, edgecolor='white', width=0.7)
ax.set_title("Top 5 Certifications Across Phases",
             fontweight="bold", fontsize=14)
ax.set_ylabel("Number of Posts", fontsize=12)
ax.legend(fontsize=10)
plt.xticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_FOLDER, "06_Cert_phases.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Graph 6 saved")


print("\n" + "="*55)
print("RQ3 COMPLETE!")
print("="*55)
print(f"Total posts          : {len(df):,}")
print(f"Certification mentions: {len(cert_df):,}")
print(f"Threat mentions      : {len(threat_df):,}")
print(f"Certifications tracked: {len(CERTIFICATIONS)}")
print(f"Threats tracked      : {len(THREATS)}")
print(f"\nFiles: {ANALYSIS_FOLDER}")
print(f"Graphs: {GRAPHS_FOLDER}")
print("\nUpload these files:")
print("  RQ3_cert_era.csv")
print("  RQ3_threat_era.csv")
print("  RQ3_cert_phase.csv")
print("  RQ3_threat_phase.csv")