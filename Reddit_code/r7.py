import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

DATA_FOLDER     = r"D:\For Rajesh'sir\Result of code"
ANALYSIS_FOLDER = r"D:\For Rajesh'sir\Result of code\RQ1_Analysis"
RQ3_FOLDER      = r"D:\For Rajesh'sir\Result of code\RQ3_Analysis"

print("=" * 55)
print("R7 — SENSITIVITY ANALYSIS")
print("AI subreddits excluded — RQ2 + RQ3 robustness")
print("=" * 55)

df = pd.read_csv(rf"{ANALYSIS_FOLDER}\RQ1_sentiment_complete.csv")
df["full_text"] = df["full_text"].fillna("").str.lower()

print(f"Full corpus: {len(df):,}")
print(f"Subreddits: {df['subreddit'].unique().tolist()}")


ai_subs     = ['ChatGPT', 'OpenAI', 'artificial']
cybersec_df = df[~df['subreddit'].isin(ai_subs)].copy()

print(f"\nCybersec-only corpus: {len(cybersec_df):,}")
print(f"Removed AI posts    : {len(df)-len(cybersec_df):,}")
print(f"Subreddits kept     : {cybersec_df['subreddit'].unique().tolist()}")


CERTIFICATIONS = {
    "SOC"            : r"\bsoc\b",
    "CCNA"           : r"\bccna\b",
    "CISSP"          : r"\bcissp\b",
    "OSCP"           : r"\boscp\b",
    "CEH"            : r"\bceh\b",
    "CISA"           : r"\bcisa\b",
    "GIAC"           : r"\bgiac\b",
    "CISM"           : r"\bcism\b",
    "eJPT"           : r"\bejpt\b",
    "CCNP"           : r"\bccnp\b",
    "CompTIA Sec+"   : r"\bsecurity\+\b|\bsec\+\b",
    "Azure Security" : r"\bazure security\b",
    "AWS Security"   : r"\baws security\b",
    "PNPT"           : r"\bpnpt\b",
    "eJPT"           : r"\bejpt\b",
}

THREATS = {
    "Vulnerability"     : r"\bvulnerability\b|\bcve\b",
    "Data Breach"       : r"\bdata breach\b|\bdata leak\b",
    "Ransomware"        : r"\bransomware\b",
    "Malware"           : r"\bmalware\b",
    "Phishing"          : r"\bphishing\b",
    "Prompt Injection"  : r"\bprompt injection\b|\bjailbreak\b",
    "Deepfake"          : r"\bdeepfake\b|\bdeep fake\b",
    "AI Phishing"       : r"\bai phishing\b|\bai.generated phishing\b",
    "Supply Chain"      : r"\bsupply chain attack\b",
}

def count_entity(df_subset, pattern):
    return df_subset["full_text"].str.contains(
        pattern, regex=True, na=False
    ).sum()

# ── CERT comparison ────────────────────────────────────────
print("\n" + "─"*55)
print("CERTIFICATION COUNTS — Full vs Cybersec-only")
print("─"*55)
print(f"{'Cert':<18} {'Full Pre':>9} {'Full Post':>10} {'CS Pre':>8} {'CS Post':>9} {'Same Dir?':>10}")

cert_results = []
for cert, pattern in CERTIFICATIONS.items():
    full_pre  = count_entity(df[df['era']=='pre_genAI'],  pattern)
    full_post = count_entity(df[df['era']=='post_genAI'], pattern)
    cs_pre    = count_entity(cybersec_df[cybersec_df['era']=='pre_genAI'],  pattern)
    cs_post   = count_entity(cybersec_df[cybersec_df['era']=='post_genAI'], pattern)

    full_dir = "UP" if full_post > full_pre else "DOWN"
    cs_dir   = "UP" if cs_post  > cs_pre  else "DOWN"
    same     = "✓" if full_dir == cs_dir else "✗"

    print(f"{cert:<18} {full_pre:>9} {full_post:>10} {cs_pre:>8} {cs_post:>9} {same:>10}")
    cert_results.append({
        "cert"     : cert,
        "full_pre" : full_pre,
        "full_post": full_post,
        "cs_pre"   : cs_pre,
        "cs_post"  : cs_post,
        "same_dir" : same
    })

pd.DataFrame(cert_results).to_csv(
    rf"{RQ3_FOLDER}\R7_cert_sensitivity.csv", index=False
)


print("\n" + "─"*55)
print("THREAT COUNTS — Full vs Cybersec-only")
print("─"*55)
print(f"{'Threat':<20} {'Full Pre':>9} {'Full Post':>10} {'CS Pre':>8} {'CS Post':>9} {'Same Dir?':>10}")

threat_results = []
for threat, pattern in THREATS.items():
    full_pre  = count_entity(df[df['era']=='pre_genAI'],  pattern)
    full_post = count_entity(df[df['era']=='post_genAI'], pattern)
    cs_pre    = count_entity(cybersec_df[cybersec_df['era']=='pre_genAI'],  pattern)
    cs_post   = count_entity(cybersec_df[cybersec_df['era']=='post_genAI'], pattern)

    full_dir = "UP" if full_post > full_pre else "DOWN"
    cs_dir   = "UP" if cs_post  > cs_pre  else "DOWN"
    same     = "✓" if full_dir == cs_dir else "✗"

    print(f"{threat:<20} {full_pre:>9} {full_post:>10} {cs_pre:>8} {cs_post:>9} {same:>10}")
    threat_results.append({
        "threat"   : threat,
        "full_pre" : full_pre,
        "full_post": full_post,
        "cs_pre"   : cs_pre,
        "cs_post"  : cs_post,
        "same_dir" : same
    })

pd.DataFrame(threat_results).to_csv(
    rf"{RQ3_FOLDER}\R7_threat_sensitivity.csv", index=False
)

print("\n" + "─"*55)
print("RQ1 SENSITIVITY (already computed):")
print("─"*55)
print("Full corpus   : p = 0.127, r = -0.009 — H0 RETAINED")
print("Cybersec-only : p = 0.911, r = +0.001 — H0 RETAINED")
print("Conclusion    : RQ1 finding robust to AI subreddit exclusion")


cert_same    = sum(1 for r in cert_results   if r['same_dir'] == '✓')
threat_same  = sum(1 for r in threat_results if r['same_dir'] == '✓')

print("\n" + "="*55)
print("SUMMARY FOR PAPER")
print("="*55)
print(f"""
Cybersec-only corpus: {len(cybersec_df):,} posts
Removed AI posts    : {len(df)-len(cybersec_df):,} posts

Certification directional agreement: {cert_same}/{len(cert_results)}
Threat directional agreement       : {threat_same}/{len(threat_results)}

RQ1: Both corpora retain H0 (p=0.127 vs p=0.911)
RQ2/RQ3: {cert_same}/{len(cert_results)} certs and {threat_same}/{len(threat_results)} threats show same directional pattern

Conclusion: Core findings are robust to exclusion of
general AI subreddits (r/ChatGPT, r/OpenAI, r/artificial)
""")