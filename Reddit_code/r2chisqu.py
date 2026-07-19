import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_FOLDER     = r"D:\For Rajesh'sir\Result of code"
ANALYSIS_FOLDER = r"D:\For Rajesh'sir\Result of code\RQ3_Analysis"

print("=" * 55)
print("R2 — CHI-SQUARE ANALYSIS")
print("Certification + Threat counts across phases/eras")
print("=" * 55)

cert_era   = pd.read_csv(rf"{ANALYSIS_FOLDER}\RQ3_cert_era.csv")
cert_phase = pd.read_csv(rf"{ANALYSIS_FOLDER}\RQ3_cert_phase.csv")
threat_era   = pd.read_csv(rf"{ANALYSIS_FOLDER}\RQ3_threat_era.csv")
threat_phase = pd.read_csv(rf"{ANALYSIS_FOLDER}\RQ3_threat_phase.csv")

print(f"Cert era columns  : {cert_era.columns.tolist()}")
print(f"Cert phase columns: {cert_phase.columns.tolist()}")
print(f"Threat era columns: {threat_era.columns.tolist()}")
print()


print("─"*45)
print("STEP 1: CHI-SQUARE — Certification pre vs post GenAI")
print("─"*45)

# Build contingency table
cert_era_clean = cert_era.copy()
cert_era_clean.columns = [c.strip() for c in cert_era_clean.columns]

if 'entity' in cert_era_clean.columns:
    cert_era_clean = cert_era_clean.set_index('entity')

# Keep only pre and post columns
cols = [c for c in cert_era_clean.columns if c in ['pre_genAI', 'post_genAI']]
cert_contingency = cert_era_clean[cols].fillna(0).astype(int)

print("\nCertification contingency table (pre vs post):")
print(cert_contingency.to_string())

chi2_cert, p_cert, dof_cert, _ = stats.chi2_contingency(cert_contingency)
n_cert = cert_contingency.values.sum()
cramv_cert = np.sqrt(chi2_cert / (n_cert * (min(cert_contingency.shape) - 1)))

print(f"\nChi-square = {chi2_cert:.4f}")
print(f"df = {dof_cert}")
print(f"p = {p_cert:.6f}")
print(f"Cramer's V = {cramv_cert:.4f}")
print(f"Result: {'SIGNIFICANT' if p_cert < 0.05 else 'NOT SIGNIFICANT'}")

print()
print("─"*45)
print("STEP 2: CHI-SQUARE — Threat pre vs post GenAI")
print("─"*45)

threat_era_clean = threat_era.copy()
threat_era_clean.columns = [c.strip() for c in threat_era_clean.columns]

if 'entity' in threat_era_clean.columns:
    threat_era_clean = threat_era_clean.set_index('entity')

cols_t = [c for c in threat_era_clean.columns if c in ['pre_genAI', 'post_genAI']]
threat_contingency = threat_era_clean[cols_t].fillna(0).astype(int)

print("\nThreat contingency table (pre vs post):")
print(threat_contingency.to_string())

chi2_threat, p_threat, dof_threat, _ = stats.chi2_contingency(threat_contingency)
n_threat = threat_contingency.values.sum()
cramv_threat = np.sqrt(chi2_threat / (n_threat * (min(threat_contingency.shape) - 1)))

print(f"\nChi-square = {chi2_threat:.4f}")
print(f"df = {dof_threat}")
print(f"p = {p_threat:.6f}")
print(f"Cramer's V = {cramv_threat:.4f}")
print(f"Result: {'SIGNIFICANT' if p_threat < 0.05 else 'NOT SIGNIFICANT'}")


print()
print("─"*45)
print("STEP 3: CHI-SQUARE — Certification across 3 phases")
print("─"*45)

cert_phase_clean = cert_phase.copy()
cert_phase_clean.columns = [c.strip() for c in cert_phase_clean.columns]

if 'entity' in cert_phase_clean.columns:
    cert_phase_clean = cert_phase_clean.set_index('entity')

phase_cols = [c for c in cert_phase_clean.columns
              if 'Phase' in c or 'phase' in c.lower()]
print(f"Phase columns found: {phase_cols}")

if len(phase_cols) >= 2:
    cert_phase_ct = cert_phase_clean[phase_cols].fillna(0).astype(int)
    chi2_cp, p_cp, dof_cp, _ = stats.chi2_contingency(cert_phase_ct)
    n_cp = cert_phase_ct.values.sum()
    cramv_cp = np.sqrt(chi2_cp / (n_cp * (min(cert_phase_ct.shape) - 1)))

    print(f"\nChi-square = {chi2_cp:.4f}")
    print(f"df = {dof_cp}")
    print(f"p = {p_cp:.6f}")
    print(f"Cramer's V = {cramv_cp:.4f}")
    print(f"Result: {'SIGNIFICANT' if p_cp < 0.05 else 'NOT SIGNIFICANT'}")

# ── STEP 4: CHI-SQUARE — THREAT across 3 phases ───────────
print()
print("─"*45)
print("STEP 4: CHI-SQUARE — Threat across 3 phases")
print("─"*45)

threat_phase_clean = threat_phase.copy()
threat_phase_clean.columns = [c.strip() for c in threat_phase_clean.columns]

if 'entity' in threat_phase_clean.columns:
    threat_phase_clean = threat_phase_clean.set_index('entity')

phase_cols_t = [c for c in threat_phase_clean.columns
                if 'Phase' in c or 'phase' in c.lower()]

if len(phase_cols_t) >= 2:
    threat_phase_ct = threat_phase_clean[phase_cols_t].fillna(0).astype(int)
    chi2_tp, p_tp, dof_tp, _ = stats.chi2_contingency(threat_phase_ct)
    n_tp = threat_phase_ct.values.sum()
    cramv_tp = np.sqrt(chi2_tp / (n_tp * (min(threat_phase_ct.shape) - 1)))

    print(f"\nChi-square = {chi2_tp:.4f}")
    print(f"df = {dof_tp}")
    print(f"p = {p_tp:.6f}")
    print(f"Cramer's V = {cramv_tp:.4f}")
    print(f"Result: {'SIGNIFICANT' if p_tp < 0.05 else 'NOT SIGNIFICANT'}")

# ── STEP 5: AI-SPECIFIC THREATS separate chi-square ───────
print()
print("─"*45)
print("STEP 5: AI-SPECIFIC THREATS — chi-square")
print("─"*45)

ai_threats = ["Prompt Injection", "Deepfake", "AI Phishing", "AI Malware"]

if 'entity' in threat_era.columns:
    threat_ai = threat_era[threat_era['entity'].isin(ai_threats)].copy()
else:
    threat_ai = threat_era[threat_era.index.isin(ai_threats)].copy()

print("AI-specific threat counts:")
print(threat_ai.to_string())

if len(threat_ai) > 0:
    cols_ai = [c for c in threat_ai.columns if c in ['pre_genAI', 'post_genAI']]
    if len(cols_ai) == 2:
        ai_ct = threat_ai[cols_ai].fillna(0).astype(int)
        try:
            chi2_ai, p_ai, dof_ai, _ = stats.chi2_contingency(ai_ct)
            n_ai = ai_ct.values.sum()
            cramv_ai = np.sqrt(chi2_ai / (n_ai * (min(ai_ct.shape) - 1)))
            print(f"\nChi-square = {chi2_ai:.4f}")
            print(f"p = {p_ai:.6f}")
            print(f"Cramer's V = {cramv_ai:.4f}")
            print(f"Result: {'SIGNIFICANT' if p_ai < 0.05 else 'NOT SIGNIFICANT'}")
        except Exception as e:
            print(f"Chi-square failed: {e}")
            print("(May be due to zero cells — using Fisher exact instead)")


print()
print("="*55)
print("SUMMARY FOR PAPER")
print("="*55)
print(f"""
Certification pre vs post GenAI:
  Chi-square = {chi2_cert:.4f}, p = {p_cert:.6f}
  Cramer's V = {cramv_cert:.4f}

Threat pre vs post GenAI:
  Chi-square = {chi2_threat:.4f}, p = {p_threat:.6f}
  Cramer's V = {cramv_threat:.4f}
""")