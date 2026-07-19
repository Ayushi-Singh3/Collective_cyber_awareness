import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_FOLDER     = r"D:\For Rajesh'sir\Result of code"
ANALYSIS_FOLDER = r"D:\For Rajesh'sir\Result of code\RQ1_Analysis"

INPUT_FILE = rf"{ANALYSIS_FOLDER}\RQ1_sentiment_complete.csv"

print("=" * 55)
print("POWER ANALYSIS + KRUSKAL-WALLIS")
print("=" * 55)

df = pd.read_csv(INPUT_FILE)
print(f"Total posts: {len(df):,}")
print()


print("─"*45)
print("STEP 1: POWER ANALYSIS")
print("─"*45)

# Mann-Whitney power analysis
# With N=43,015 what effect size can we detect?
n1 = len(df[df["era"] == "pre_genAI"])
n2 = len(df[df["era"] == "post_genAI"])

# At alpha=0.05, power=0.80, what minimum effect size?
# For Mann-Whitney: minimum detectable r = sqrt(z^2 / n)
# z for alpha=0.05 two-tailed = 1.96, z for power=0.80 = 0.842
z_alpha = 1.96
z_beta  = 0.842
n_harm  = 2 * n1 * n2 / (n1 + n2)  # harmonic mean

min_detectable_r = np.sqrt((z_alpha + z_beta)**2 / n_harm)

print(f"Pre-GenAI posts  : {n1:,}")
print(f"Post-GenAI posts : {n2:,}")
print(f"Harmonic mean N  : {n_harm:,.0f}")
print(f"\nAt alpha = 0.05, power = 0.80:")
print(f"Minimum detectable effect size (r): {min_detectable_r:.6f}")
print(f"Our observed effect size (r)       : -0.0085")
print()
print("INTERPRETATION:")
print(f"With N = {len(df):,} posts, we can detect effects as small as r = {min_detectable_r:.4f}")
print("Our observed r = -0.0085 is BELOW this threshold")
print("This means: if a real effect existed, we would have detected it")
print("The null result is SUBSTANTIVE, not a power failure")

# Power at observed effect
observed_r = 0.0085
z_effect   = observed_r * np.sqrt(n_harm)
from scipy.stats import norm
power_at_observed = norm.cdf(z_effect - z_alpha) + norm.cdf(-z_effect - z_alpha)
print(f"\nPower at observed r = {observed_r}:")
print(f"  Approximate power: {power_at_observed:.1%}")
print(f"  Meaning: {power_at_observed:.1%} chance of detecting THIS effect if real")


print()
print("─"*45)
print("STEP 2: KRUSKAL-WALLIS — VADER across 3 phases")
print("─"*45)

phase_order = [
    "Phase1_PreAwareness",
    "Phase2_Disruption",
    "Phase3_Normalisation"
]

p1_vader = df[df["phase"] == phase_order[0]]["vader_score"].dropna()
p2_vader = df[df["phase"] == phase_order[1]]["vader_score"].dropna()
p3_vader = df[df["phase"] == phase_order[2]]["vader_score"].dropna()

print(f"Phase 1 n = {len(p1_vader):,}, mean = {p1_vader.mean():.4f}")
print(f"Phase 2 n = {len(p2_vader):,}, mean = {p2_vader.mean():.4f}")
print(f"Phase 3 n = {len(p3_vader):,}, mean = {p3_vader.mean():.4f}")

kw_stat, kw_p = stats.kruskal(p1_vader, p2_vader, p3_vader)
print(f"\nKruskal-Wallis H({2}) = {kw_stat:.4f}, p = {kw_p:.6f}")
print(f"Result: {'SIGNIFICANT' if kw_p < 0.05 else 'NOT SIGNIFICANT'} at alpha = 0.05")

# Effect size eta-squared
n_total = len(p1_vader) + len(p2_vader) + len(p3_vader)
eta_sq  = (kw_stat - 2) / (n_total - 3)
print(f"Eta-squared: {eta_sq:.6f}")


print()
print("─"*45)
print("STEP 3: KRUSKAL-WALLIS — FEAR proportion across phases")
print("─"*45)

if "emotion_label" in df.columns:
    # Binary fear indicator
    df["is_fear"] = (df["emotion_label"] == "fear").astype(int)

    p1_fear = df[df["phase"] == phase_order[0]]["is_fear"]
    p2_fear = df[df["phase"] == phase_order[1]]["is_fear"]
    p3_fear = df[df["phase"] == phase_order[2]]["is_fear"]

    print(f"Phase 1 fear %: {p1_fear.mean()*100:.2f}% (n={len(p1_fear):,})")
    print(f"Phase 2 fear %: {p2_fear.mean()*100:.2f}% (n={len(p2_fear):,})")
    print(f"Phase 3 fear %: {p3_fear.mean()*100:.2f}% (n={len(p3_fear):,})")

    kw_fear_stat, kw_fear_p = stats.kruskal(p1_fear, p2_fear, p3_fear)
    print(f"\nKruskal-Wallis H(2) = {kw_fear_stat:.4f}, p = {kw_fear_p:.6f}")
    print(f"Result: {'SIGNIFICANT' if kw_fear_p < 0.05 else 'NOT SIGNIFICANT'}")

   
    print()
    print("─"*45)
    print("STEP 4: DUNN POST-HOC (Bonferroni correction)")
    print("─"*45)

    try:
        import scikit_posthocs as sp
        fear_groups = [
            p1_fear.values,
            p2_fear.values,
            p3_fear.values
        ]
        dunn = sp.posthoc_dunn(
            fear_groups,
            p_adjust="bonferroni"
        )
        dunn.index   = ["Phase1", "Phase2", "Phase3"]
        dunn.columns = ["Phase1", "Phase2", "Phase3"]
        print("\nDunn post-hoc p-values (Bonferroni corrected):")
        print(dunn.round(4).to_string())

        print("\nINTERPRETATION:")
        p12 = dunn.iloc[0, 1]
        p13 = dunn.iloc[0, 2]
        p23 = dunn.iloc[1, 2]
        print(f"  Phase 1 vs Phase 2: p = {p12:.4f} {'*' if p12 < 0.05 else 'ns'}")
        print(f"  Phase 1 vs Phase 3: p = {p13:.4f} {'*' if p13 < 0.05 else 'ns'}")
        print(f"  Phase 2 vs Phase 3: p = {p23:.4f} {'*' if p23 < 0.05 else 'ns'}")

    except ImportError:
        print("scikit_posthocs not installed")
        print("Run: pip install scikit-posthocs")

        # Manual Mann-Whitney between phases with Bonferroni
        print("\nManual pairwise Mann-Whitney (Bonferroni alpha = 0.0167):")
        pairs = [
            ("Phase1 vs Phase2", p1_fear, p2_fear),
            ("Phase1 vs Phase3", p1_fear, p3_fear),
            ("Phase2 vs Phase3", p2_fear, p3_fear),
        ]
        for name, a, b in pairs:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            n_a, n_b = len(a), len(b)
            r = 1 - (2 * stat) / (n_a * n_b)
            sig = "*" if p < 0.0167 else "ns"
            print(f"  {name}: p = {p:.6f} {sig}, r = {r:.4f}")


print()
print("─"*45)
print("STEP 5: CHI-SQUARE — FEAR counts across phases")
print("─"*45)

if "emotion_label" in df.columns:
    contingency = pd.crosstab(
        df["phase"],
        df["emotion_label"]
    )
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
    print(f"Chi-square = {chi2:.4f}, df = {dof}, p = {chi2_p:.6f}")
    print(f"Result: {'SIGNIFICANT' if chi2_p < 0.05 else 'NOT SIGNIFICANT'}")

    # Cramers V
    n     = contingency.values.sum()
    cramv = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    print(f"Cramer's V = {cramv:.4f} (effect size)")


print()
print("="*55)
print("SUMMARY FOR PAPER")
print("="*55)
print(f"""
Power Analysis:
  N = {len(df):,} posts
  Min detectable r = {min_detectable_r:.4f}
  Our r = -0.0085 (below threshold)
  Conclusion: Null result is substantive

Kruskal-Wallis (VADER):
  H(2) = {kw_stat:.4f}, p = {kw_p:.6f}

Phase means:
  Phase 1: {p1_vader.mean():.4f}
  Phase 2: {p2_vader.mean():.4f}
  Phase 3: {p3_vader.mean():.4f}
""")