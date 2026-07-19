import pandas as pd
import os


OUT_FOLDER  = r"D:\For Rajesh'sir\Result of code"
OLD_FILE    = os.path.join(OUT_FOLDER, "old 40k.csv")
NEW_FILE    = os.path.join(OUT_FOLDER, "corpus_50k_FINAL.csv")
FINAL_FILE  = os.path.join(OUT_FOLDER, "corpus_50k_FINAL.csv")
RQ1_FILE    = os.path.join(OUT_FOLDER, "RQ1_FINAL.csv")
RQ2_FILE    = os.path.join(OUT_FOLDER, "RQ2_FINAL.csv")
RQ3_FILE    = os.path.join(OUT_FOLDER, "RQ3_FINAL.csv")


from datetime import datetime
GENAI_BOUNDARY = datetime(2022, 11, 1)
PHASE1_END     = datetime(2022, 10, 31)
PHASE2_END     = datetime(2023, 6, 30)

def get_era(dt):
    return "post_genAI" if dt >= GENAI_BOUNDARY else "pre_genAI"

def get_phase(dt):
    if dt <= PHASE1_END:
        return "Phase1_PreAwareness"
    elif dt <= PHASE2_END:
        return "Phase2_Disruption"
    else:
        return "Phase3_Normalisation"


print("=" * 55)
print("MERGING OLD 40K + NEW CORPUS")
print("=" * 55)

print("\nLoading old 40k file...")
df_old = pd.read_csv(OLD_FILE)
print(f"Old corpus : {len(df_old):,} posts")
print("Subreddits:")
print(df_old['subreddit'].value_counts().to_string())

print("\nLoading new corpus...")
df_new = pd.read_csv(NEW_FILE)
print(f"New corpus : {len(df_new):,} posts")
print("Subreddits:")
print(df_new['subreddit'].value_counts().to_string())

# ── Fix old corpus — add phase if missing 
if 'phase' not in df_old.columns:
    print("\nAdding phase column to old corpus...")
    df_old['date_dt'] = pd.to_datetime(
        df_old['date'], dayfirst=True, errors='coerce')
    df_old['phase'] = df_old['date_dt'].apply(
        lambda x: get_phase(x) if pd.notna(x) 
        else "Phase1_PreAwareness")
    df_old.drop(columns=['date_dt'], inplace=True)

# Fix era if missing
if 'era' not in df_old.columns:
    df_old['date_dt'] = pd.to_datetime(
        df_old['date'], dayfirst=True, errors='coerce')
    df_old['era'] = df_old['date_dt'].apply(
        lambda x: get_era(x) if pd.notna(x)
        else "pre_genAI")
    df_old.drop(columns=['date_dt'], inplace=True)

# Fix full_text if missing
if 'full_text' not in df_old.columns:
    df_old['full_text'] = (
        df_old['title'].fillna("") + " " +
        df_old['text'].fillna("")
    ).str.strip()


print("\n" + "=" * 55)
print("Merging...")
print("=" * 55)

df_combined = pd.concat([df_old, df_new], ignore_index=True)
before = len(df_combined)

# Remove duplicates
df_combined = df_combined.drop_duplicates(
    subset=["title", "date", "subreddit"])
removed = before - len(df_combined)
df_combined = df_combined.fillna("")
df_combined = df_combined[
    df_combined["year"].between(2019, 2025)]

print(f"Old posts      : {len(df_old):,}")
print(f"New posts      : {len(df_new):,}")
print(f"Duplicates out : {removed:,}")
print(f"TOTAL FINAL    : {len(df_combined):,}")

# ── Save main file 
df_combined.to_csv(FINAL_FILE, index=False)
print(f"\nSaved: corpus_50k_FINAL.csv")

# ── Save RQ files 
print("\nSaving RQ files...")
for rq, rq_file in [("RQ1", RQ1_FILE),
                     ("RQ2", RQ2_FILE),
                     ("RQ3", RQ3_FILE)]:
    rq_df = df_combined[df_combined["rq"] == rq]
    rq_df.to_csv(rq_file, index=False)
    print(f"  {rq}: {len(rq_df):,} posts")

# ── Final Summary 
print("\n" + "=" * 55)
print("FINAL CORPUS SUMMARY")
print("=" * 55)
print(f"Total posts: {len(df_combined):,}")
print()

print("Phase wise:")
phase = df_combined['phase'].value_counts().sort_index()
print(phase.to_string())
print()

p1 = df_combined[
    df_combined['phase']=='Phase1_PreAwareness']
p2 = df_combined[
    df_combined['phase']=='Phase2_Disruption']
p3 = df_combined[
    df_combined['phase']=='Phase3_Normalisation']
print("Posts per month:")
print(f"  Phase 1 (47 months): {len(p1)/47:.0f}/month")
print(f"  Phase 2 (8 months) : {len(p2)/8:.0f}/month")
print(f"  Phase 3 (30 months): {len(p3)/30:.0f}/month")
print()

print("Subreddit wise:")
print(df_combined['subreddit'].value_counts().to_string())
print()

print("Year wise:")
print(df_combined['year'].value_counts()
      .sort_index().to_string())
print()

print("Era wise:")
print(df_combined['era'].value_counts().to_string())
print()

print("=" * 55)
print("Done! Corpus ready for analysis.")
print("=" * 55)