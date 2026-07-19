

import requests
import pandas as pd
import time
import os
from datetime import datetime


# Code folder
CODE_FOLDER   = r"D:\For Rajesh'sir\Python code"

# Output folder — sir ko yahi share karoge
OUT_FOLDER    = r"D:\For Rajesh'sir\Result of code"

# Existing 25k file — jo copy ki hai
EXISTING_FILE = r"D:\For Rajesh'sir\Result of code\corpus_FINAL_clean.csv"

# New files jo banenge
NEW_RAW_FILE  = os.path.join(OUT_FOLDER, "corpus_NEW_raw.csv")
FINAL_50K     = os.path.join(OUT_FOLDER, "corpus_50k_FINAL.csv")
CHECKPOINT    = os.path.join(OUT_FOLDER, "checkpoint_new_data.csv")

# RQ files
RQ1_FILE      = os.path.join(OUT_FOLDER, "RQ1_FINAL.csv")
RQ2_FILE      = os.path.join(OUT_FOLDER, "RQ2_FINAL.csv")
RQ3_FILE      = os.path.join(OUT_FOLDER, "RQ3_FINAL.csv")

os.makedirs(OUT_FOLDER, exist_ok=True)


GENAI_BOUNDARY = datetime(2022, 11, 1)
PHASE1_END     = datetime(2022, 10, 31)
PHASE2_END     = datetime(2023, 6, 30)
PHASE3_START   = datetime(2023, 7, 1)

TIMEOUT        = 30
MAX_RETRIES    = 3
LIMIT          = 100

# All 7 subreddits
SUBREDDITS = [
    "cybersecurity",
    "netsec",
    "ITCareerQuestions",
    "AskNetsec",
    "SecurityCareerAdvice",
    "netsecstudents",
    "cybersecurityjobs",
]

# Only new subreddits for 2020-2025
NEW_SUBREDDITS = [
    "netsecstudents",
    "cybersecurityjobs",
]


# 2019 — all 7 subreddits (completely new data)
CHUNKS_2019 = [
    ("2019-01-01", "2019-06-30"),
    ("2019-07-01", "2019-12-31"),
]

# 2020-2025 — only new subreddits
CHUNKS_2020_2025 = [
    ("2020-01-01", "2020-06-30"),
    ("2020-07-01", "2020-12-31"),
    ("2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-10-31"),
    ("2022-11-01", "2023-04-30"),
    ("2023-05-01", "2023-10-31"),
    ("2023-11-01", "2024-04-30"),
    ("2024-05-01", "2024-10-31"),
    ("2024-11-01", "2025-12-31"),
]


RQ1_QUERIES = [
    "cybersecurity career",
    "cyber attack",
    "data breach",
    "network security",
    "cybersecurity jobs",
    "SOC analyst",
    "malware threat",
    "phishing attack",
    "cybersecurity skills",
    "information security",
    "zero day vulnerability",
    "ransomware attack",
    "generative AI cybersecurity",
    "ChatGPT cybersecurity",
    "AI threat cybersecurity",
    "LLM security",
    "AI replace cybersecurity",
]

RQ2_QUERIES = [
    "cybersecurity threat",
    "cloud security",
    "AI security threat",
    "supply chain attack",
    "IoT security",
    "password security",
    "threat intelligence",
    "social engineering",
    "zero trust security",
    "AI hacking",
    "deepfake attack",
    "prompt injection",
    "AI phishing",
    "cybersecurity automation",
]

RQ3_QUERIES = [
    "CISSP certification",
    "CompTIA Security Plus",
    "CEH certification",
    "OSCP certification",
    "cybersecurity certification",
    "CISM certification",
    "CISA certification",
    "cybersecurity bootcamp",
    "best cybersecurity certification",
    "certification worth it AI",
    "cybersecurity degree AI",
    "cert still relevant AI",
    "CRISC certification",
    "eJPT certification",
]

ALL_QUERIES = {
    "RQ1": RQ1_QUERIES,
    "RQ2": RQ2_QUERIES,
    "RQ3": RQ3_QUERIES,
}



def get_era(dt):
    return "post_genAI" if dt >= GENAI_BOUNDARY else "pre_genAI"

def get_phase(dt):
    if dt <= PHASE1_END:
        return "Phase1_PreAwareness"
    elif dt <= PHASE2_END:
        return "Phase2_Disruption"
    else:
        return "Phase3_Normalisation"

def standardize_date(date_str):
    for fmt in ["%d-%m-%Y", "%Y-%m-%d",
                "%d/%m/%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(
                str(date_str), fmt
            ).strftime("%d-%m-%Y")
        except:
            continue
    return str(date_str)



def test_api():
    print("Testing Arctic Shift API...")
    url    = ("https://arctic-shift.photon-reddit.com"
              "/api/posts/search")
    params = {
        "query":     "cybersecurity",
        "subreddit": "cybersecurity",
        "limit":     3,
    }
    try:
        r = requests.get(url, params=params,
                         timeout=30)
        if r.status_code == 200:
            data = r.json().get("data", [])
            print(f"API working!  "
                  f"Sample: {len(data)} posts")
            return True
        else:
            print(f"API error: {r.status_code}")
            return False
    except Exception as e:
        print(f"Connection failed: {e}")
        return False



def fetch_posts(query, subreddit,
                after_date, before_date):
    url    = ("https://arctic-shift.photon-reddit.com"
              "/api/posts/search")
    params = {
        "query":     query,
        "subreddit": subreddit,
        "limit":     LIMIT,
        "after":     after_date,
        "before":    before_date,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params,
                             timeout=TIMEOUT)
            if r.status_code == 200:
                posts     = r.json().get("data", [])
                collected = []
                for p in posts:
                    try:
                        ts      = p.get("created_utc", 0)
                        created = datetime.utcfromtimestamp(
                            int(ts))
                        collected.append({
                            "id":       p.get("id", ""),
                            "query":    query,
                            "rq":       "",
                            "subreddit":p.get("subreddit",
                                              subreddit),
                            "title":    p.get("title", ""),
                            "text":     p.get("selftext",
                                              "")[:500],
                            "score":    p.get("score", 0),
                            "comments": p.get(
                                "num_comments", 0),
                            "date":     created.strftime(
                                "%d-%m-%Y"),
                            "year":     created.year,
                            "era":      get_era(created),
                            "phase":    get_phase(created),
                            "url":      p.get("url", ""),
                        })
                    except:
                        continue
                return collected
            else:
                time.sleep(2)
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2)
    return []


def collect_new_data():
    all_posts = []

    # ── Part A: 2019 — all 7 subreddits 
    print("\n" + "="*55)
    print("PART A: 2019 data — all 7 subreddits")
    print("="*55)

    for ci, (cs, ce) in enumerate(CHUNKS_2019, 1):
        print(f"\nChunk [{ci}/{len(CHUNKS_2019)}]: "
              f"{cs} → {ce}")
        for rq_name, queries in ALL_QUERIES.items():
            for query in queries:
                for sub in SUBREDDITS:
                    posts = fetch_posts(
                        query, sub, cs, ce)
                    for p in posts:
                        p["rq"] = rq_name
                    all_posts.extend(posts)
                    if len(posts) > 0:
                        print(f"  {rq_name} | "
                              f"r/{sub} | "
                              f"{query[:25]:<25} | "
                              f"{len(posts)} posts")
                    time.sleep(0.5)

        _save_checkpoint(all_posts)
        print(f"   Checkpoint saved: "
              f"{len(all_posts)} posts")


    print("\n" + "="*55)
    print("PART B: 2020-2025 — new subreddits only")
    print(f"  {NEW_SUBREDDITS}")
    print("="*55)

    for ci, (cs, ce) in enumerate(CHUNKS_2020_2025, 1):
        print(f"\nChunk [{ci}/{len(CHUNKS_2020_2025)}]: "
              f"{cs} → {ce}")
        for rq_name, queries in ALL_QUERIES.items():
            for query in queries:
                for sub in NEW_SUBREDDITS:
                    posts = fetch_posts(
                        query, sub, cs, ce)
                    for p in posts:
                        p["rq"] = rq_name
                    all_posts.extend(posts)
                    if len(posts) > 0:
                        print(f"  {rq_name} | "
                              f"r/{sub} | "
                              f"{query[:25]:<25} | "
                              f"{len(posts)} posts")
                    time.sleep(0.5)

        _save_checkpoint(all_posts)
        print(f"  Checkpoint saved: "
              f"{len(all_posts)} posts")

   
    if not all_posts:
        print("No new posts collected!")
        return pd.DataFrame()

    df_new = pd.DataFrame(all_posts)
    df_new = df_new.drop_duplicates(
        subset=["title", "date", "subreddit"])
    df_new = df_new.fillna("")
    df_new = df_new[df_new["year"].between(2019, 2025)]
    df_new["full_text"] = (
        df_new["title"] + " " + df_new["text"]
    ).str.strip()

    df_new.to_csv(NEW_RAW_FILE, index=False)
    print(f"\nNew raw data saved: "
          f"{len(df_new):,} posts")
    print(f"File: {NEW_RAW_FILE}")
    return df_new

def _save_checkpoint(posts):
    if posts:
        df_temp = pd.DataFrame(posts)
        df_temp = df_temp.drop_duplicates(
            subset=["title", "date", "subreddit"])
        df_temp.to_csv(CHECKPOINT, index=False)



def merge_with_existing(df_new):
    print("\n" + "="*55)
    print("STEP 4: Merging with existing 25,814 posts")
    print("="*55)

    # Load existing
    if not os.path.exists(EXISTING_FILE):
        print(f"ERROR: File not found!")
        print(f"  {EXISTING_FILE}")
        print("Saving new data only...")
        df_new.to_csv(FINAL_50K, index=False)
        return df_new

    df_existing = pd.read_csv(EXISTING_FILE)
    print(f"Existing posts loaded: {len(df_existing):,}")

    # Standardize date format
    print("Standardizing dates...")
    df_existing["date"] = df_existing["date"].apply(
        standardize_date)

    # Add phase column if missing
    if "phase" not in df_existing.columns:
        print("Adding phase column...")
        df_existing["date_dt"] = pd.to_datetime(
            df_existing["date"],
            format="%d-%m-%Y",
            errors="coerce"
        )
        df_existing["phase"] = df_existing[
            "date_dt"].apply(
            lambda x: get_phase(x)
            if pd.notna(x)
            else "Phase1_PreAwareness"
        )
        df_existing.drop(
            columns=["date_dt"],
            inplace=True
        )

    # Merge
    df_combined = pd.concat(
        [df_existing, df_new],
        ignore_index=True
    )

    # Remove duplicates
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(
        subset=["title", "date", "subreddit"])
    removed = before - len(df_combined)
    print(f"Duplicates removed: {removed:,}")

    # Clean
    df_combined = df_combined.fillna("")
    df_combined = df_combined[
        df_combined["year"].between(2019, 2025)]

    # Ensure full_text
    if "full_text" not in df_combined.columns:
        df_combined["full_text"] = (
            df_combined["title"].fillna("") + " " +
            df_combined["text"].fillna("")
        ).str.strip()

    # Save separate RQ files
    print("\nSaving RQ files...")
    for rq, rq_file in [
        ("RQ1", RQ1_FILE),
        ("RQ2", RQ2_FILE),
        ("RQ3", RQ3_FILE)
    ]:
        rq_df = df_combined[df_combined["rq"] == rq]
        rq_df.to_csv(rq_file, index=False)
        print(f"  {rq}: {len(rq_df):,} posts ")

    # Save final
    df_combined.to_csv(FINAL_50K, index=False)


    print(f"\n{'='*55}")
    print(f"FINAL CORPUS READY!")
    print(f"{'='*55}")
    print(f"Existing posts  : {len(df_existing):,}")
    print(f"New posts       : {len(df_new):,}")
    print(f"Duplicates out  : {removed:,}")
    print(f"TOTAL FINAL     : {len(df_combined):,}")

    print(f"\nEra split:")
    print(df_combined["era"].value_counts()
          .to_string())

    print(f"\nPhase split:")
    print(df_combined["phase"].value_counts()
          .sort_index().to_string())

    print(f"\nSubreddit split:")
    print(df_combined["subreddit"].value_counts()
          .to_string())

    print(f"\nYear split:")
    print(df_combined["year"].value_counts()
          .sort_index().to_string())

    print(f"\n{'='*55}")
    print(f"Files saved in:")
    print(f"  {OUT_FOLDER}")
    print(f"{'='*55}")
    print(f"  corpus_50k_FINAL.csv  ← main file")
    print(f"  RQ1_FINAL.csv")
    print(f"  RQ2_FINAL.csv")
    print(f"  RQ3_FINAL.csv")
    print(f"  corpus_NEW_raw.csv    ← new data only")

    return df_combined



if __name__ == "__main__":

    print("=" * 55)
    print("DATA COLLECTION — 50K TARGET")
    print("7 Subreddits | 2019-2025")
    print("=" * 55)
    print(f"\nCode folder  : {CODE_FOLDER}")
    print(f"Output folder: {OUT_FOLDER}")
    print(f"Existing file: {EXISTING_FILE}")

    # Step 1: Test API
    print("\n" + "-"*55)
    if not test_api():
        print("\nAPI not working!")
        print("Check internet and try again.")
        exit()

    print("\nStarting...")
    print("Expected time : 2-3 hours")
    print("Checkpoint    : auto-saved every chunk")
    print("If interrupted: restart — will continue")
    print("-"*55)

    # Step 2: Collect
    df_new = collect_new_data()

    if df_new.empty:
        print("\nNo new data collected!")
        exit()

    # Step 3: Merge
    df_final = merge_with_existing(df_new)

    # Final check
    print("\n" + "="*55)
    if len(df_final) >= 50000:
        print(f" TARGET ACHIEVED: "
              f"{len(df_final):,} posts!")
    else:
        print(f"Total: {len(df_final):,} posts")
        print(f"   Target: 50,000")

    
    print(f"Use file : corpus_50k_FINAL.csv")