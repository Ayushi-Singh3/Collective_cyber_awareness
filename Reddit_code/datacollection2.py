

import requests
import pandas as pd
import time
import os
from datetime import datetime


OUT_FOLDER    = r"D:\For Rajesh'sir\Result of code"
EXISTING_FILE = os.path.join(OUT_FOLDER,
                             "corpus_50k_FINAL.csv")
ADD_RAW_FILE  = os.path.join(OUT_FOLDER,
                             "corpus_ADDITIONAL_raw.csv")
FINAL_FILE    = os.path.join(OUT_FOLDER,
                             "corpus_50k_FINAL.csv")
CHECKPOINT    = os.path.join(OUT_FOLDER,
                             "checkpoint_additional.csv")
RQ1_FILE      = os.path.join(OUT_FOLDER, "RQ1_FINAL.csv")
RQ2_FILE      = os.path.join(OUT_FOLDER, "RQ2_FINAL.csv")
RQ3_FILE      = os.path.join(OUT_FOLDER, "RQ3_FINAL.csv")



GENAI_BOUNDARY = datetime(2022, 11, 1)
PHASE1_END     = datetime(2022, 10, 31)
PHASE2_END     = datetime(2023, 6, 30)

TIMEOUT        = 30
MAX_RETRIES    = 3
LIMIT          = 100



SUBREDDITS = [
    "cybersecurity",
    "netsec",
    "ITCareerQuestions",
    "AskNetsec",
    "SecurityCareerAdvice",
    "netsecstudents",
    "cybersecurityjobs",
]



ALL_CHUNKS = [
    ("2019-01-01", "2019-06-30"),
    ("2019-07-01", "2019-12-31"),
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



NEW_RQ1_QUERIES = [
    "cybersecurity news",
    "AI security",
    "security breach",
    "cyber threat",
    "cybersecurity tools",
    "incident response",
    "penetration testing",
    "vulnerability assessment",
    "ChatGPT hacking",
    "AI generated malware",
]

NEW_RQ2_QUERIES = [
    "endpoint security",
    "cyber insurance",
    "security awareness",
    "dark web",
    "VPN security",
    "multi factor authentication",
    "AI cybersecurity tools",
    "ChatGPT security tools",
    "network monitoring",
    "cloud security breach",
]

NEW_RQ3_QUERIES = [
    "CompTIA Network Plus",
    "CompTIA A Plus",
    "OSCP worth it",
    "CISSP worth it",
    "cybersecurity salary",
    "cybersecurity interview",
    "CompTIA CySA",
    "cybersecurity without degree",
    "security certification 2024",
]

NEW_QUERIES = {
    "RQ1": NEW_RQ1_QUERIES,
    "RQ2": NEW_RQ2_QUERIES,
    "RQ3": NEW_RQ3_QUERIES,
}



def get_era(dt):
    return ("post_genAI"
            if dt >= GENAI_BOUNDARY
            else "pre_genAI")

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
    print("Testing API...")
    url = ("https://arctic-shift.photon-reddit.com"
           "/api/posts/search")
    try:
        r = requests.get(url, params={
            "query": "cybersecurity",
            "subreddit": "cybersecurity",
            "limit": 3
        }, timeout=30)
        if r.status_code == 200:
            print("API working! ")
            return True
        print(f"API error: {r.status_code}")
        return False
    except Exception as e:
        print(f"Connection failed: {e}")
        return False



def fetch_posts(query, subreddit,
                after_date, before_date):
    url = ("https://arctic-shift.photon-reddit.com"
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
                        ts = p.get("created_utc", 0)
                        created = datetime.utcfromtimestamp(
                            int(ts))
                        collected.append({
                            "id":       p.get("id", ""),
                            "query":    query,
                            "rq":       "",
                            "subreddit":p.get(
                                "subreddit", subreddit),
                            "title":    p.get("title", ""),
                            "text":     p.get(
                                "selftext", "")[:500],
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
            time.sleep(2)
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2)
    return []

def _save_checkpoint(posts):
    if posts:
        df = pd.DataFrame(posts)
        df = df.drop_duplicates(
            subset=["title", "date", "subreddit"])
        df.to_csv(CHECKPOINT, index=False)



def collect_additional():
    all_posts = []
    total_queries = sum(
        len(q) for q in NEW_QUERIES.values())
    total_combos  = (total_queries *
                     len(ALL_CHUNKS) *
                     len(SUBREDDITS))

    print(f"New queries    : {total_queries}")
    print(f"Subreddits     : {len(SUBREDDITS)}")
    print(f"Time chunks    : {len(ALL_CHUNKS)}")
    print(f"Total API calls: ~{total_combos}")

    for ci, (cs, ce) in enumerate(ALL_CHUNKS, 1):
        print(f"\nChunk [{ci}/{len(ALL_CHUNKS)}]: "
              f"{cs} → {ce}")

        for rq_name, queries in NEW_QUERIES.items():
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
        print(f"   Checkpoint: {len(all_posts)} posts")

    if not all_posts:
        print("No additional posts found!")
        return pd.DataFrame()

    df_add = pd.DataFrame(all_posts)
    df_add = df_add.drop_duplicates(
        subset=["title", "date", "subreddit"])
    df_add = df_add.fillna("")
    df_add = df_add[df_add["year"].between(2019, 2025)]
    df_add["full_text"] = (
        df_add["title"] + " " + df_add["text"]
    ).str.strip()

    df_add.to_csv(ADD_RAW_FILE, index=False)
    print(f"\nAdditional raw: {len(df_add):,} posts")
    return df_add



def merge_additional(df_add):
    print("\n" + "="*55)
    print("MERGING with existing 31,535 posts")
    print("="*55)

    if not os.path.exists(EXISTING_FILE):
        print(f"ERROR: File not found!")
        print(f"  {EXISTING_FILE}")
        exit()

    df_existing = pd.read_csv(EXISTING_FILE)
    print(f"Existing posts : {len(df_existing):,}")
    print(f"New posts      : {len(df_add):,}")

    # Standardize dates
    df_existing["date"] = df_existing[
        "date"].apply(standardize_date)

    # Merge
    df_combined = pd.concat(
        [df_existing, df_add],
        ignore_index=True
    )

    # Remove duplicates — title+date+subreddit
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(
        subset=["title", "date", "subreddit"])
    removed = before - len(df_combined)
    print(f"Duplicates removed: {removed:,}")

    # Clean
    df_combined = df_combined.fillna("")
    df_combined = df_combined[
        df_combined["year"].between(2019, 2025)]

    # full_text ensure
    if "full_text" not in df_combined.columns:
        df_combined["full_text"] = (
            df_combined["title"].fillna("") + " " +
            df_combined["text"].fillna("")
        ).str.strip()

    # Save RQ files
    print("\nSaving RQ files...")
    for rq, rq_file in [
        ("RQ1", RQ1_FILE),
        ("RQ2", RQ2_FILE),
        ("RQ3", RQ3_FILE)
    ]:
        rq_df = df_combined[df_combined["rq"] == rq]
        rq_df.to_csv(rq_file, index=False)
        print(f"  {rq}: {len(rq_df):,} posts ")

    # Save final — overwrites existing
    df_combined.to_csv(FINAL_FILE, index=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"UPDATED CORPUS READY!")
    print(f"{'='*55}")
    print(f"Was            : {len(df_existing):,}")
    print(f"Added          : {len(df_add):,}")
    print(f"Duplicates out : {removed:,}")
    print(f"TOTAL FINAL    : {len(df_combined):,}")

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

    print(f"\nFile saved:")
    print(f"  {FINAL_FILE}")

    return df_combined



if __name__ == "__main__":

    print("=" * 55)
    print("ADDITIONAL DATA COLLECTION")
    print("New queries only | Same 7 subreddits")
    print("Target: 42,000-43,000 posts")
    print("=" * 55)

    # Test API
    if not test_api():
        print("API not working! Check internet.")
        exit()

    print(f"\nExisting file  : {EXISTING_FILE}")
    print(f"Output folder  : {OUT_FOLDER}")
    print(f"Expected time  : 1-2 hours")
    print(f"Checkpoint     : auto-saved every chunk")
    print("-"*55)

    # Collect
    df_add = collect_additional()

    if df_add.empty:
        print("\nNo additional posts found!")
        exit()

    # Merge
    df_final = merge_additional(df_add)

    # Final check
    print("\n" + "="*55)
    if len(df_final) >= 42000:
        print(f"TARGET ACHIEVED: "
              f"{len(df_final):,} posts!")
    else:
        print(f"Total: {len(df_final):,} posts")

    print(f"\nNext step: Run RQ1 analysis!")
    print(f"Use file : corpus_50k_FINAL.csv")