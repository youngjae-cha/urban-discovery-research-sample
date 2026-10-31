"""Historical semantic-analysis method, prepared for source inspection.

The actual29 seed banks, scoring helpers and chunk-processing loop are retained.
Public changes expose input/output/model arguments and put execution behind main.
Seed centroids are encoded before the original half-precision/256-token review settings.
Importing this module neither loads an encoder nor reads review data.
"""
import argparse,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

score_categories = {
    # === Core Dimensions (Refined Sentence-Based) ===
    "Happiness": {
        "positive": [
            "This place made me happy",
            "I enjoyed this experience",
            "I felt comfortable here"
        ],
        "negative": [
            "I felt sad here",
            "This place made my mind unstable",
            "This experience was unpleasant"
        ]
    },
    "Meaningfulness": {
        "positive": [
            "This experience was fulfilling to me",
            "I felt purposeful here",
            "It was a meaningful experience here"
        ],
        "negative": [
            "My mind felt disorganized during this experience",
            "This experience felt meaningless",
            "This visit felt pointless"
        ]
    },
    "Psychological_Richness": {
        "positive": [
            "This experience was interesting",
            "The experience here was psychologically enriching",
            "The experience here felt dramatic",
            "This visit was complex"
        ],
        "negative": [
            "This experience was boring",
            "The experiences I got here were monotonous",
            "This visit was dull"
        ]
    },

    # === Additional Core Dimensions (Sentences Converted) ===
    "FG_Richness": {
        "positive": [
            "This place offered great variety",
            "The experiences here were diverse",
            "I found varied options here"
        ],
        "negative": [
            "This place was repetitive",
            "The experiences here were monotonous"
        ]
    },
    "Exploration_Behavior": {
        "positive": [
            "This place made me want to try different things",
            "I found myself exploring different options here",
            "There was a lot to look around and discover here",
            "I tried things here that I wouldn’t normally choose",
            "This place made me curious to explore more"
        ],
        "negative": [
            "I just stuck with the usual option here",
            "There was no reason to try anything different here",
            "I didn’t feel the need to explore anything here",
            "Everything here led me to make the same choice as usual",
            "Nothing here encouraged me to look around or try something new"
        ]
    },
    
    # === Satisfaction with Life Scale (SWLS) - Experience Version ===
    "Satisfaction_SWLS": {
        "positive": [
            "This place was close to ideal",
            "The overall quality was excellent",
            "I am satisfied with my visit",
            "I got what I was looking for",
            "I would visit again without hesitation"
        ],
        "negative": [
            "This place was far from ideal",
            "The overall experience was poor",
            "I am dissatisfied with my visit",
            "I didn't get what I was looking for",
            "I would not visit again"
        ]
    },

    # === Meaning in Life Questionnaire (MLQ) - Experience Version ===
    "Meaningfulness_MLQ": {
        "positive": [
            "This visit felt meaningful to me",
            "This place had a clear purpose",
            "I found something valuable here",
            "I understand what made this visit worthwhile"
        ],
        "negative": [
            "This visit felt pointless"
        ]
    },
    "Purpose_MLQ": {
        "positive": [
            "I had a clear purpose for visiting this place",
            "This place served a compelling purpose"
        ],
        "negative": [
            "I didn't know what to expect from this visit",
            "This visit lacked a clear purpose"
        ]
    },

    # === Meaning Sub-dimensions ===
    "Appreciation": {
        "positive": [
            "I appreciated the beauty of this place",
            "I found this visit genuinely interesting",
            "This place offered a variety of experiences",
            "I appreciated the small details here",
            "I felt engaged during my visit"
        ],
        "negative": [
            "I didn't appreciate anything about this place",
            "I found this visit uninteresting",
            "This place offered nothing special",
            "I didn't notice any details",
            "I felt disengaged during my visit"
        ]
    },
    "Mattering": {
        "positive": [
            "This experience mattered to me",
            "This place made me feel important"
        ],
        "negative": [
            "This visit felt insignificant",
            "This visit was not significant to me"
        ]
    },
    "Coherence": {
        "positive": [
            "I understood what this place was about",
            "This visit felt coherent and well-organized"
        ],
        "negative": [
            "This experience didn't make sense to me",
            "This experience felt disjointed"
        ]
    },

    # === Psychological Richness Scale (Full 17-item concept) ===
    "PR_Scale_Overall": {
        "positive": [
            "This visit was psychologically rich",
            "This experience was experientially rich",
            "This place offered interesting experiences",
            "I had novel experiences here",
            "This visit included unique, unusual moments",
            "This experience was full of rich, intense moments",
            "I experienced a full range of emotions here",
            "I have stories to tell about this visit",
            "This was an interesting experience",
            "I saw and learned a lot here"
        ],
        "negative": [
            "This visit was monotonous",
            "I felt bored during this visit",
            "This experience was uneventful",
            "I didn't experience anything new here"
        ]
    },

    # === Transformative Experience ===
    "Transformative_Experience": {
        "positive": [
            "This visit challenged some of my firmly held ideas",
            "This experience changed the way I look at myself",
            "This visit changed my normal way of doing things",
            "I discovered new perspectives that challenged my previous beliefs",
            "I experienced a deep shift in the way I see some things",
            "This experience was truly transformative"
        ],
        "negative": [
            "This visit didn't challenge any of my ideas",
            "This experience didn't change me at all",
            "This visit didn't change my way of doing things",
            "I discovered no new perspectives",
            "I experienced no shift in my thinking",
            "This experience was not transformative"
        ]
    },

    # === Curiosity & Exploration Scales ===
    "Curiosity_Stretching": {
        "positive": [
            "I actively sought as much information as I could here",
            "I was at my best with the complex or challenging aspects",
            "I viewed this as an opportunity to grow and learn",
            "I looked for experiences that challenged how I think",
            "I sought out opportunities to challenge myself and grow"
        ],
        "negative": [
            "I didn't seek much information here",
            "I avoided the complex or challenging aspects",
            "I didn't view this as a growth opportunity",
            "I avoided experiences that challenged my thinking",
            "I didn't seek opportunities to challenge myself"
        ]
    },
    "Curiosity_Embracing": {
        "positive": [
            "I really enjoyed the uncertainty of this experience",
            "I was out looking for new things or experiences",
            "I liked the things here that were a little frightening",
            "I enjoyed the unpredictable nature of this place",
            "I embraced the unfamiliar people, events, and places"
        ],
        "negative": [
            "I didn't enjoy the uncertainty here",
            "I wasn't looking for new things or experiences",
            "I avoided things that were frightening",
            "I disliked the unpredictable nature",
            "I didn't embrace the unfamiliar aspects"
        ]
    },

    # === New Scales: Self-Expansion & Intolerance of Uncertainty ===
    "Self_Expansion": {
        "positive": [
            "I enjoyed gaining a more thorough understanding of this place",
            "I was interested in finding new things to try here",
            "I sought out new opportunities or experiences here",
            "Trying new things here was important for my happiness",
            "I placed a lot of importance on experiencing new things here"
        ],
        "negative": [
            "I didn't feel it was necessary to learn more about this place",
            "Trying to learn more about this place was not worth the effort",
            "I preferred familiarity and stability over what this place offered",
            "Exploring this place in depth was tedious and boring",
            "There were better ways to spend my time than trying to learn more about this"
        ]
    },
    "Intolerance_of_Uncertainty": {
        "positive": [
            "The unforeseen events here upset me greatly",
            "It frustrated me not having all the information I needed here",
            "I couldn't stand being taken by surprise here",
            "The uncertainty of this visit kept me from enjoying it",
            "I felt paralyzed by the uncertainty here",
            "I felt I couldn't function well because of the uncertainty",
            "I wanted to get away from the uncertain situation here"
        ],
        "negative": [ # Tolerance (Reverse items)
            "I was comfortable with the unforeseen events here",
            "I didn't mind not having all the information",
            "I enjoyed the surprises this place offered",
            "Uncertainty didn't stop me from having a full experience"
        ]
    },

    # === Cognitive & Openness ===
    "Learning_Knowledge": {
        "positive": [
            "I learned something here",
            "This was educational",
            "This was informative",
            "I gained new insights",
            "This was enlightening"
        ],
        "negative": [
            "I learned nothing here",
            "This was uninformative",
            "This taught me nothing new"
        ]
    },
    "Openness_to_Uncertainty": {
        "positive": [
            "I enjoyed the unpredictability of this place",
            "I liked not knowing what to expect",
            "The ambiguity here was exciting",
            "I appreciated the uncertain nature of this experience",
            "The unexpected elements made this interesting"
        ],
        "negative": [
            "I disliked the unpredictability here",
            "I felt uncomfortable not knowing what to expect",
            "The ambiguity here was frustrating",
            "I needed more structure and clarity",
            "The uncertainty made me anxious"
        ]
    },
    "Need_for_Cognition": {
        "positive": [
            "I preferred the complex aspects of this experience",
            "I enjoyed thinking deeply about what I saw here",
            "This required a lot of thinking and I liked it",
            "I found satisfaction in deliberating about this experience",
            "I enjoyed solving the puzzles this place presented",
            "The intellectual challenge here appealed to me",
            "I appreciated the abstract thinking this required",
            "This made me think in new ways"
        ],
        "negative": [
            "Thinking about this experience was not fun",
            "I would have preferred something requiring less thought",
            "I tried to avoid thinking deeply about this",
            "I felt relief when I didn't have to think hard here",
            "I didn't care how or why things worked here",
            "This didn't excite me intellectually"
        ]
    },

    # === Urbanism & Perspective ===
    "Vitality_Aliveness": {
        "positive": [
            "I felt alive and vital here",
            "This place energized me",
            "I felt a sense of energy and spirit",
            "Being here made me feel alert and awake",
            "I had a lot of energy while I was here",
            "This experience invigorated me"
        ],
        "negative": [
            "I felt drained here",
            "This place depleted my energy",
            "I felt lifeless during this visit",
            "Being here made me feel tired",
            "I had no energy while I was here"
        ]
    },
    "Cosmopolitanism": {
        "positive": [
            "This place felt cosmopolitan",
            "This provided opportunities to build connections",
            "This was a multicultural experience",
            "This place was tolerant toward diversity",
            "This was an open-minded environment",
            "This had an active creative scene",
            "This fostered creativity and innovation",
            "This provided equal opportunities",
            "This was an exciting place"
        ],
        "negative": [
            "This place felt provincial",
            "This provided no opportunities to build connections",
            "This was a monocultural experience",
            "This place was intolerant toward diversity",
            "This was a closed-minded environment",
            "This had no creative scene",
            "This didn't foster creativity or innovation",
            "This didn't provide equal opportunities",
            "This was a boring place"
        ]
    },
    "Shared_Reality": {
        "positive": [
            "People here seemed to share the same thoughts and feelings",
            "This place helped create a shared perspective among visitors",
            "Events felt more real when experienced with others here",
            "People here often seemed to be on the same wavelength",
            "This created a sense of collective understanding",
            "I felt like we created our own reality together here",
            "There was a strong sense of shared experience",
            "People here connected on a deeper level"
        ],
        "negative": [
            "People here seemed disconnected from each other",
            "This place didn't create any shared perspective",
            "Events felt isolated and individual",
            "People here were not on the same wavelength",
            "There was no sense of collective understanding",
            "I felt no shared reality here",
            "There was no sense of shared experience",
            "People here didn't connect"
        ]
    },

    # === Perspective & Surprise (Sentences Converted) ===
    "Perspective_Change": {
        "positive": [
            "This was a transformative experience",
            "This was life-changing",
            "This was eye-opening",
            "This shifted my perspective",
            "I had an epiphany here",
            "This challenged my views",
            "This gave me a new outlook",
            "I experienced a deep shift here",
            "This made me rethink things",
            "This was an awakening"
        ],
        "negative": [
            "This was uninspiring",
            "This was conventional",
            "This reinforced the status quo",
            "This was stagnant",
            "This confirmed what I already knew",
            "This was just the same old thing"
        ]
    },
    "Surprise": {
        "positive": [
            "This was surprising",
            "This was unexpected",
            "This was mind-blowing",
            "This was astonishing",
            "This felt fresh and novel",
            "This was a revelation",
            "This was a startling discovery",
            "I was amazed by this place"
        ],
        "negative": [
            "This was predictable",
            "This was exactly what I expected",
            "This was typical",
            "This was routine",
            "This was formulaic",
            "This was a cliche",
            "This was boring",
            "This was standard"
        ]
    },

    # === Affect & Benchmarks (Sentences Converted) ===
    "SPANE_PA": {
        "positive": [
            "I felt positive emotions here",
            "This was a good experience",
            "I felt pleasant here",
            "I felt happy here",
            "I felt joyful here",
            "I felt contented here"
        ],
        "negative": []
    },
    "SPANE_NA": {
        "positive": [
            "I felt negative emotions here",
            "This was a bad experience",
            "I felt unpleasant here",
            "I felt sad here",
            "I felt afraid here",
            "I felt angry here"
        ],
        "negative": []
    },
    "Price_Benchmark": {
        "positive": [
            "This was expensive",
            "This was pricey",
            "This was costly",
            "This was overpriced"
        ],
        "negative": [
            "This was cheap",
            "This was affordable",
            "This was inexpensive",
            "This was budget-friendly"
        ]
    },
    "Service_Speed_Benchmark": {
        "positive": [
            "The service was fast",
            "The service was quick",
            "The service was prompt",
            "The service was speedy"
        ],
        "negative": [
            "The service was slow",
            "The service was delayed",
            "I had to wait too long",
            "The service was sluggish"
        ]
    }
}

centroid_scores={}

def clean_text(text):
    if isinstance(text, str): return re.sub(r'[^A-Za-z0-9\s.,!\'\"]+', '', text)
    return text

def calculate_centroid_scores(embeddings):
    scores = {}
    for cat, centroids in centroid_scores.items():
        if "positive" in centroids and "negative" in centroids:
            direction = centroids["positive"] - centroids["negative"]
            scores[f'{cat}_Score'] = cosine_similarity(embeddings, [direction]).flatten()
        elif "positive" in centroids:
            scores[f'{cat}_Score'] = cosine_similarity(embeddings, [centroids["positive"]]).flatten()
        elif "negative" in centroids:
            scores[f'{cat}_Score'] = -cosine_similarity(embeddings, [centroids["negative"]]).flatten()
    return scores


def run_semantic_generation(model, review_path, temp_dir, *, year=2019,
                            batch_size=1024, chunk_size=100000, source_label="source"):
    """Original computation with explicit operational inputs; requires authorized reviews."""
    import gc,os,time,torch
    global centroid_scores
    TARGET_STATE=source_label
    TARGET_YEAR=year
    SAMPLE_MODE=year is not None
    TEMP_DIR=str(Path(temp_dir))
    Path(TEMP_DIR).mkdir(parents=True,exist_ok=True)
    review_path=str(review_path)
    centroid_scores = {}
    for category, adjectives in score_categories.items():
        centroid_scores[category] = {}
        if adjectives["positive"]:
            centroid_scores[category]["positive"] = np.mean(model.encode(adjectives["positive"]), axis=0)
        if adjectives["negative"]:
            centroid_scores[category]["negative"] = np.mean(model.encode(adjectives["negative"]), axis=0)
    model.half()
    model.max_seq_length=256
    reader = pd.read_json(review_path, lines=True, compression='gzip', chunksize=chunk_size)
    total_start=time.time()
    chunk_id=0
    total_filtered_reviews=0
    gc.collect()
    torch.cuda.empty_cache()
    if os.path.exists(TEMP_DIR):
        processed = [f for f in os.listdir(TEMP_DIR) if f.startswith('chunk_') and f.endswith('.csv')]
        if processed:
            chunk_id = len(processed)
            print(f"[*] Resuming from chunk {chunk_id} (Skipping already processed)...")
    for i, chunk_df in enumerate(reader):
        if i < chunk_id: 
            continue 
    
        chunk_start = time.time()
    
        # ---------------------------------------------------------
        # [Critical] Date filtering (keep only TARGET_YEAR if SAMPLE_MODE)
        # ---------------------------------------------------------
        if 'time' in chunk_df.columns:
            chunk_df['datetime'] = pd.to_datetime(chunk_df['time'], unit='ms')
        
            # Extract date components
            chunk_df['year'] = chunk_df['datetime'].dt.year
            chunk_df['month'] = chunk_df['datetime'].dt.month
            chunk_df['week'] = chunk_df['datetime'].dt.isocalendar().week
        
            # Apply year filter only in SAMPLE_MODE
            if SAMPLE_MODE:
                chunk_df = chunk_df[chunk_df['year'] == TARGET_YEAR]
        
            if len(chunk_df) == 0:
                if SAMPLE_MODE:
                    print(f"    Chunk {i}: No reviews from {TARGET_YEAR} (Skipping)")
                else:
                    print(f"    Chunk {i}: No valid reviews (Skipping)")
                continue
        else:
            print("[!] 'time' column missing. Skipping chunk.")
            continue

        # 1. Clean text
        if 'text' not in chunk_df.columns: 
            continue
        chunk_df['text'] = chunk_df['text'].fillna("").astype(str)
        chunk_df['text'] = chunk_df['text'].str.replace(r'[^A-Za-z0-9\s.,!\'"]+', '', regex=True)
        chunk_df = chunk_df[chunk_df['text'].str.strip() != ""]
    
        if len(chunk_df) == 0: 
            continue

        # 2. Generate embeddings (auto batch size adjustment)
        embeddings = None
        current_batch = batch_size
    
        while embeddings is None:
            try:
                embeddings = model.encode(
                    chunk_df['text'].tolist(),
                    batch_size=current_batch,
                    show_progress_bar=True, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            except Exception as e:
                if "out of memory" in str(e).lower():
                    old_batch = current_batch
                    current_batch //= 2
                    print(f"    [!] OOM with batch {old_batch}. Retrying with {current_batch}...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    if current_batch < 16:
                        print("    [x] Batch too small. Skipping chunk.")
                        break
                else:
                    print(f"Error in chunk {chunk_id}: {e}")
                    break

        if embeddings is None: 
            continue

        # 3. Calculate scores (new dimensions included)
        scores = calculate_centroid_scores(embeddings)
    
        # 4. Save (gmap_id, year, month, week, scores)
        meta_cols = ['gmap_id', 'year', 'month', 'week']
        output_df = chunk_df[meta_cols].copy()
    
        for k, v in scores.items(): 
            output_df[k] = v.astype(np.float16) 
        
        save_path = os.path.join(TEMP_DIR, f'chunk_{chunk_id:04d}.csv')
        output_df.to_csv(save_path, index=False)
    
        total_filtered_reviews += len(chunk_df)
        elapsed = time.time() - chunk_start
        if SAMPLE_MODE:
            print(f"✓ Chunk {chunk_id} done: {len(chunk_df):,} reviews ({TARGET_YEAR}) in {elapsed:.1f}s")
        else:
            print(f"✓ Chunk {chunk_id} done: {len(chunk_df):,} reviews in {elapsed:.1f}s")
    
        del embeddings, scores, output_df, chunk_df
        gc.collect()
        torch.cuda.empty_cache()
    
        chunk_id += 1
    return {"processed_reviews":total_filtered_reviews,"elapsed_seconds":time.time()-total_start}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',required=True,help='Authorized local encoder path or explicitly requested model ID')
    p.add_argument('--device',default=None)
    p.add_argument('--year',type=int,default=2019)
    p.add_argument('--all-years',action='store_true')
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--chunk-size',type=int,default=100000)
    a=p.parse_args()
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(a.model,device=a.device)
    print(run_semantic_generation(model,a.reviews,a.output_dir,
          year=None if a.all_years else a.year,batch_size=a.batch_size,chunk_size=a.chunk_size))

if __name__=='__main__':main()
