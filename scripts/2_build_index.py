import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import os

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows

# Creates pairs (prompt_id, prompt_text, prompt_speaker) -> (reply_id, reply_text, reply_speaker) when the speaker changes

def build_pairs(df_text):
    idx = df_text.index
    rows = []
    for i in range(len(df_text) - 1):
        r1 = df_text.iloc[i]
        r2 = df_text.iloc[i + 1]
        s1, s2 = r1.get("speaker"), r2.get("speaker")
        t1, t2 = r1.get("text"),    r2.get("text")
        if (
            pd.notna(s1) and pd.notna(s2) and str(s1) != "" and str(s2) != "" and
            pd.notna(t1) and pd.notna(t2) and str(t1) != "" and str(t2) != "" and
            s1 != s2
        ):
            rows.append({
                "prompt_id": idx[i],
                "prompt_text": str(t1),
                "prompt_speaker": str(s1),
                "reply_id": idx[i + 1],
                "reply_text": str(t2),
                "reply_speaker": str(s2),
            })
    return pd.DataFrame(rows)

def main(INPUT_JSONL, OUT_DIR):
    print("Loading JSONL")
    data = load_jsonl(INPUT_JSONL)
    if not data:
        print("Check messages is in processed!")
        return

    df = pd.DataFrame(data)

    print("\n Constructing words TF IDF")
    tfidf_words = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=4,
        max_df=0.8,
        max_features=80000,
        lowercase=True,
        strip_accents=None,
        sublinear_tf=True,
        norm="l2"
    )
    X_words = tfidf_words.fit_transform(df["text"].tolist())

    with open(os.path.join(OUT_DIR, "vectorizer_words.pkl"), "wb") as f:
        pickle.dump(tfidf_words, f)
    sparse.save_npz(os.path.join(OUT_DIR, "tfidf_words.npz"), X_words)

    print("\n Constructing characters TF IDF")
    tfidf_chars = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_features=80000,
        sublinear_tf=True,
        norm="l2"
    )
    X_chars = tfidf_chars.fit_transform(df["text"].tolist())

    with open(os.path.join(OUT_DIR, "vectorizer_chars.pkl"), "wb") as f:
        pickle.dump(tfidf_chars, f)
    sparse.save_npz(os.path.join(OUT_DIR, "tfidf_chars.npz"), X_chars)

    print("\n🔗 Constructing pairs (prompt -> reply)")
    pairs = build_pairs(df[["text","speaker","timestamp"]])
    pairs_path = os.path.join(OUT_DIR, "pairs.parquet")
    pairs.to_parquet(pairs_path, index=False)

    Xw_pairs = tfidf_words.transform(pairs["prompt_text"].tolist())
    Xc_pairs = tfidf_chars.transform(pairs["prompt_text"].tolist())
    sparse.save_npz(os.path.join(OUT_DIR, "Xw_pairs.npz"), Xw_pairs)
    sparse.save_npz(os.path.join(OUT_DIR, "Xc_pairs.npz"), Xc_pairs)

if __name__ == "__main__":
    main("C:/Users/Iriondo Delgado/Documents/chatbot/processed/messages.jsonl", "C:/Users/Iriondo Delgado/Documents/chatbot/index")

