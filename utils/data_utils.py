import ast
import pandas as pd
from pathlib import Path

from utils.logger import Logger

def _safe_eval(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except: return None
    return x

def _norm_triplet(t):
    """
    Normalizes a single triplet 't'.
    't' can be:
      - a dict: {'sub':..., 'rel':..., 'obj':...}
      - a tuple/list of length 6: (sub, sub_type, rel, rel_category, obj, obj_type)
      - a tuple/list of length >= 3: (sub, rel, obj, ...)
    Returns a clean dict or None if invalid.
    """
    if isinstance(t, dict):
        sub = t.get('sub') or t.get('subject') or t.get('s')
        rel = t.get('rel') or t.get('predicate') or t.get('p')
        obj = t.get('obj') or t.get('object') or t.get('o')
        if not (isinstance(sub, str) and isinstance(rel, str) and isinstance(obj, str)): return None
        sub_type = t.get('sub_type') or 'UNK'
        obj_type = t.get('obj_type') or 'UNK'
        rel_cat = t.get('rel_category') or 'UNK'
        w = t.get('w') or 1.0
        try: w = float(w)
        except: w = 1.0
        return {'sub': sub, 'rel': rel, 'obj': obj,
                'sub_type': sub_type, 'obj_type': obj_type, 'rel_category': rel_cat, 'w': w}

    if isinstance(t, (list, tuple)):
        # Format found in legacy CSV: (sub, sub_type, rel, rel_category, obj, obj_type)
        if len(t) == 6:
            return {
                'sub': str(t[0]), 'sub_type': str(t[1]),
                'rel': str(t[2]), 'rel_category': str(t[3]),
                'obj': str(t[4]), 'obj_type': str(t[5]),
                'w': 1.0
            }
        # Fallback for simpler tuples
        elif len(t) >= 3:
            return {
                'sub': str(t[0]), 'rel': str(t[1]), 'obj': str(t[2]),
                'sub_type': 'UNK', 'obj_type': 'UNK', 'rel_category': 'UNK',
                'w': 1.0
            }
    return None

def load_df_from_csv(path: str | Path, strategy: str = "original") -> pd.DataFrame:
    path = Path(path)
    Logger.info(f"Loading {path} using strategy='{strategy}'...")
    if not path.exists():
        Logger.info(f"Error: File {path} does not exist.")
        return pd.DataFrame()

    raw = pd.read_csv(path)

    def select_triplets_content(row):
        orig = str(row.get("output_triplets", "[]"))

        if "Revised triplets" not in row.index:
            return orig

        rev = str(row["Revised triplets"]).strip()
        rev_lower = rev.lower()

        is_valid_list = rev.startswith("[") and rev.endswith("]")
        is_good_enough = "evaluation meets expectation" in rev_lower

        # STRATEGY SELECTION
        if strategy == "original":
            return orig
        elif strategy == "fallback":
            if is_valid_list: return rev
            return orig
        elif strategy == "strict":
            if is_valid_list: return rev
            if is_good_enough: return orig
            return "[]"
        return orig

    raw["selected_triplets"] = raw.apply(select_triplets_content, axis=1)
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce', utc=True).dt.tz_convert(None)
    raw["parsed_triplets"] = raw["selected_triplets"].apply(_safe_eval)
    raw = raw.dropna(subset=["parsed_triplets"]).copy()

    rows = []
    for _, r in raw.iterrows():
        ts = r['date']
        if pd.isna(ts): continue
        tlist = r["parsed_triplets"]

        if not isinstance(tlist, (list, tuple)) or len(tlist) == 0: continue
        for t in tlist:
            d = _norm_triplet(t)
            if d:
                d['date'] = ts
                rows.append(d)

    if not rows:
        Logger.info("No valid triplets extracted from rows.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for c in ['sub_type', 'obj_type', 'rel_category']:
        if c not in df.columns: df[c] = 'UNK'
    if 'w' not in df.columns: df['w'] = 1.0

    df = df.dropna(subset=['sub', 'rel', 'obj', 'date'])
    Logger.info(f"Loaded {len(df)} triplets after filtering.")
    return df
