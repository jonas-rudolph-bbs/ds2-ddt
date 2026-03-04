import re
import numpy as np
from collections import Counter
import pandas as pd

def char_class(ch: str):
    if 'A' <= ch <= 'Z': return 'A'
    if 'a' <= ch <= 'z': return 'a'
    if '0' <= ch <= '9': return 'd'
    if ch in "-_": return '_'
    return '.'  # fallback

CLASS_TO_REGEX = {
    'A': r'[A-Z]',
    'a': r'[a-z]',
    'd': r'\d',
    '_': r'[-_]',
    '.': r'.',
}

def compress_classes(classes):
    # classes like: ['A','A','d','d','d'] -> [('A',2),('d',3)]
    runs = []
    for c in classes:
        if not runs or runs[-1][0] != c:
            runs.append([c, 1])
        else:
            runs[-1][1] += 1
    return [(c,n) for c,n in runs]

def suggest_regex(series: pd.Series, min_samples=50, min_coverage=0.8, max_len=64):
    vals = [str(v) for v in series.tolist() if (v is not None and str(v).strip() != "")]
    if len(vals) < min_samples:
        return None

    # filter very long strings
    vals = [v for v in vals if len(v) <= max_len]
    if len(vals) < min_samples:
        return None

    lengths = [len(v) for v in vals]
    L, cnt = Counter(lengths).most_common(1)[0]
    # require dominant length
    if cnt / len(vals) < 0.8:
        return None

    same_len = [v for v in vals if len(v) == L]
    # infer class per position by majority vote
    pos_classes = []
    for i in range(L):
        cs = [char_class(v[i]) for v in same_len]
        majority = Counter(cs).most_common(1)[0][0]
        pos_classes.append(majority)

    runs = compress_classes(pos_classes)
    pattern = '^' + ''.join(
        f"{CLASS_TO_REGEX[c]}{{{n}}}" for c,n in runs
    ) + '$'

    rgx = re.compile(pattern)
    coverage = np.mean([bool(rgx.match(v)) for v in vals])
    examples = [v for v in vals if bool(rgx.match(v))]

    if coverage >= min_coverage:
        return {"pattern": pattern, "coverage": float(coverage), "length_mode": int(L), "examples": examples}
    return None
