import json
import numpy as np
from collections import Counter
from scipy import stats

def load_data(gen_path, label_path, is_jsonl=False):
    with open(gen_path) as f:
        gen = json.loads(f.read().strip().split("\n")[0]) if is_jsonl else json.load(f)
    with open(label_path) as f:
        labels = json.load(f)
    return gen, labels

def build_memory_pool(gen):
    pool = []
    for sess_idx, sess in enumerate(gen["sessions"]):
        for mp in sess.get("memory_points", []):
            pool.append({"content": mp["memory_content"], "session": sess_idx})
    return pool

def build_matrix(gen, labels):
    pool = build_memory_pool(gen)
    update_labels = {r["memory_content"]: r.get("memory_update_type") for r in labels["memory_update_records"]}
    qa_labels = {r["question"]: r.get("result_type") for r in labels["question_answering_records"]}
    matrix = Counter()
    for sess in gen["sessions"]:
        for q in sess.get("questions", []):
            ql = qa_labels.get(q["question"])
            if not ql: continue
            for ev in q.get("evidence", []):
                ec = ev.get("memory_content", "")
                matches = [m for m in pool if m["content"] == ec]
                if not matches: continue
                ul = update_labels.get(ec)
                if ul:
                    matrix[(ul, ql)] += 1
    return matrix

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (round(center - margin, 3), round(center + margin, 3))

def analyze(matrix, name):
    total = sum(matrix.values())
    off_diag = sum(v for (a,b),v in matrix.items() if a != b)
    oh = matrix.get(("Omission","Hallucination"), 0)
    ci = wilson_ci(oh, off_diag)
    p = stats.binomtest(oh, off_diag, p=0.5, alternative="greater").pvalue
    print(f"\n=== {name} ===")
    print(f"Total pairs: {total} | Off-diag: {off_diag} | O->H: {oh}")
    print(f"O->H / off-diag: {oh/off_diag*100:.1f}% CI{ci} p={p:.6f}")
    return matrix

if __name__ == "__main__":
    users = [
        ("User 0 (Martin)", "data/generation/halumem_pilot_user0_generation_results.json",
                            "data/eval/halumem_pilot_user0_FULL_results.json", False),
        ("User 3 (Sarah)",  "data/generation/halumem_pilot_user3_generation_results.jsonl",
                            "data/eval/halumem_pilot_user3_FULL_results.json", True),
        ("User 12 (Sharon)","data/generation/halumem_pilot_user12_generation.jsonl",
                            "data/eval/halumem_pilot_user12_detailed.json", True),
    ]
    matrices = {}
    for name, gp, lp, is_jsonl in users:
        gen, labels = load_data(gp, lp, is_jsonl)
        m = build_matrix(gen, labels)
        matrices[name] = analyze(m, name)

    pooled = Counter()
    for m in matrices.values():
        pooled += m
    analyze(pooled, "Pooled (All Users)")

