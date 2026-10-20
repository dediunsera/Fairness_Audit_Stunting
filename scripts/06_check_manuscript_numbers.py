"""Cross-check: every decimal number in the manuscript text must match (after rounding)
a value in results/*.csv (also as a percentage), or be listed as a known constant."""
import re, glob, json, subprocess
import numpy as np, pandas as pd
txt = subprocess.run(["node", "-e", "const C=require('./manuscript/'+(process.env.CONTENT||'content.js'));console.log(JSON.stringify([C.ABSTRACT.map(a=>a[1]).join(' '),C.BODY.filter(b=>b.p).map(b=>b.p).join(' ')]))"],
                     capture_output=True, text=True, env={**__import__('os').environ}).stdout
txt = " ".join(json.loads(txt)).replace("−", "-")
vals = []
for f in glob.glob("results/*.csv"):
    d = pd.read_csv(f)
    for c in d.columns:
        v = pd.to_numeric(d[c], errors="coerce").dropna().values
        vals.extend(v.tolist())
vals = np.array(vals)
KNOWN = {"0.7", "0.05", "1.0", "0.01", "0.1", "0.2", "0.3", "0.4", "21.5", "19.8", "14.2", "39.4", "3.13", "1.9.1", "3.0.5", "2.5.3", "1.18.1", "0.80", "69"}
bad = []
for m in re.finditer(r"-?\d+\.\d+", txt):
    s = m.group(); 
    if s in KNOWN or s.lstrip("-") in KNOWN: continue
    x = float(s); dec = len(s.split(".")[1])
    ok = np.any(np.round(vals, dec) == x) or np.any(np.round(vals * 100, dec) == x) or np.any(np.round(vals, dec) == -x)
    if not ok: bad.append(s)
print("decimals checked:", len(re.findall(r"-?\d+\.\d+", txt)), "| unmatched:", sorted(set(bad)))
