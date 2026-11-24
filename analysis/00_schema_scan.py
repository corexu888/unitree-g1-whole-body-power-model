"""Schema scan: which key paths exist in which batch, and how often."""
import json, glob, os, collections, sys
ROOT = sys.argv[1]
def paths(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from paths(v, pre + "/" + k)
    elif isinstance(o, list) and o and isinstance(o[0], dict):
        yield from paths(o[0], pre + "[]")
    else:
        t = type(o).__name__
        if isinstance(o, list): t = "list%d" % len(o)
        yield pre + " :" + t
files = sorted(glob.glob(os.path.join(ROOT, "data", "**", "*telemetry.jsonl"), recursive=True))
bybatch = collections.defaultdict(collections.Counter)
nframes = collections.Counter(); bad = collections.Counter()
for f in files:
    batch = os.path.relpath(f, os.path.join(ROOT, "data")).split(os.sep)[0]
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: o = json.loads(line)
            except Exception as e:
                bad[batch] += 1; continue
            nframes[batch] += 1
            for p in set(paths(o)): bybatch[batch][p] += 1
allp = sorted(set(p for c in bybatch.values() for p in c))
batches = sorted(bybatch)
print("frames:", dict(nframes)); print("bad lines:", dict(bad))
print("path | " + " | ".join(b[:10] for b in batches))
for p in allp:
    print(p, "|", " | ".join("%d" % bybatch[b][p] if bybatch[b][p] != nframes[b] else "all" for b in batches))
