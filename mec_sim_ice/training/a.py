from collections import Counter
import json, glob

counter = Counter()
for f in glob.glob("trainer_output/*/scene_*/*.json"):
    data = json.load(open(f))
    for d in data["detections"]:
        counter[d["class"]] += 1

for cls, n in sorted(counter.items(), key=lambda x: x[1]):
    print(f"{cls}: {n}")
