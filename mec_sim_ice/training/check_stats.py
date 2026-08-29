import json, glob
from collections import defaultdict

occ_by_class = defaultdict(list)
area_by_class = defaultdict(list)

for f in glob.glob("trainer_output/*/scene_*/*.json"):
    data = json.load(open(f))
    for d in data["detections"]:
        occ_by_class[d["class"]].append(d["occlusion"])
        x1, y1, x2, y2 = d["bbox"]
        area_by_class[d["class"]].append((x2-x1)*(y2-y1))

print(f"{'class':<20}{'avg_occ':<10}{'avg_area(px2)':<15}{'n'}")
for cls in occ_by_class:
    occ = occ_by_class[cls]
    area = area_by_class[cls]
    print(f"{cls:<20}{sum(occ)/len(occ):<10.3f}{sum(area)/len(area):<15.0f}{len(occ)}")
