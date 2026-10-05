"""Write <repo>/problems.yaml: chungmu, first Stage1 problem only, one Stage2 problem.
(The platform poses only the first enabled Stage2 entry anyway.)"""
import os, subprocess, sys, yaml
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
s2 = sys.argv[1]
d = yaml.safe_load(subprocess.check_output(["git", "-C", root, "show", "HEAD:problems.yaml"]))
for i, e in enumerate(d["stage1"]):
    e["enabled"] = (i == 0)
for e in d["stage2"]:
    e["enabled"] = (e["id"] == s2)
assert any(e["enabled"] for e in d["stage2"]), "unknown stage2 id %s" % s2
yaml.safe_dump(d, open(os.path.join(root, "problems.yaml"), "w"), sort_keys=False)
