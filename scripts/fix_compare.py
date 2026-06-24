path = "benchmark/compare_stitching.py"
text = open(path, encoding="utf-8").read()
bad = '''    if prov.get("type") == "silver_bootstrap":
        results["notes"] = (
            "- Ground truth is silver bootstrap (not human verified); replace with annotated GT before decisions.
"
            "- Stitching thresholds / temporal tau / IoU weights / graph strategy are frozen during benchmark."
        )'''
good = '''    if prov.get("type") == "silver_bootstrap":
        results["notes"] = (
            "- Ground truth is silver bootstrap (not human verified); replace with annotated GT before decisions.\\n"
            "- Stitching thresholds / temporal tau / IoU weights / graph strategy are frozen during benchmark."
        )'''
if bad not in text:
    raise SystemExit("pattern not found")
open(path, "w", encoding="utf-8", newline="\n").write(text.replace(bad, good, 1))
print("fixed")
