"""
Precache sample records for instantaneous dashboard performance.
"""
from src.data_loader import load_mitdb_record

for rec in ["100", "106", "119", "200"]:
    print(f"Pre-caching Record {rec} (60s)...")
    try:
        data = load_mitdb_record(rec, max_seconds=60)
        print(f"Record {rec} cached: {len(data['signal'])} samples, {len(data['ann_samples'])} beats.")
    except Exception as e:
        print(f"Notice caching {rec}: {e}")
print("Pre-caching complete!")
