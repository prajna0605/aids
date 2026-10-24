"""
ECG Insight AI - Utility Script to Pre-cache PhysioNet Records
Pre-caches key recordings (100, 106, 119, 200) for instant response times.
"""
from src.data_loader import load_mitdb_record

def precache_all():
    records = ["100", "106", "119", "200"]
    for rec in records:
        print(f"Pre-caching Record {rec} (60s)...")
        try:
            data = load_mitdb_record(rec, max_seconds=60)
            print(f"Record {rec} cached: {len(data['signal'])} samples, {len(data['ann_samples'])} beats.")
        except Exception as e:
            print(f"Notice caching {rec}: {e}")
    print("Pre-caching complete!")

if __name__ == "__main__":
    precache_all()
