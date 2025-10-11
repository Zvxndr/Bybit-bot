import sqlite3
from datetime import datetime, timedelta

# Test timestamp generation
end_time = datetime.now()
start_time = end_time - timedelta(days=7)

print(f"Start time: {start_time}")
print(f"End time: {end_time}")

# Test creating a few timestamps
timestamps = []
current_time = start_time
for i in range(5):
    ts = int(current_time.timestamp() * 1000)
    timestamps.append(ts)
    print(f"Timestamp {i}: {ts} -> {datetime.fromtimestamp(ts/1000)}")
    current_time += timedelta(minutes=1)

print(f"\nTimestamp differences:")
for i in range(1, len(timestamps)):
    diff = timestamps[i] - timestamps[i-1]
    print(f"  Diff {i}: {diff} ms = {diff/1000/60} minutes")

# Check if there are duplicates
print(f"\nUnique timestamps: {len(set(timestamps))}")
print(f"Total timestamps: {len(timestamps)}")