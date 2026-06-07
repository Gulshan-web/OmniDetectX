import sqlite3
from collections import Counter

DB_NAME = "omnidetect.db"

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("SELECT * FROM scan_history ORDER BY scan_date DESC")
rows = cursor.fetchall()

print("\n========== OmniDetect X Scan History ==========\n")

if not rows:
    print("No scan history found.")
else:
    total_scans = len(rows)

    real_count = 0
    ai_count = 0
    all_objects = []

    for row in rows:
        objects = row[3]
        authenticity = row[4]

        if "Real" in authenticity:
            real_count += 1
        elif "AI" in authenticity or "Fake" in authenticity:
            ai_count += 1

        if objects:
            all_objects.extend(
                [obj.strip() for obj in objects.split(",")]
            )

    print(f"Total Scans: {total_scans}")
    print(f"Real Images: {real_count}")
    print(f"AI/Fake Images: {ai_count}")

    if all_objects:
        most_common = Counter(all_objects).most_common(5)

        print("\nMost Detected Objects:")
        for obj, count in most_common:
            print(f"- {obj}: {count} times")

    print("\nRecent Scan History:\n")

    for row in rows[:10]:
        scan_id = row[0]
        image_name = row[1]
        scan_date = row[2]
        objects = row[3]
        authenticity = row[4]
        confidence = row[5]

        print(f"ID: {scan_id}")
        print(f"Image: {image_name}")
        print(f"Date: {scan_date}")
        print(f"Objects: {objects}")
        print(f"Authenticity: {authenticity}")
        print(f"Confidence: {confidence}%")
        print("-" * 45)

conn.close()