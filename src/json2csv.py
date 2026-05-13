import json
import csv
import sys
import os


def json_to_csv(json_path: str, csv_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("imagesLib", [])

    rows = []
    for image in images:
        raw_name = image.get("name", "")
        name = os.path.splitext(raw_name)[0]

        annotations = image.get("annotations", [])
        if not annotations:
            #rows.append([name, "", "", "", ""])
            continue

        position = annotations[0].get("position", {})
        x      = position.get("x", "")
        y      = position.get("y", "")
        width  = position.get("width", "")
        height = position.get("height", "")
        rows.append([name, x, y, width, height])
        

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "width", "height"])
        writer.writerows(rows)

    print(f"Done — {len(rows)} row(s) written to: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_csv.py <input.json> <output.csv>")
        sys.exit(1)

    json_path = sys.argv[1]
    csv_path  = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"Error: file not found: {json_path}")
        sys.exit(1)

    json_to_csv(json_path, csv_path)
