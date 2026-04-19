import csv
import json
import os
from datetime import datetime


def analyze_jsonl_file(filepath):
    row_count = 0
    bad_lines = 0
    keys = set()
    file_size = os.path.getsize(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                row_count += 1
                if isinstance(record, dict):
                    keys.update(record.keys())
            except json.JSONDecodeError:
                bad_lines += 1

    avg_row_size = file_size / row_count if row_count else 0

    return {
        "file": os.path.basename(filepath),
        "type": "jsonl",
        "rows": row_count,
        "size_mb": round(file_size / (1024 ** 2), 2),
        "avg_row_bytes": round(avg_row_size, 2),
        "num_columns_or_keys": len(keys),
        "malformed_lines": bad_lines,
    }

def analyze_csv_file(filepath):
    file_size = os.path.getsize(filepath)

    row_count = 0
    num_columns = 0

    total_chars = 0
    total_words = 0
    max_chars = 0
    empty_posts = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        num_columns = len(reader.fieldnames) if reader.fieldnames else 0

        # Detect correct text column
        if "body" in reader.fieldnames:
            text_col = "body"
        elif "text" in reader.fieldnames:
            text_col = "text"
        else:
            print(f"Skipping {os.path.basename(filepath)}. No 'body' or 'text' column.")
            return None

        for row in reader:
            row_count += 1

            text = row.get(text_col, "")
            if text is None:
                text = ""

            text = text.strip()

            if text == "" or text.lower() == "[deleted]":
                empty_posts += 1
                continue

            char_len = len(text)
            word_len = len(text.split())

            total_chars += char_len
            total_words += word_len
            max_chars = max(max_chars, char_len)

    avg_row_size = file_size / row_count if row_count else 0

    valid_posts = row_count - empty_posts
    avg_chars = total_chars / valid_posts if valid_posts else 0
    avg_words = total_words / valid_posts if valid_posts else 0

    empty_pct = (empty_posts / row_count * 100) if row_count else 0

    return {
        "file": os.path.basename(filepath),
        "type": "csv",
        "rows": row_count,
        "size_mb": round(file_size / (1024 ** 2), 2),
        "avg_row_bytes": round(avg_row_size, 2),
        "num_columns_or_keys": num_columns,
        "avg_post_chars": round(avg_chars, 2),
        "avg_post_words": round(avg_words, 2),
        "max_post_chars": max_chars,
        "empty_post_pct": round(empty_pct, 2),
        "malformed_lines": 0,
    }

def analyze_jsonl_directory(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            path = os.path.join(directory, filename)
            results.append(analyze_jsonl_file(path))
    return results


def analyze_csv_directory(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)
            result = analyze_csv_file(path)
            if result is not None:
                results.append(result)
    return results

def write_log(results, log_path):
    jsonl_files = [r for r in results if r["type"] == "jsonl"]
    csv_files = [r for r in results if r["type"] == "csv"]

    total_rows = sum(r["rows"] for r in results)
    total_size_mb = sum(r["size_mb"] for r in results)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("DATASET ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        # JSONL First
        if jsonl_files:
            f.write("JSONL FILES\n\n")
            for r in jsonl_files:
                f.write(f"File: {r['file']}\n")
                f.write(f"Rows: {r['rows']}\n")
                f.write(f"Size (MB): {r['size_mb']}\n")
                f.write(f"Avg Row Size (bytes): {r['avg_row_bytes']}\n")
                f.write(f"Keys: {r['num_columns_or_keys']}\n")
                f.write(f"Malformed Lines: {r['malformed_lines']}\n")
                f.write("\n")

        # Then CSV
        if csv_files:
            f.write("CSV FILES\n\n")
            for r in csv_files:
                f.write(f"File: {r['file']}\n")
                f.write(f"Rows: {r['rows']}\n")
                f.write(f"Size (MB): {r['size_mb']}\n")
                f.write(f"Avg Row Size (bytes): {r['avg_row_bytes']}\n")
                f.write(f"Columns: {r['num_columns_or_keys']}\n")
                f.write(f"Avg Post Characters: {r['avg_post_chars']}\n")
                f.write(f"Avg Post Words: {r['avg_post_words']}\n")
                f.write(f"Max Post Characters: {r['max_post_chars']}\n")
                f.write(f"Empty Post %: {r['empty_post_pct']}\n")
                f.write("\n")

def write_csv_summary(results, csv_path):
    if not results:
        return

    fieldnames = results[0].keys()

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    jsonl_directory = "path_to_jsonl_directory"
    csv_directory = "path_to_csv_directory"

    jsonl_results = analyze_jsonl_directory(jsonl_directory)
    csv_results = analyze_csv_directory(csv_directory)

    results = jsonl_results + csv_results
    write_log(results, "dataset_analysis.log")
    write_csv_summary(results, "dataset_analysis_summary.csv")

    print("Analysis complete.")
    print("Log written to dataset_analysis.log")
    print("CSV summary written to dataset_analysis_summary.csv")
