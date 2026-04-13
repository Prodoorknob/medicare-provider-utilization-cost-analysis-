"""
Split large SQL INSERT files into smaller chunks of ~150 rows each.
Writes chunks to local_pipeline/_upload_sql_chunks/
"""
import os
import glob

SQL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_pipeline", "_upload_sql")
CHUNK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_pipeline", "_upload_sql_chunks")
CHUNK_SIZE = 150  # rows per chunk

os.makedirs(CHUNK_DIR, exist_ok=True)

patterns = ["lstm_*.sql", "s1_*.sql", "s2_*.sql"]
all_files = []
for pattern in patterns:
    matches = sorted(glob.glob(os.path.join(SQL_DIR, pattern)))
    all_files.extend(matches)

total_chunks = 0

for filepath in all_files:
    filename = os.path.basename(filepath)
    stem = filename.replace('.sql', '')

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # First line is the INSERT INTO ... VALUES header
    header_line = lines[0].strip()
    # Remove trailing " VALUES" and store
    if header_line.endswith(' VALUES'):
        insert_header = header_line  # Keep "INSERT INTO ... VALUES"
    else:
        insert_header = header_line

    # Data lines are lines[1:] - each is a tuple "(...)," or the last with "ON CONFLICT..."
    data_lines = []
    on_conflict = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith('ON CONFLICT'):
            on_conflict = stripped
        elif stripped:
            data_lines.append(stripped)

    # Split data_lines into chunks
    chunk_idx = 0
    for i in range(0, len(data_lines), CHUNK_SIZE):
        chunk = data_lines[i:i + CHUNK_SIZE]

        # Fix last line of chunk: remove trailing comma, add newline + ON CONFLICT
        chunk_text = []
        for j, row in enumerate(chunk):
            if j == len(chunk) - 1:
                # Last row: remove trailing comma if present
                if row.endswith(','):
                    row = row[:-1]
            chunk_text.append(row)

        sql = insert_header + "\n" + "\n".join(chunk_text) + "\n" + (on_conflict or "ON CONFLICT DO NOTHING;")

        chunk_filename = f"{stem}_c{chunk_idx:03d}.sql"
        chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(sql)

        chunk_idx += 1
        total_chunks += 1

    print(f"{filename}: {len(data_lines)} rows -> {chunk_idx} chunks")

print(f"\nTotal: {len(all_files)} files -> {total_chunks} chunks in {CHUNK_DIR}")
