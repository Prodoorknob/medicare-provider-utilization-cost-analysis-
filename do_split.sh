#!/bin/bash
SQL_DIR="C:/Users/rajas/Documents/ADS/medicare/medicare-provider-utilization-cost-analysis-/local_pipeline/_upload_sql"
CHUNK_DIR="C:/Users/rajas/Documents/ADS/medicare/medicare-provider-utilization-cost-analysis-/local_pipeline/_upload_sql_chunks"
mkdir -p "$CHUNK_DIR"

for FILE in "$SQL_DIR"/lstm_*.sql "$SQL_DIR"/s1_*.sql "$SQL_DIR"/s2_*.sql; do
    BASENAME=$(basename "$FILE" .sql)
    HEADER=$(head -1 "$FILE")
    CONFLICT=$(tail -1 "$FILE")
    TOTAL=$(wc -l < "$FILE")
    DATA_START=2
    DATA_END=$((TOTAL - 1))
    CHUNK=0
    LINE=$DATA_START
    while [ $LINE -le $DATA_END ]; do
        END=$((LINE + 149))
        if [ $END -gt $DATA_END ]; then
            END=$DATA_END
        fi
        OUTFILE="$CHUNK_DIR/${BASENAME}_c$(printf '%03d' $CHUNK).sql"
        echo "$HEADER" > "$OUTFILE"
        sed -n "${LINE},${END}p" "$FILE" >> "$OUTFILE"
        # Fix last line: remove trailing comma and add ON CONFLICT
        sed -i '$ s/,$//' "$OUTFILE"
        echo "$CONFLICT" >> "$OUTFILE"
        CHUNK=$((CHUNK + 1))
        LINE=$((END + 1))
    done
    echo "$BASENAME: $CHUNK chunks"
done
echo "Done splitting"
