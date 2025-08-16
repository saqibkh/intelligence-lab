#!/bin/bash

# File: generate_lullabies.sh
# Usage: bash generate_lullabies.sh

# Array of prompts (you can also read from a file if preferred)
PROMPTS=(
"magical piano lullaby under a starlit fairy forest"
)

# Loop through prompts
for PROMPT in "${PROMPTS[@]}"; do
    echo "Generating: $PROMPT"

    # Run your musicgen script
    python3 musicgen.py "$PROMPT" --length 30s 
done
echo "✅ All files generated in: $OUTPUT_DIR"

