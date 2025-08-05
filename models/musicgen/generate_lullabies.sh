#!/bin/bash

# File: generate_lullabies.sh
# Usage: bash generate_lullabies.sh

# Array of prompts (you can also read from a file if preferred)
PROMPTS=(
  "A gentle lullaby with soft piano, music box tones, and a slow, soothing melody. Calm and peaceful atmosphere, perfect for helping a baby fall asleep."
  "A soft music box lullaby with delicate plucked notes and a dreamy, slow melody. Very gentle and calming, perfect for bedtime."
  "A relaxing acoustic lullaby with fingerpicked guitar, soft humming, and a warm, comforting melody. Slow and peaceful, ideal for a baby's sleep."
  "A dreamy ambient lullaby with soft synth pads, gentle chimes, and a slowly drifting melody. Designed to calm and soothe babies to sleep."
  "A peaceful lullaby with soft wind chimes, slow harp melodies, and gentle sounds of nature like a flowing stream and rustling leaves. Calm and magical."
  "A magical lullaby with soft bells, harps, and dreamy orchestration. Slow tempo and comforting mood, like a bedtime story set to music."
  "A minimal lullaby with no percussion, just soft music box notes and a gentle melody. Very quiet and soothing, perfect for sleeping babies."
)

# Loop through prompts
for PROMPT in "${PROMPTS[@]}"; do
    echo "Generating: $PROMPT"

    # Run your musicgen script
    python3 musicgen.py "$PROMPT" --length 2m& 
done

echo "✅ All files generated in: $OUTPUT_DIR"

