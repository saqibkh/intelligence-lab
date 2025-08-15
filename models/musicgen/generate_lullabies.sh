#!/bin/bash

# File: generate_lullabies.sh
# Usage: bash generate_lullabies.sh

# Array of prompts (you can also read from a file if preferred)
PROMPTS=(
"magical piano lullaby under a starlit fairy forest"
"soft harp and gentle rain in an enchanted moon garden"
"soothing guitar lullaby with ocean waves and glowing fireflies"
"tender music box melody drifting through a cloud kingdom"
"calm flute and strings under the northern lights"
"gentle humming lullaby on a warm summer night"
"dreamy piano melody floating over pastel skies"
"warm cello and harp in a candlelit nursery"
"peaceful orchestral lullaby in a blooming flower meadow"
"soft ukulele and waves on a tropical moonlit beach"
"slow harp and piano duet in a crystal cave"
"gentle rain lullaby with faint chimes and firefly lights"
"music box melody with soft heartbeats and moonlight glow"
"floating lullaby on a paper boat under starry skies"
"serene flute and harp in a misty mountain valley"
"calm piano arpeggios with the sound of distant whales"
"quiet guitar lullaby in a forest with glowing lanterns"
"whisper-soft strings in a snowy winter wonderland"
"magical music box in a dream-filled cloud castle"
"slow waltz lullaby with shimmering fairy dust"
)

# Loop through prompts
for PROMPT in "${PROMPTS[@]}"; do
    echo "Generating: $PROMPT"

    # Run your musicgen script
    python3 musicgen.py "$PROMPT" --length 30s 
done
echo "✅ All files generated in: $OUTPUT_DIR"

