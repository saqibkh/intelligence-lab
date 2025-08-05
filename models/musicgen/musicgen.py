import os
import re
import argparse
import scipy
import numpy as np
from pydub import AudioSegment
from transformers import pipeline


def parse_length(length_str):
    """Parses a duration string like '30s', '2m', or '1h' into seconds."""
    if not length_str:
        return None
    match = re.match(r'^(\d+)([smh])$', length_str.lower())
    if not match:
        raise ValueError("Invalid length format. Use number followed by 's', 'm', or 'h'. E.g. 30s, 2m, 1h.")
    
    value, unit = match.groups()
    value = int(value)
    if unit == 's':
        return value
    elif unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600


def sanitize_filename(name):
    """Sanitize prompt to be a valid filename."""
    return re.sub(r'\W+', '_', name.strip())


def generate_music(prompt: str, length: str = None):
    # Load the MusicGen model
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")
    
    print(f"Generating music for prompt: '{prompt}'")
    music = synthesiser(
            prompt, 
            forward_params={
                "temperature": 0.7,
                "do_sample": True
            }
    )
    
    sampling_rate = music["sampling_rate"]
    audio = music["audio"]
    
    prompt_clean = sanitize_filename(prompt)
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    temp_wav_path = os.path.join(output_dir, f"{prompt_clean}_temp.wav")
    scipy.io.wavfile.write(temp_wav_path, rate=sampling_rate, data=audio)
    
    # Load generated audio and prepare for looping
    sound = AudioSegment.from_wav(temp_wav_path)
    original_duration_ms = len(sound)
    
    total_duration_sec = parse_length(length) if length else original_duration_ms / 1000

    loops_needed = int(total_duration_sec * 1000 // original_duration_ms) + 1
    looped = sound * loops_needed
    final_audio = looped[:int(total_duration_sec * 1000)]
    
    output_path = os.path.join(output_dir, f"{prompt_clean}.mp3")
    final_audio.export(output_path, format="mp3")
    
    os.remove(temp_wav_path)
    print(f"✅ Music saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate music from a text prompt using MusicGen.")
    parser.add_argument("prompt", type=str, help="Text prompt to generate music from.")
    parser.add_argument("--length", type=str, default=None,
                        help="Optional total duration (e.g. 30s, 2m, 1h).")
    
    args = parser.parse_args()
    generate_music(args.prompt, args.length)


if __name__ == "__main__":
    main()

