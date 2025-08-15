# Lullaby Generator & Music Video Creator

This project allows you to automatically generate lullaby music from text prompts, combine them into a longer track, and then create a music video using generated images.

---

## 📂 Project Overview

1. **`generate_lullabies.sh`**
   - Reads a list of prompts provided by the user.
   - Passes each prompt to `musicgen.py` to generate individual MP3 files.
   - Loops these MP3 files back-to-back until the total duration matches the length specified by the user.
   - Outputs:
     - Individual MP3 files for each prompt.
     - A final combined MP3 file.

2. **`create_video.py`**
   - Takes the generated MP3 files and the images stored under the `output/` directory.
   - Creates an MP4 music video by combining the audio with the images.
   - Outputs:
     - A music video (`.mp4`) that syncs visuals with the lullaby audio.

---

To RUN:
sudo apt-get install python3-venv <-- Install virtual environment
./setup_env.sh && source venv/bin/activate <-- Install required dependencies within the virtual environment
huggingface-cli login <-- Enter your token here

