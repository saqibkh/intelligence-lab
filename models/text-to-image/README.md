# intelligence-lab
This directory train AI and machine learning model for text-to-image generation

Step by step guide for text-to-image model refinement
1. Get labelled data with image and caption
2. Get models and weights for a specific model
3. Generate a few images as reference point
4. Train model and generate new weights
5. Generate a few images as reference points for comparision
6. Upload the model to huggingface if training produces good results

To RUN:
sudo apt-get install python3-venv <-- Install virtual environment
./setup_env.sh && source venv/bin/activate <-- Install required dependencies within the virtual environment
huggingface-cli login <-- Enter your token here
