import os
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, StableDiffusion3Pipeline
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
from datasets import Dataset
from accelerate import Accelerator
from tqdm import tqdm
from huggingface_hub import snapshot_download

# ========== Configuration ==========
BASE_MODEL = "stabilityai/stable-diffusion-3.5-large"
PROMPTS = [
    "A futuristic cityscape at dusk",
    "A medieval castle on a mountain during sunrise",
    "A surreal painting of a cat made of clouds"
]

# Build path relative to this script (no matter the working directory)
SCRIPT_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(SCRIPT_PATH)
IMAGE_DIR = os.path.join(BASE_DIR, "data/images")
CAPTION_DIR = os.path.join(BASE_DIR, "data/captions")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")
FINETUNED_DIR = os.path.join(BASE_DIR, "data/finetuned")

for d in [IMAGE_DIR, CAPTION_DIR, OUTPUT_DIR, FINETUNED_DIR]:
    os.makedirs(d, exist_ok=True)

# ROCm-compatible device
device = "cuda" if torch.cuda.is_available() else "cpu"

########################################################################################
#  These functions are for Stable Diffusion 3.5 Large
#
#
def model_already_downloaded(model_dir: str) -> bool:
    return os.path.exists(os.path.join(model_dir, "model_index.json"))

def load_sd_3_5_pipeline(model_dir: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(model_dir, exist_ok=True)

    # Download if not already present
    if not model_already_downloaded(model_dir):
        print(f"Downloading model: {BASE_MODEL} into {model_dir}")
        snapshot_download(
            repo_id=BASE_MODEL,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
    else:
        print(f"Model already exists in {model_dir}. Skipping download.")

    # Load the pipeline
    #pipe = DiffusionPipeline.from_pretrained(
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,  # for AMD ROCm
    ).to(device)

    print("Stable Diffusion 3.5 pipeline loaded successfully.")
    return pipe


#
#
#
########################################################################################







# ========== 1. Load or Download Model ==========
def load_pipeline(model_dir=BASE_MODEL, custom=False):
    print(f"🔄 Loading {'fine-tuned' if custom else 'base'} pipeline...")
    if BASE_MODEL == "stabilityai/stable-diffusion-3.5-large":
        return load_sd_3_5_pipeline(model_dir)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,  # No fp16 on AMD/ROCm
            use_safetensors=True,
        ).to(device)
        return pipe

# ========== 2. Generate Images ==========
def generate_images(pipe, prompts, out_folder):
    print("🖼️ Generating images...")
    for idx, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        image_path = os.path.join(out_folder, f"prompt_{idx:03d}.png")
        image.save(image_path)
        print(f"✅ Saved: {image_path}")

# ========== 3. Fine-tune the Model ==========
def finetune_model():
    print("🎓 Starting fine-tuning...")

    # Prepare data
    data = []
    for fname in os.listdir(IMAGE_DIR):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(IMAGE_DIR, fname)
        cap_path = os.path.join(CAPTION_DIR, fname.replace(".jpg", ".txt"))
        if not os.path.exists(cap_path):
            continue
        with open(cap_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        data.append({"image": img_path, "text": caption})

    if not data:
        print("⚠️ No image-caption pairs found. Skipping training.")
        return

    dataset = Dataset.from_list(data)

    # Load pipeline & UNet
    pipe = load_pipeline()
    unet = pipe.unet

    # Apply LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        task_type=TaskType.UNET
    )
    unet = get_peft_model(unet, lora_config)

    # Training setup
    unet.train()
    accelerator = Accelerator()
    unet, = accelerator.prepare(unet)
    optimizer = torch.optim.Adam(unet.parameters(), lr=5e-6)

    for epoch in range(1):  # increase epochs for real use
        for sample in tqdm(dataset, desc=f"Epoch {epoch}"):
            image = Image.open(sample["image"]).convert("RGB").resize((512, 512))
            image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)

            caption = sample["text"]
            text_inputs = pipe.tokenizer(
                caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77
            ).input_ids.to(device)
            encoder_hidden_states = pipe.text_encoder(text_inputs)[0]

            latents = pipe.vae.encode(image_tensor).latent_dist.sample()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    # Save
    unet.save_pretrained(FINETUNED_DIR)
    print("✅ Fine-tuned weights saved to:", FINETUNED_DIR)

# ========== Main ==========
if __name__ == "__main__":
    base_pipe = load_pipeline()
    generate_images(base_pipe, PROMPTS, OUTPUT_DIR)

    finetune_model()

    tuned_pipe = load_pipeline(model_dir=FINETUNED_DIR, custom=True)
    generate_images(tuned_pipe, PROMPTS, os.path.join(OUTPUT_DIR, "after_finetune"))

