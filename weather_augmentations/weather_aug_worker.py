import cv2
import random
import torch
import os
import numpy as np
import logging
from utils.data_utils import ImageNetSubset
from pathlib import Path
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, DPMSolverMultistepScheduler
import argparse

IMAGE_DIR = Path("x")
OUT_DIR = Path("x")
LOG_DIR = OUT_DIR / "logs"

SDXL_MODEL_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
CONTROLNET_CANNY = "diffusers/controlnet-canny-sdxl-1.0"

MIN_SIZE = 768
MAX_SIZE = 1024
DEVICE = "cuda:0" #SLURM handles the GPU assignments to processes internally
STRENGTH = 0.95
GUIDANCE_SCALE = 11
CONTROLNET_DEPTH_SCALE = 0.6
CONTROLNET_CANNY_SCALE = 0.6
NUM_STEPS = 15

WEATHER_CONFIGS = {
    "rain": {
        "prompt": "{label}, torrential rain, heavy rain streaks, thick falling raindrops, completely drenched, dramatic downpour, large puddles, soaking wet reflective surfaces",
        "negative": "dry, sunshine, clear sky, bright colors, warm tones, snow, fog, style change, painting, cartoon, deformed, artifacts",
    },
    "fog": {
        "prompt": "{label}, extremely thick dense fog, heavy fog filling the entire scene, fog in the foreground and background, foreground objects obscured by fog, near zero visibility, impenetrable fog wall, thick fog covering the whole image, no clear areas, uniform dense fog everywhere",
        "negative": "clear visibility, sharp details, visible background, bright colors, sunshine, rain, snow, dry, style change, painting, cartoon, deformed, artifacts",
    },
    "snow": {
        "prompt": "{label}, whiteout blizzard, heavy snowfall, thick snowflakes filling the air, snow covering every surface, white overcast sky, snowstorm",
        "negative": "dry ground, bare ground, green grass, warm tones, sunshine, rain, fog, summer, style change, painting, cartoon, deformed, artifacts",
    },
    "sunny": {
        "prompt": "{label}, harsh midday sunlight, intense sunbeams, deep hard shadows, strong sunlight, warm golden light, bright glare",
        "negative": "overcast, clouds, rain, fog, snow, dark sky, muted colors, style change, painting, cartoon, deformed, artifacts",
    },
}

####################### Helpers ##############################

#VAE requires multiples of 64
def round64(v: int) -> int:
    r = v % 64
    if r == 0: return v
    return v + (64 - r) if r >= 32 else v - r

#We do this to preserve aspect ratios as well as possible
def preprocess(img: Image.Image) -> Image.Image:
    w, h = img.size
    long, short = max(w, h), min(w, h)
    if long > MAX_SIZE:
        scale = MAX_SIZE / long      # downscale large images
    elif short < MIN_SIZE:
        scale = MIN_SIZE / short     # upscale small images
    else:
        scale = 1.0                  # already in range, leave it
    nw, nh = round64(int(w * scale)), round64(int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)

#Get controlnet Canny map
def get_canny(img: Image.Image) -> Image.Image:
    arr  = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    high, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = cv2.Canny(gray, high * 0.4, high)
    return Image.fromarray(cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB))

#Get controlnet Depth map
def get_depth(img: Image.Image, processor, model) -> Image.Image:
    with torch.no_grad():
        d = model(**processor(images=img, return_tensors="pt").to(DEVICE)).predicted_depth
    w, h = img.size
    d = torch.nn.functional.interpolate(
        d.unsqueeze(1), (h, w), mode="bicubic", align_corners=False
    ).squeeze()
    d = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).byte().cpu().numpy()
    return Image.fromarray(np.stack([d] * 3, axis=-1))

#################### Pipeline ####################

def load_pipeline(): 
    print("Loading ControlNet models...")
    controlnet_depth = ControlNetModel.from_pretrained(CONTROLNET_DEPTH, torch_dtype=torch.bfloat16)
    controlnet_canny = ControlNetModel.from_pretrained(CONTROLNET_CANNY, torch_dtype=torch.bfloat16)

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        SDXL_MODEL_ID,
        controlnet=[controlnet_depth, controlnet_canny],
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe.set_progress_bar_config(disable=True)

    #channels_last memory format for faster conv throughput on CUDA
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    pipe.controlnet.nets[0] = pipe.controlnet.nets[0].to(memory_format=torch.channels_last)
    pipe.controlnet.nets[1] = pipe.controlnet.nets[1].to(memory_format=torch.channels_last)

    pipe.vae.to(torch.float32)
    pipe.controlnet.nets[0] = torch.compile(pipe.controlnet.nets[0], mode="default")
    pipe.controlnet.nets[1] = torch.compile(pipe.controlnet.nets[1], mode="default")
    pipe.unet = torch.compile(pipe.unet, mode="default")
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default")

    return pipe

########### Main script #################

def main(): 
    #Get the folders to process
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int) 
    args = parser.parse_args()
    #list of 100 folders
    all_folders = sorted(os.listdir("x"))  
    #Split into 4 chunks of 25
    chunk_size = len(all_folders) // 4
    start = args.task_id * chunk_size
    end   = start + chunk_size
    worker_dirs = all_folders[start:end]
    
    #Ensure reproducibility 
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    dataset = ImageNetSubset(root="x")
    synset_to_label = {dataset.wnids[idx]: name for name, idx in dataset.class_to_idx.items()}

    dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    dpt_model     = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").eval().to(DEVICE)
    pipe          = load_pipeline()
    
    #Create out dir if missing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    #Init logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"worker_{args.task_id}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )

    for class_dir in worker_dirs:
        class_path = IMAGE_DIR / class_dir
        out_class_path = OUT_DIR / class_dir
        out_class_path.mkdir(parents=True, exist_ok=True)

        label = synset_to_label.get(class_dir, class_dir)

        # Get all images in class
        images = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPEG"))
        ])

        #Assign weather to each image deterministically (25% each)
        weather_list = list(WEATHER_CONFIGS.keys())  #[rain, fog, snow, sunny]
        assigned_weathers = [weather_list[i % 4] for i in range(len(images))]

        for img_file, weather in zip(images, assigned_weathers):
            stem = Path(img_file).stem
            out_name = f"{stem}_{weather}.JPEG"
            out_path = out_class_path / out_name

            #Resume: skip if already done
            if out_path.exists(): 
                continue

            try:
                img = Image.open(class_path / img_file).convert("RGB")
                original_size = img.size  # (W, H) — save for resizing output back
                img = preprocess(img)
                canny = get_canny(img)
                depth = get_depth(img, dpt_processor, dpt_model)

                cfg = WEATHER_CONFIGS[weather]
                prompt   = cfg["prompt"].format(label=label)
                negative = cfg["negative"]

                #Generator within loop such that number of images
                #processed before is irrelevant
                generator = torch.Generator(device=DEVICE).manual_seed(SEED)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=img,
                    control_image=[depth, canny],
                    strength=STRENGTH,
                    guidance_scale=GUIDANCE_SCALE,
                    controlnet_conditioning_scale=[CONTROLNET_DEPTH_SCALE, CONTROLNET_CANNY_SCALE],
                    num_inference_steps=NUM_STEPS,
                    generator=generator,
                ).images[0]

                # Resize back to original dimensions
                result = result.resize(original_size, Image.LANCZOS)
                result.save(out_path, format="JPEG", quality=95)

            except Exception as e:
                logging.error(f"Failed {img_file}: {e}")
                continue
            
        logging.info(f"Finished class {class_dir} ({label})")

if __name__ == "__main__":
    main()
