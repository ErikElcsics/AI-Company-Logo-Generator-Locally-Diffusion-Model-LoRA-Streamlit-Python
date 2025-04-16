import sys
from pathlib import Path
from transformers import pipeline
import torch
import streamlit as st
import os
from diffusers import DiffusionPipeline
import ctransformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from io import BytesIO
from PIL import Image
import random

sys.path.append(str(Path(__file__).resolve().parents[2]))

############################################
# Image Generation Using SDXL Model
############################################

@st.cache_resource()
def load_model_local_sdxl(model_path_sdxl, model_path_sdxl_refiner=None, model_path_lora=None, torch_dtype=torch.float16, use_safetensors=True):
    base = DiffusionPipeline.from_pretrained(model_path_sdxl,
                                             torch_dtype=torch.float16,
                                             use_safetensors=True)
    if model_path_lora:
        base.load_lora_weights(model_path_lora)
    base.enable_model_cpu_offload()

    if model_path_sdxl_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            model_path_sdxl_refiner,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
        )
        refiner.enable_model_cpu_offload()
    else:
        refiner = None

    return base, refiner

def generate_image_local_sdxl(base, prompt, refiner=None, num_inference_steps=20, guidance_scale=15,
                               high_noise_frac=0.8, output_type="latent", verbose=False, temprature=0.7):
    if refiner:
        image = base(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=high_noise_frac,
                     output_type=output_type, verbose=verbose, guidance_scale=guidance_scale,
                     temprature=temprature).images
        image = refiner(prompt=prompt, num_inference_steps=num_inference_steps, denoising_start=high_noise_frac,
                        image=image, verbose=verbose).images[0]
    else:
        image = base(prompt=prompt, num_inference_steps=num_inference_steps,
                     guidance_scale=guidance_scale).images[0]

    return image

############################################
# Streamlit App UI
############################################

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧠 AI Logo Generator - Create Your Company Logo Locally!</h1>
    <p style='text-align: center; font-size: 18px;'>Create stunning, professional-looking logos in seconds</p>
""", unsafe_allow_html=True)

# App state
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "last_styles" not in st.session_state:
    st.session_state.last_styles = {}

# Style config
style_labels = [
    "Colorful", "Black & White", "Detailed", "Minimalistic", "Circle",
    "3D", "Vintage", "Futuristic", "Gradient", "Typography-Focused",
    "Square", "Rectangle", "Diamond"
]

style_emojis = [
    "🎨", "🖤", "🔍", "🧼", "⭕",
    "🧊", "🧵", "🤖", "🌈", "🔠",
    "🟥", "🟦", "🔷"
]

style_phrases = {
    "Colorful": "colorful",
    "Black & White": "black and white",
    "Minimalistic": "minimalistic",
    "Detailed": "detailed",
    "Circle": "circular design",
    "3D": "3D",
    "Vintage": "vintage style",
    "Futuristic": "futuristic",
    "Gradient": "gradient",
    "Typography-Focused": "typography-focused",
    "Square": "square format",
    "Rectangle": "rectangular format",
    "Diamond": "diamond shape"
}

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Customize Your Logo")

    preset = st.selectbox("🎯 Choose a preset", ["None", "Tech", "Food", "Luxury", "Nature"])

    if st.button("🎲 I'm Feeling Lucky"):
        samples = ["space unicorn", "minimal fox", "cyber dragon", "retro cassette", "digital panda"]
        st.session_state.feeling_lucky = random.choice(samples)
    else:
        st.session_state.feeling_lucky = None

    default_prompt = st.session_state.feeling_lucky if st.session_state.feeling_lucky else "car"
    user_input = st.text_input("✏️ Enter your prompt", value=default_prompt)

    # Define preset defaults
    selected_styles = set()
    if preset == "Tech":
        selected_styles.update(["Minimalistic", "Futuristic", "Typography-Focused"])
    elif preset == "Food":
        selected_styles.update(["Colorful", "Vintage"])
    elif preset == "Luxury":
        selected_styles.update(["Black & White", "Detailed", "Typography-Focused"])
    elif preset == "Nature":
        selected_styles.update(["Colorful", "Detailed", "Vintage"])

    num_images = st.slider("🖼️ Select number of logos to generate:", 1, 10, 2)

    st.markdown("**✨ Style Options:**")
    checkbox_states = {}
    for i, style in enumerate(style_labels):
        default_val = style in selected_styles
        checkbox_states[style] = st.checkbox(f"{style_emojis[i]} {style}", value=default_val)

# Load model
model_path_sdxl = ("C:../Models/models--stabilityai--stable-diffusion-xl-base-1.0/"
                   "snapshots/462165984030d82259a11f4367a4eed129e94a7b")
lora_path = "Loras/LogoRedmondV2-Logo-LogoRedmAF.safetensors"
base, refiner = load_model_local_sdxl(model_path_sdxl, None, lora_path)

# Prompt builder
lora_trigger = "logo of a "
prompt = lora_trigger + user_input

for style, active in checkbox_states.items():
    if active:
        prompt += f", {style_phrases[style]}"

st.session_state.last_prompt = prompt
st.session_state.last_styles = checkbox_states

# Prompt preview
with st.expander("🔍 See your final prompt"):
    st.code(prompt)

# Generate logos
if st.button("🚀 Generate Logos") or st.button("🔄 Regenerate Variants"):
    with st.spinner('Cooking up some logos... 🧑‍🍳'):
        progress_bar = st.progress(0)
        cols = st.columns(3)

        for i in range(num_images):
            generated_image = generate_image_local_sdxl(base, prompt)

            with cols[i % 3]:
                st.image(generated_image, caption=f"Logo {i + 1}", use_container_width=True)

                buf = BytesIO()
                generated_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(label="⬇️ Download", data=byte_im, file_name=f"logo_{i+1}.png", mime="image/png")

                st.slider("⭐ Rate this logo", 1, 5, 3, key=f"rating_{i}")

            progress = ((i + 1) / num_images)
            progress_bar.progress(int(progress * 100))
