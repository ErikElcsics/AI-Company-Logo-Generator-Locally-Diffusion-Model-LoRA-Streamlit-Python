# AI Logo Generator - Create Your Company Logo Locally

Create your professional, stunning company logos in seconds using the power of Stable Diffusion XL (SDXL), LoRA's and Streamlit! Customize styles, apply presets, and generate logos locally—perfect for startups, brands, and creators.

LoRA (Low-Rank Adaptation) is a technique used to fine-tune large AI models efficiently by adding a few small, trainable layers instead of retraining the whole model. This makes it faster, cheaper, and easier to customize models for specific tasks like logo design, art styles, or voice cloning.

This app runs entirely offline after downloading the models. No image generation is sent to the cloud!

## Features

- Generate logos with Stable Diffusion XL + custom LoRA support
- Style preset themes: Tech, Food, Luxury, Nature
- Prompt Selectable styles modifier checkboxes (10 styles) : Minimal, 3D, Typography, Gradient & more  
- “I’m Feeling Lucky” button - creative mode, random prompt generator
- Download buttons for generated logos for each logo
- Rate each logo on UI with Rating slider per image
- Powered by locally hosted AI models
- Prompt preview display and copy button 
- Custom text prompt input   
- Adjustable number of images to generate  
- Regenerate last prompt button  
- Image display in responsive columns  
- User rating slider for each image  
- Sidebar tooltips for guidance  
- Clean, styled title and descriptions with HTML  
- Progress bar during generation


## Style Options

- Colorful 🎨  
- Black & White 🖤  
- Detailed 🔍  
- Minimalistic 🧼  
- Circle ⭕  
- 3D 🧊  
- Vintage 🧵  
- Futuristic 🤖  
- Gradient 🌈  
- Typography-Focused 🔠  
- Square 🟥  
- Rectangle 🟦  
- Diamond 🔷  


## Presets

- Tech → Minimal, Futuristic, Typography
- Food → Colorful, Vintage
- Luxury → Black & White, Detailed
- Nature → Colorful, Detailed, Vintage


## Technologies Used

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Diffusers Library (SDXL)](https://github.com/huggingface/diffusers)
- [LoRA for logo design tuning]
- [Streamlit](https://streamlit.io/)
- Python 3.10+
- Local model inference with CPU offloading
- Streamlit



## 🛠️ Setup Instructions

1. Clone the repo


- git clone https://github.com/ErikElcsics/AI-Company-Logo-Generator-Locally-Diffusion-Model-LoRA-Streamlit-Python.git

- cd ai-logo-generator


2. install dependencies
- pip install streamlit ctransformers transformers
- pip install accelerate pelt
- PyTorch for GPU, but use for your GPU
	- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- pip install diffusers


3. Download the models

You'll need the following model paths locally:
- Stable Diffusion XL Base Model
- LoRA logo fine-tuned weights

> Place your model files under `/models/` or configure paths manually in the script.

4. Run the app in Terminal
streamlit run AILogoGenerator.py


## Folder Structure


![image](https://github.com/user-attachments/assets/8e7796df-8100-4f75-b974-50c00d20612f)



## How It Works

- Loads SDXL from HuggingFace via `DiffusionPipeline`
- Applies a logo-focused LoRA (LogoRedmondV2)
- Combines your text + selected styles into a smart prompt
- Generates one or more logo variants
- Offers download + rating interface


## Local-First

This app runs entirely offline after downloading the models. No image generation is sent to the cloud!


## 🤝 Contributing

Pull requests, suggestions, and feature ideas are always welcome!



## License

MIT License



## Credits

- [Hugging Face](https://huggingface.co/)
- [Stability AI - SDXL](https://stability.ai/)
- [Streamlit](https://streamlit.io/)



## 🌟 Show Some Love

If you like this project, give it a ⭐ on GitHub and share it with your friends!

![image](https://github.com/user-attachments/assets/30d116d2-050c-43dc-ae54-b562c402f85c)

