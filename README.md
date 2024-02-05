# ComfyUI-TemporaryLoader
This is a custom node of ComfyUI that downloads and loads models from the input URL. The model is temporarily downloaded into memory and not saved to storage.

This could be useful when trying out models or when using various models on machines with limited storage. Since the model is downloaded into memory, expect higher memory usage than usual.

## Installation:
1. Use `git clone https://github.com/pkpkTech/ComfyUI-TemporaryLoader` in your ComfyUI custom nodes directory
1. Use `pip install -r requirements.txt` in ComfyUI-TemporaryLoader directory

## Usage
The "Load Checkpoint (Temporary)" and "Load LoRA (Temporary)" nodes will be added to the temporary_loader category.

In addition to the standard Load node, the following items are added
- ckpt_url: URL of the model
- ckpt_type: With `auto`, it looks at the file extension of the downloaded file.
- download_split: Specify the number of splits for parallel downloading.
