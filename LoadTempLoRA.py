import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.checkpoint_pickle

from .utils import download_file, load_torch_bin

class LoadTempLoRA:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", ),
                "ckpt_url": ("STRING", {"default": ""}),
                "ckpt_type": (["safetensors", "other"], {"default": "safetensors"}),
                "download_split": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "temporary_loaders"

    def load_lora(self, model, clip, ckpt_url, ckpt_type, download_split, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == ckpt_url:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            bin, file_name = download_file(ckpt_url, download_split)
            if bin is None:
                raise file_name if file_name is not None else Exception("Download failed.")

            lora = load_torch_bin(bin, ckpt_type=="safetensors" or file_name.endswith(".safetensors"), safe_load=True)
            self.loaded_lora = (ckpt_url, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)
