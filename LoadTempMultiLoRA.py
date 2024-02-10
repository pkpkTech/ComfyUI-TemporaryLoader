import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.checkpoint_pickle

import folder_paths

from .utils import download_file, load_torch_bin

class LoadTempMultiLoRA:
    CKPT_TYPE = ["auto", "safetensors", "other"]
    def __init__(self):
        self.loaded_lora = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", ),
                "ckpt_url": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "ckpt_type": (LoadTempMultiLoRA.CKPT_TYPE, {"default": "auto"}),
                "download_split": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_multi_lora"

    CATEGORY = "temporary_loaders"

    def load_multi_lora(self, model, clip, ckpt_url, ckpt_type, download_split, strength_model, strength_clip):
        model_lora, clip_lora = model, clip
        lora_dic = {}
        for line in ckpt_url.splitlines():
            if line.startswith("#"):
                continue
            else:
                info = line.split(":")
                if len(info) < 2:
                    continue
                if line.startswith("file:"):
                    file_name = line[5:]
                    model_lora, clip_lora = self.load_lora_file(model_lora, clip_lora, file_name, strength_model, strength_clip, lora_dic)
                    continue
                if line.startswith("http:") or line.startswith("https:"):
                    url = "{}:{}".format(info[0].strip(), info[1].strip())
                    model_lora, clip_lora = self.load_lora_url(model_lora, clip_lora, url, ckpt_type, download_split, strength_model, strength_clip, lora_dic)
                    continue
                try:
                    isFile = info[-2].strip().startswith("file")
                    url = info[-1].strip() if isFile else "{}:{}".format(info[-2].strip(), info[-1].strip())
                    strength_m = float(info[0]) if len(info[0]) > 0 else strength_model
                    strength_c = float(info[1]) if len(info[1]) > 0 else strength_clip
                    m_type = info[2].strip() if len(info) == 5 and info[2].strip() in LoadTempMultiLoRA.CKPT_TYPE else ckpt_type
                except:
                    raise Exception("Not according to format. The only accepted formats are\r\n{url}\r\n{strength_model}:{strength_clip}:{url}\r\n{strength_model}:{strength_clip}:{ckpt_type}:{url}")
                if isFile:
                    model_lora, clip_lora = self.load_lora_file(model_lora, clip_lora, url, strength_m, strength_c, lora_dic)
                else:
                    model_lora, clip_lora = self.load_lora_url(model_lora, clip_lora, url, m_type, download_split, strength_m, strength_c, lora_dic)

        tmp = self.loaded_lora
        self.loaded_lora = lora_dic
        tmp.clear()
        del tmp

        return (model_lora, clip_lora)

    def load_lora_url(self, model, clip, ckpt_url, ckpt_type, download_split, strength_model, strength_clip, lora_dic):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora = self.loaded_lora.get(ckpt_url)
        if lora is None:
            bin, file_name = download_file(ckpt_url, download_split)
            if bin is None:
                raise file_name if file_name is not None else Exception("Download failed.")

            is_safetensors = file_name.endswith(".safetensors") if ckpt_type =="auto" else ckpt_type == "safetensors"
            lora = load_torch_bin(bin, is_safetensors, safe_load=True)

        lora_dic[ckpt_url] = lora

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

    def load_lora_file(self, model, clip, lora_name, strength_model, strength_clip, lora_dic):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = self.loaded_lora.get(lora_path)
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        lora_dic[lora_path] = lora

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)
