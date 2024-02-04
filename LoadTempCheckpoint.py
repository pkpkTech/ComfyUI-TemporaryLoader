import torch

import folder_paths

import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.checkpoint_pickle

from .utils import download_file, load_torch_bin

class LoadTempCheckpoint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_url": ("STRING", {"default": ""}),
                "ckpt_type": (["auto", "safetensors", "other"], {"default": "auto"}),
                "download_split": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1})
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION")
    FUNCTION = "load_checkpoint"

    CATEGORY = "temporary_loaders"

    def load_checkpoint(self, ckpt_url, ckpt_type, download_split, output_model=True, output_vae=True, output_clip=True, output_clipvision=True):

        bin, file_name = download_file(ckpt_url, download_split)

        if bin is None:
            raise file_name if file_name is not None else Exception("Download failed.")

        is_safetensors = file_name.endswith(".safetensors") if ckpt_type =="auto" else ckpt_type == "safetensors"
        sd = load_torch_bin(bin, is_safetensors)
        sd_keys = sd.keys()
        clip = None
        clipvision = None
        vae = None
        model = None
        model_patcher = None
        clip_target = None

        parameters = comfy.utils.calculate_parameters(sd, "model.diffusion_model.")
        unet_dtype = comfy.model_management.unet_dtype(model_params=parameters)
        load_device = comfy.model_management.get_torch_device()
        manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)

        class WeightsLoader(torch.nn.Module):
            pass

        model_config = comfy.model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
        model_config.set_manual_cast(manual_cast_dtype)

        if model_config is None:
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_url))

        if model_config.clip_vision_prefix is not None:
            if output_clipvision:
                clipvision = comfy.clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

        if output_model:
            inital_load_device = comfy.model_management.unet_inital_load_device(parameters, unet_dtype)
            offload_device = comfy.model_management.unet_offload_device()
            model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
            model.load_model_weights(sd, "model.diffusion_model.")

        if output_vae:
            # vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
            vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in "first_stage_model."}, filter_keys=True)
            vae_sd = model_config.process_vae_state_dict(vae_sd)
            vae = comfy.sd.VAE(sd=vae_sd)

        if output_clip:
            w = WeightsLoader()
            clip_target = model_config.clip_target()
            if clip_target is not None:
                clip = comfy.sd.CLIP(clip_target, embedding_directory=folder_paths.get_folder_paths("embeddings"))
                w.cond_stage_model = clip.cond_stage_model
                sd = model_config.process_clip_state_dict(sd)
                comfy.sd.load_model_weights(w, sd)

        left_over = sd.keys()
        if len(left_over) > 0:
            print("left over keys:", left_over)

        if output_model:
            model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device(), current_device=inital_load_device)
            if inital_load_device != torch.device("cpu"):
                print("loaded straight to GPU")
                comfy.model_management.load_model_gpu(model_patcher)

        return (model_patcher, clip, vae, clipvision)
