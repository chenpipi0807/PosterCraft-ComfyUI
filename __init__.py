import gc
import os
import torch
import comfy
import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file,save_torch_file,ProgressBar
from diffusers import FluxTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device,is_torch_version
from tqdm import tqdm
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from .utils import log,calculate_shift,retrieve_timesteps
now_dir = os.path.dirname(__file__)

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.clip.configuration_clip import CLIPTextConfig
class FluxBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 10, "min": 0, "max": 19, "step": 1, "tooltip": "Number of transformer blocks to swap"}),
                "single_blocks_to_swap": ("INT", {"default": 19, "min": 0, "max": 38, "step": 1, "tooltip": "Number of transformer blocks to swap"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload img_emb to offload_device"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload time_emb to offload_device"}),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {"default": True, "tooltip": "Use non-blocking memory transfer for offloading, reserves more RAM but is faster"}),
            },
        }
    RETURN_TYPES = ("FLUXBLOCKSWAPARGS",)
    RETURN_NAMES = ("flux_block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "AIFSH/FluxModel"
    DESCRIPTION = "Settings for block swapping, reduces VRAM use by swapping blocks to CPU memory"

    def setargs(self, **kwargs):
        return (kwargs, )
    

#region Model loading
class FluxModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "load_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "flex_attention",
                    #"spargeattn", needs tuning
                    #"spargeattn_tune",
                    ], {"default": "sdpa"}),
                "block_swap_args": ("FLUXBLOCKSWAPARGS", ),
            }
        }

    RETURN_TYPES = ("FluxModel",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "AIFSH/FluxModel"

    def loadmodel(self, model, load_device,attention_mode="sdpa", block_swap_args=None):
        
        transformer = None
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        transformer_load_device = device if load_device == "main_device" else offload_device
        base_dtype = torch.bfloat16
        model_path = folder_paths.get_full_path("diffusion_models", model)
        # model_name = "PosterCraft-v1_RL_fp16.safetensors"
        # save_model_path = os.path.join(folder_paths.folder_names_and_paths["diffusion_models"][0][0],model_name)
        # print(save_model_path)
        if os.path.exists(model_path):
            sd = torch.load(model_path,weights_only=True)
            
            TRANSFORMER_CONFIG = {
            "attention_head_dim": 128,
            "axes_dims_rope": [
                16,
                56,
                56
            ],
            "guidance_embeds": True,
            "in_channels": 64,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "out_channels": None,
            "patch_size": 1,
            "pooled_projection_dim": 768
            }
            
            with init_empty_weights():
                transformer = FluxTransformer2DModel(**TRANSFORMER_CONFIG)
            
        else:
            # Try to load from local ComfyUI models directory
            model_path = folder_paths.get_full_path("diffusion_models", "PosterCraft-v1_RL_fp16.safetensors")
            
            # If exact name not found, try to find similar files
            if not model_path or not os.path.exists(model_path):
                # Try alternative file names (like with (1) suffix)
                alt_names = [
                    "PosterCraft-v1_RL_fp16(1).safetensors",
                    "PosterCraft-v1_RL_fp16 (1).safetensors"
                ]
                for alt_name in alt_names:
                    model_path = folder_paths.get_full_path("diffusion_models", alt_name)
                    if model_path and os.path.exists(model_path):
                        break
                else:
                    raise FileNotFoundError(f"PosterCraft model not found. Please download PosterCraft-v1_RL_fp16.safetensors and place it in ComfyUI/models/diffusion_models/")
            
            # Load from local safetensors file
            transformer = FluxTransformer2DModel(**TRANSFORMER_CONFIG)
            state_dict = load_torch_file(model_path)
            transformer.load_state_dict(state_dict, strict=False)
        transformer.eval()
        # transformer.to(transformer_load_device)
        comfy_model = FluxModel(
            FluxModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        
        log.info("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(), 
                desc=f"Loading transformer parameters to {transformer_load_device}", 
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype
            set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])
        
        del sd
        gc.collect()
                 
        # torch.save(transformer.state_dict(),model_path)
        comfy_model.diffusion_model = transformer
        comfy_model.load_device = transformer_load_device
        
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
        patcher.model.is_patched = False

        if load_device == "offload_device" and patcher.model.diffusion_model.device != offload_device:
            log.info(f"Moving diffusion model from {patcher.model.diffusion_model.device} to {offload_device}")
            patcher.model.diffusion_model.to(offload_device)
            gc.collect()
            mm.soft_empty_cache()

        transformer.block_swap_args = block_swap_args
        transformer.__class__.offload_device = offload_device
        transformer.__class__.main_device = device
        transformer.__class__.block_swap = block_swap
        transformer.__class__.attention_mode = attention_mode
        transformer.__class__.use_non_blocking = block_swap_args.get("use_non_blocking", True)

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)

        transformer.__class__.forward = block_forward
        return (transformer, )

class ClipTextModelEncoder:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model": (folder_paths.get_filename_list("text_encoders"),),
                "prompt":("STRING",),
                "offload_model":("BOOLEAN",{
                    "default":True,
                })
            }
        }

    RETURN_TYPES = ("ClipPromptEmbeds",)
    RETURN_NAMES = ("clip_prompt_embeds",)
    FUNCTION = "encode"
    CATEGORY = "AIFSH/FluxModel"
    def encode(self,model,prompt,offload_model):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        prompt = [prompt]
        model_path = folder_paths.get_full_path("text_encoders",model)
        save_model_path =  save_model_path = os.path.join(folder_paths.folder_names_and_paths["text_encoders"][0][0],
                                       "flux.1-dev_text_encoder_bf16.safetensors")
        if not os.path.exists(save_model_path):
            # Try to load from local ComfyUI models directory
            text_encoder_path = folder_paths.get_full_path("text_encoders", "flux.1-dev_text_encoder_bf16.safetensors")
            
            # If exact name not found, try to find similar files
            if not text_encoder_path or not os.path.exists(text_encoder_path):
                # Try alternative file names (like with (1) suffix)
                alt_names = [
                    "flux.1-dev_text_encoder_bf16(1).safetensors",
                    "flux.1-dev_text_encoder_bf16 (1).safetensors"
                ]
                for alt_name in alt_names:
                    text_encoder_path = folder_paths.get_full_path("text_encoders", alt_name)
                    if text_encoder_path and os.path.exists(text_encoder_path):
                        break
                else:
                    raise FileNotFoundError(f"Text encoder model not found. Please download flux.1-dev_text_encoder_bf16.safetensors and place it in ComfyUI/models/text_encoders/")
            
            # Load the text encoder from safetensors using local config
            with init_empty_weights():
                params = {
                    "attention_dropout": 0.0,
                    "bos_token_id": 0,
                    "dropout": 0.0,
                    "eos_token_id": 2,
                    "hidden_act": "quick_gelu",
                    "hidden_size": 768,
                    "initializer_factor": 1.0,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "layer_norm_eps": 1e-05,
                    "max_position_embeddings": 77,
                    "model_type": "clip_text_model",
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "pad_token_id": 1,
                    "projection_dim": 768,
                    "torch_dtype": "bfloat16",
                    "vocab_size": 49408
                }
                cfg = CLIPTextConfig(**params)
                text_encoder = CLIPTextModel(cfg)
            
            state_dict = load_torch_file(text_encoder_path)
            # Load weights using accelerate with error handling
            param_count = sum(1 for _ in text_encoder.named_parameters())
            for name, param in tqdm(text_encoder.named_parameters(), 
                desc=f"Loading text_encoder parameters to device", 
                total=param_count,
                leave=True):
                if name in state_dict:
                    dtype_to_use = torch.bfloat16
                    set_module_tensor_to_device(text_encoder, name, device=device, dtype=dtype_to_use, value=state_dict[name])
                else:
                    log.warning(f"Missing weight for {name} in text encoder state dict")
            text_encoder.text_model.embeddings.position_ids = text_encoder.text_model.embeddings.position_ids.to(device)

            torch.save(text_encoder.state_dict(),save_model_path)
        else:
            sd = torch.load(model_path)
            with init_empty_weights():
                params = {
                    "attention_dropout": 0.0,
                    "bos_token_id": 0,
                    "dropout": 0.0,
                    "eos_token_id": 2,
                    "hidden_act": "quick_gelu",
                    "hidden_size": 768,
                    "initializer_factor": 1.0,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "layer_norm_eps": 1e-05,
                    "max_position_embeddings": 77,
                    "model_type": "clip_text_model",
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "pad_token_id": 1,
                    "projection_dim": 768,
                    "torch_dtype": "bfloat16",
                    "vocab_size": 49408
                }

                cfg = CLIPTextConfig(**params)
                text_encoder = CLIPTextModel(cfg)
                param_count = sum(1 for _ in text_encoder.named_parameters())
                for name, param in tqdm(text_encoder.named_parameters(), 
                    desc=f"Loading text_encoder parameters to cuda", 
                    total=param_count,
                    leave=True):
                    dtype_to_use = torch.bfloat16
                    set_module_tensor_to_device(text_encoder, name, device=device, dtype=dtype_to_use, value=sd[name])
                text_encoder.text_model.embeddings.position_ids = text_encoder.text_model.embeddings.position_ids.to(device)
                del sd
        text_encoder.eval()
        
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(now_dir,"flux/tokenizer"))
        tokenizer_max_length = tokenizer.model_max_length
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            log.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        if offload_model:
            # text_encoder.to(offload_device)
            del text_encoder
            gc.collect()
            mm.unload_all_models()
            mm.soft_empty_cache()

        return prompt_embeds

class T5TextModelEncoder:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model": (folder_paths.get_filename_list("text_encoders"),),
                "prompt":("STRING",),
                "offload_model":("BOOLEAN",{
                    "default":True,
                })
            }
        }

    RETURN_TYPES = ("T5PromptEmbeds",)
    RETURN_NAMES = ("t5_prompt_embeds",)
    FUNCTION = "encode"
    CATEGORY = "AIFSH/FluxModel"
    def encode(self,model,prompt,offload_model):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        prompt = [prompt]
        model_path = folder_paths.get_full_path("text_encoders",model)
        tokenizer_2 = T5TokenizerFast.from_pretrained(os.path.join(now_dir,"flux/tokenizer_2"))
        save_model_path = os.path.join(folder_paths.folder_names_and_paths["text_encoders"][0][0],
                                       "flux.1-dev_text_encoder_2_bf16.safetensors")
        if not os.path.exists(save_model_path):
            # Try to load from local ComfyUI models directory  
            text_encoder_2_path = folder_paths.get_full_path("text_encoders", "flux.1-dev_text_encoder_2_bf16.safetensors")
            
            # If exact name not found, try to find similar files
            if not text_encoder_2_path or not os.path.exists(text_encoder_2_path):
                # Try alternative file names (like with (1) suffix)
                alt_names = [
                    "flux.1-dev_text_encoder_2_bf16(1).safetensors",
                    "flux.1-dev_text_encoder_2_bf16 (1).safetensors"
                ]
                for alt_name in alt_names:
                    text_encoder_2_path = folder_paths.get_full_path("text_encoders", alt_name)
                    if text_encoder_2_path and os.path.exists(text_encoder_2_path):
                        break
                else:
                    raise FileNotFoundError(f"Text encoder 2 model not found. Please download flux.1-dev_text_encoder_2_bf16.safetensors and place it in ComfyUI/models/text_encoders/")
            
            # Load the T5 text encoder from safetensors using local config
            with init_empty_weights():
                params = {
                    "classifier_dropout": 0.0,
                    "d_ff": 10240,
                    "d_kv": 64,
                    "d_model": 4096,
                    "decoder_start_token_id": 0,
                    "dense_act_fn": "gelu_new",
                    "dropout_rate": 0.1,
                    "eos_token_id": 1,
                    "feed_forward_proj": "gated-gelu",
                    "initializer_factor": 1.0,
                    "is_encoder_decoder": True,
                    "is_gated_act": True,
                    "layer_norm_epsilon": 1e-06,
                    "model_type": "t5",
                    "num_decoder_layers": 24,
                    "num_heads": 64,
                    "num_layers": 24,
                    "output_past": True,
                    "pad_token_id": 0,
                    "relative_attention_max_distance": 128,
                    "relative_attention_num_buckets": 32,
                    "tie_word_embeddings": False,
                    "torch_dtype": "bfloat16",
                    "use_cache": False,
                    "vocab_size": 32128
                }
                cfg = T5Config(**params)
                text_encoder_2 = T5EncoderModel(cfg)
            
            state_dict = load_torch_file(text_encoder_2_path)
            # Load weights using accelerate with error handling
            param_count = sum(1 for _ in text_encoder_2.named_parameters())
            for name, param in tqdm(text_encoder_2.named_parameters(), 
                desc=f"Loading text_encoder_2 parameters to cpu", 
                total=param_count,
                leave=True):
                if name in state_dict:
                    dtype_to_use = torch.bfloat16
                    set_module_tensor_to_device(text_encoder_2, name, device=offload_device, dtype=dtype_to_use, value=state_dict[name])
                else:
                    log.warning(f"Missing weight for {name} in text encoder 2 state dict")
            torch.save(text_encoder_2.state_dict(),save_model_path)
        else:
            sd = torch.load(model_path)
            with init_empty_weights():
                params = {
                "classifier_dropout": 0.0,
                "d_ff": 10240,
                "d_kv": 64,
                "d_model": 4096,
                "decoder_start_token_id": 0,
                "dense_act_fn": "gelu_new",
                "dropout_rate": 0.1,
                "eos_token_id": 1,
                "feed_forward_proj": "gated-gelu",
                "initializer_factor": 1.0,
                "is_encoder_decoder": True,
                "is_gated_act": True,
                "layer_norm_epsilon": 1e-06,
                "model_type": "t5",
                "num_decoder_layers": 24,
                "num_heads": 64,
                "num_layers": 24,
                "output_past": True,
                "pad_token_id": 0,
                "relative_attention_max_distance": 128,
                "relative_attention_num_buckets": 32,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "use_cache": False,
                "vocab_size": 32128
                }

                cfg = T5Config(**params)
                text_encoder_2 = T5EncoderModel(cfg)
            param_count = sum(1 for _ in text_encoder_2.named_parameters())
            for name, param in tqdm(text_encoder_2.named_parameters(), 
                desc=f"Loading text_encoder_2 parameters to cuda", 
                total=param_count,
                leave=True):
                dtype_to_use = torch.bfloat16
                set_module_tensor_to_device(text_encoder_2, name, device=device, dtype=dtype_to_use, value=sd[name])
            del sd
        text_encoder_2.eval()
        
        # text_encoder_2.to(device)
        max_sequence_length = 512
        tokenizer_max_length = tokenizer_2.model_max_length
        text_inputs = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer_2.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            log.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        if offload_model:
            # text_encoder_2.to(offload_device)
            del text_encoder_2
            gc.collect()
            mm.unload_all_models()
            mm.soft_empty_cache()


        return prompt_embeds

class FluxPromptEmbed:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pooled_prompt_embeds":("ClipPromptEmbeds",),
                "prompt_embeds":("T5PromptEmbeds",),
            }
        }
    RETURN_TYPES = ("TextEmbed",)
    RETURN_NAMES = ("text_embeds", )
    FUNCTION = "embed"
    CATEGORY = "AIFSH/FluxModel"

    def embed(self,pooled_prompt_embeds,prompt_embeds):
        text_embeds = {
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds":prompt_embeds,
        }
        return (text_embeds,)

class PosterCraftSample:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "transformer":("FluxModel",),
                "text_embeds":("TextEmbed",),
                "width":("INT",{
                    "default":832,
                }),
                "height":("INT",{
                    "default":1216,
                }),
                "num_inference_steps":("INT",{
                    "default":28,
                }),
                "guidance_scale":("FLOAT",{
                    "default":3.5,
                }),
                "enable_teacache":("BOOLEAN",{
                    "default":False,
                }),
                "rel_l1_thresh":("FLOAT",{
                    "default":0.25,
                    "tooltip":"0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup"
                }),
                "offload_model":("BOOLEAN",{
                    "default":True,
                }),
                "seed":("INT",{
                    "default":42,
                })
            }
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples", )
    FUNCTION = "sample"
    CATEGORY = "AIFSH/FluxModel"
    def get_batch_text_embed(self,text_embeds,batch_size):
        pooled_prompt_embeds = text_embeds['pooled_prompt_embeds'].unsqueeze(0)
        prompt_embeds = text_embeds['prompt_embeds'].unsqueeze(0)
        # log.info(f"prompt_embeds.shape:{prompt_embeds.shape}")
        num_images_per_prompt=1
        
        _, seq_len, _ = prompt_embeds.shape

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        
        # log.info(f"after prompt_embeds.shape:{prompt_embeds.shape}")
        return pooled_prompt_embeds,prompt_embeds,text_ids

    def sample(self,transformer,text_embeds,width,height,
               num_inference_steps,guidance_scale,enable_teacache,
               rel_l1_thresh,offload_model,seed):
        device = mm.get_torch_device()
        batch_size = 1
        param_count = sum(1 for _ in transformer.named_parameters())
        if enable_teacache:
            transformer.__class__.forward = teacache_forward
            transformer.__class__.enable_teacache = True
            transformer.__class__.cnt = 0
            transformer.__class__.num_steps = num_inference_steps
            transformer.__class__.rel_l1_thresh = rel_l1_thresh # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
            transformer.__class__.accumulated_rel_l1_distance = 0
            transformer.__class__.previous_modulated_input = None
            transformer.__class__.previous_residual = None
        else:
            transformer.__class__.forward = block_forward
        
        for name, param in tqdm(transformer.named_parameters(), 
            desc=f"Copy transformer parameters to {device}", 
            total=param_count,
            leave=True):
            dtype_to_use = torch.bfloat16
            if "block" not in name:
                set_module_tensor_to_device(transformer, name, device=device, dtype=dtype_to_use, value=param)
        block_swap_args = transformer.block_swap_args
        transformer.block_swap(
            block_swap_args["blocks_to_swap"] - 1 ,
            block_swap_args["single_blocks_to_swap"] - 1 ,
            block_swap_args["offload_txt_emb"],
            block_swap_args["offload_img_emb"],
        )

        num_images_per_prompt = 1
        generator = torch.Generator(device=device).manual_seed(seed)
        self.vae_scale_factor = 8
        scheduler = FlowMatchEulerDiscreteScheduler(
            base_image_seq_len = 256,
            base_shift=0.5,
            max_image_seq_len=4096,
            max_shift=1.15,
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
        )
        pooled_prompt_embeds,prompt_embeds,text_ids = self.get_batch_text_embed(text_embeds,batch_size)
        
        # 4. Prepare latent variables
        num_channels_latents = transformer.config.in_channels // 4
        # Store original dimensions for later use
        original_height, original_width = height, width
        
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )

        latents = latents.to(device)
        latent_image_ids = latent_image_ids.to(device)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
        guidance = guidance.to(device)

        comfy_par = ProgressBar(total=num_inference_steps)
        # 6. Denoising loop
        for i, t in tqdm(enumerate(timesteps),total=num_inference_steps):
            
           # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)
            
            comfy_par.update(1)

        
        latents = self._unpack_latents(latents,height=original_height,width=original_width,
                                           vae_scale_factor =self.vae_scale_factor)
        scaling_factor = 0.3611
        shift_factor = 0.1159
        latents = (latents / scaling_factor) + shift_factor
        samples = {
            "samples":latents
        }
        
        transformer_load_device = mm.unet_offload_device()
        if offload_model:
            for name, param in tqdm(transformer.named_parameters(), 
                    desc=f"Copy transformer parameters to {transformer_load_device}", 
                    total=param_count,
                    leave=True):
                dtype_to_use = torch.bfloat16
                set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=param)
            mm.unload_all_models()
            mm.soft_empty_cache()
        return (samples,)
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        # Note: height and width should already be adjusted in prepare_latents
        adjusted_height = 2 * (int(height) // (vae_scale_factor * 2))
        adjusted_width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, adjusted_height // 2, adjusted_width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), adjusted_height, adjusted_width)

        return latents

aifsh_dir = os.path.join(folder_paths.models_dir,"AIFSH")
from transformers import AutoModelForCausalLM, AutoTokenizer
class QwenRecapAgent:
    def __init__(self):
        self.device = mm.get_torch_device()
        self.model_path = os.path.join(aifsh_dir,"Qwen3-8B")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.model_kwargs = {"torch_dtype": "auto"}
        self.model_kwargs["device_map"] = None
            
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.model_kwargs)
        
        self.prompt_template = """You are an expert poster prompt designer. Your task is to rewrite a user's short poster prompt into a detailed and vivid long-format prompt. Follow these steps carefully:

            `**Step 1: Analyze the Core Requirements**
            Identify the key elements in the user's prompt. Do not miss any details.
            - **Subject:** What is the main subject? (e.g., a person, an object, a scene)
            - **Style:** What is the visual style? (e.g., photorealistic, cartoon, vintage, minimalist)
            - **Text:** Is there any text, like a title or slogan?
            - **Color Palette:** Are there specific colors mentioned?
            - **Composition:** Are there any layout instructions?

            **Step 2: Expand and Add Detail**
            Elaborate on each core requirement to create a rich description.
            - **Do Not Omit:** You must include every piece of information from the original prompt.
            - **Enrich with Specifics:** Add professional and descriptive details.
                - **Example:** If the user says "a woman with a bow", you could describe her as "a young woman with a determined expression, holding a finely crafted wooden longbow, with an arrow nocked and ready to fire."
            - **Fill in the Gaps:** If the original prompt is simple (e.g., "a poster for a coffee shop"), use your creativity to add fitting details. You might add "The poster features a top-down view of a steaming latte with delicate art on its foam, placed on a rustic wooden table next to a few scattered coffee beans."

            **Step 3: Handle Text Precisely**
            - **Identify All Text Elements:** Carefully look for any text mentioned in the prompt. This includes:
                - **Explicit Text:** Subtitles, slogans, or any text in quotes.
                - **Implicit Titles:** The name of an event, movie, or product is often the main title. For example, if the prompt is "generate a 'Inception' poster ...", the title is "Inception".
            - **Rules for Text:**
                - **If Text Exists:**
                    - You must use the exact text identified from the prompt.
                    - Do NOT add new text or delete existing text.
                    - Describe each text's appearance (font, style, color, position). Example: `The title 'Inception' is written in a bold, sans-serif font, integrated into the cityscape.`
                - **If No Text Exists:**
                    - Do not add any text elements. The poster must be purely visual.
            - Most posters have titles. When a title exists, you must extend the title's description. Only when you are absolutely sure that there is no text to render, you can allow the extended prompt not to render text.

            **Step 4: Final Output Rules**
            - **Output ONLY the rewritten prompt.** No introductions, no explanations, no "Here is the prompt:".
            - **Use a descriptive and confident tone.** Write as if you are describing a finished, beautiful poster.
            - **Keep it concise.** The final prompt should be under 100 words.

            ---
            **User Prompt:**
            {brief_description}"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt":("STRING",),
                "offload_model":("BOOLEAN",{
                    "default":True,
                })
            }            
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("recap_string", )
    FUNCTION = "recap"
    CATEGORY = "AIFSH/FluxModel"

    def recap(self,prompt,offload_model):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.model_kwargs)
        self.model.to(self.device)
        final_prompt = self.recap_prompt(prompt)
        if offload_model:
            self.model = None
            mm.soft_empty_cache()
        return (final_prompt,)

    def recap_prompt(self, original_prompt):
        full_prompt = self.prompt_template.format(brief_description=original_prompt)
        messages = [{"role": "user", "content": full_prompt}]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096, temperature=0.6)
            
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            full_response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            final_answer = self._extract_final_answer(full_response)
            
            if final_answer:
                return final_answer.strip()
            
            print("Qwen returned an empty answer. Using original prompt.")
            return original_prompt
        except Exception as e:
            print(f"Qwen recap failed: {e}. Using original prompt.")
            return original_prompt

    def _extract_final_answer(self, full_response):
        if "</think>" in full_response:
            return full_response.split("</think>")[-1].strip()
        if "<think>" not in full_response:
            return full_response.strip()
        return None
    


def get_module_memory_mb(module):
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            memory += param.nelement() * param.element_size()
    return memory / (1024 * 1024)  # Convert to MB


def block_swap(self, blocks_to_swap,
               single_blocks_to_swap,
               offload_txt_emb=False, 
               offload_img_emb=False
               ):
    log.info(f"Swapping {blocks_to_swap + 1} transformer blocks")
    log.info(f"Swapping {single_blocks_to_swap + 1} single transformer blocks")
    self.blocks_to_swap = blocks_to_swap
    self.single_blocks_to_swap = single_blocks_to_swap
    self.offload_img_emb = offload_img_emb
    self.offload_txt_emb = offload_txt_emb

    total_offload_memory = 0
    total_main_memory = 0
    
    self.pos_embed.to(self.main_device)
    self.time_text_embed.to(self.main_device)
    self.context_embedder.to(self.main_device)
    self.x_embedder.to(self.main_device)
    self.norm_out.to(self.main_device)
    self.proj_out.to(self.main_device)

    for b, block in tqdm(enumerate(self.transformer_blocks), total=len(self.transformer_blocks), desc="Initializing block swap"):
        block_memory = get_module_memory_mb(block)
        
        if b > self.blocks_to_swap:
            block.to(self.main_device)
            total_main_memory += block_memory
        else:
            block.to(self.offload_device, non_blocking=self.use_non_blocking)
            total_offload_memory += block_memory
    
    for b, block in tqdm(enumerate(self.single_transformer_blocks), total=len(self.single_transformer_blocks), desc="Initializing block swap"):
        block_memory = get_module_memory_mb(block)
        
        if b > self.single_blocks_to_swap:
            block.to(self.main_device)
            total_main_memory += block_memory
        else:
            block.to(self.offload_device, non_blocking=self.use_non_blocking)
            total_offload_memory += block_memory

    mm.soft_empty_cache()
    gc.collect()

    log.info("----------------------")
    log.info(f"Block swap memory summary:")
    log.info(f"Transformer blocks on {self.offload_device}: {total_offload_memory:.2f}MB")
    log.info(f"Transformer blocks on {self.main_device}: {total_main_memory:.2f}MB")
    log.info(f"Total memory used by transformer blocks: {(total_offload_memory + total_main_memory):.2f}MB")
    log.info(f"Non-blocking memory transfer: {self.use_non_blocking}")
    log.info("----------------------")



class FluxModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


from comfy.latent_formats import Flux
latent_format = Flux

class FluxModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = latent_format
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

from typing import Any, Dict, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

def block_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                log.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            log.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            log.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if index_block <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                block.to(self.main_device)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            if index_block <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if index_block <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                block.to(self.main_device)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
            if index_block <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                log.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            log.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            log.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        if self.enable_teacache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            self.transformer_blocks[0].to(self.main_device)
            modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp 
            self.cnt += 1 
            if self.cnt == self.num_steps:
                self.cnt = 0           
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for index_block, block in enumerate(self.transformer_blocks):
                    if index_block <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                        block.to(self.main_device)
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                    
                    if index_block <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                        block.to(self.offload_device, non_blocking=self.use_non_blocking)

                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                for index_block, block in enumerate(self.single_transformer_blocks):
                    if index_block <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                        block.to(self.main_device)
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:
                        hidden_states = block(
                            hidden_states=hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )
                    if index_block <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                        block.to(self.offload_device, non_blocking=self.use_non_blocking)


                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                if index_block <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.main_device)
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                if index_block <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if index_block <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                        block.to(self.main_device)
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

                if index_block <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

NODE_CLASS_MAPPINGS = {
    "FluxBlockSwap":FluxBlockSwap,
    "FluxModelLoader": FluxModelLoader,
    "PosterCraftSample":PosterCraftSample,
    "ClipTextModelEncoder":ClipTextModelEncoder,
    "T5TextModelEncoder":T5TextModelEncoder,
    "FluxPromptEmbed":FluxPromptEmbed,
    "QwenRecapAgent":QwenRecapAgent,
}
