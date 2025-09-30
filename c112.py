import os
import torch
import numpy as np
import folder_paths
from PIL import Image
import json
from comfy.comfy_types import IO
import comfy.utils
import comfy.sd
import safetensors.torch
from datetime import datetime

# 预先注册自定义文件夹路径
output_dir = folder_paths.get_output_directory()
vae_dir = os.path.join(output_dir, "saved_vaes")
latent_dir = os.path.join(output_dir, "saved_latents")

# 确保目录存在
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(latent_dir, exist_ok=True)

# 注册到 folder_paths
folder_paths.add_model_folder_path("saved_vaes", vae_dir)
folder_paths.add_model_folder_path("saved_latents", latent_dir)

class VaeLatentDecoderNode:
    CATEGORY = "Me/VLatNode"
    NODE_DISPLAY_NAME = "VAE Latent Saver"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "vae": (IO.VAE,),# Input type for VAE models in ComfyUI
                "latent": ("LATENT",),
                "vae_name": ("STRING", {"default": "vae_model","tooltip": "The customer vae name"}),
            },
            "required": {
                "save_vae": ("BOOLEAN", {"default": False}),
                "save_latent": ("BOOLEAN", {"default": False}),
                "decode_image": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("status", "image")
    OUTPUT_OPTIONAL = {
        "image": True
    }
    FUNCTION = "process"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    def save_vae(self, vae_model, model_name, save_directory):
        """正确保存VAE模型数据"""
        try:
            # 创建保存目录
            os.makedirs(save_directory, exist_ok=True)
            vae_path = os.path.join(save_directory, f"{model_name}.safetensors")
            safetensors.torch.save_file(vae_model.get_sd(), vae_path)
            # 保存为 .safetensors 文件
            return f"VAE saved: {os.path.basename(vae_path)}"

        except Exception as e:
            return f"Failed to save VAE: {str(e)}"

    def save_latent(self, latent, save_path):
        """保存latent数据"""
        try:
            # 创建保存目录
            os.makedirs(save_path, exist_ok=True)

            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latent_path = os.path.join(save_path, f"latent_{timestamp}.pt")

            # 提取latent数据并确保在CPU上
            if isinstance(latent, dict) and "samples" in latent:
                latent_tensor = latent["samples"].detach().cpu()
                latent_shape = latent_tensor.shape
                latent_batch = latent_tensor.shape[0]
            else:
                latent_tensor = latent.detach().cpu() if hasattr(latent, 'detach') else latent
                latent_shape = latent_tensor.shape if hasattr(latent_tensor, 'shape') else "unknown"
                latent_batch = latent_shape[0] if isinstance(latent_shape, torch.Size) and len(latent_shape) > 0 else "unknown"

            # 保存latent数据
            torch.save({
                'latent': latent_tensor,
                'samples': latent_tensor,  # 兼容性字段
                'saved_at': datetime.now().isoformat(),
                'shape': latent_shape,
                'batch_size': latent_batch,
                'type': 'latent',
                'dtype': str(latent_tensor.dtype)
            }, latent_path)

            return f"Latent saved: {os.path.basename(latent_path)} (shape: {list(latent_shape)})"
        except Exception as e:
            return f"Failed to save latent: {str(e)}"

    def decode_latent_to_image(self, vae, latent):
        """使用VAE解码latent为图像"""
        try:
            # 提取latent tensor
            if isinstance(latent, dict) and "samples" in latent:
                latent_tensor = latent["samples"]
            else:
                latent_tensor = latent

            # 确保latent在正确的设备上
            if hasattr(vae, 'parameters'):
                device = next(vae.parameters()).device
                latent_tensor = latent_tensor.to(device)

            # 使用VAE解码
            with torch.no_grad():
                if hasattr(vae, 'decode'):
                    decoded_image = vae.decode(latent_tensor)
                elif hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'decode'):
                    decoded_image = vae.first_stage_model.decode(latent_tensor)
                else:
                    return None, "VAE does not have decode method"

                # 安全的形状处理
                if decoded_image.dim() == 4:
                    # 正确的形状 [batch, channels, height, width]
                    pass
                elif decoded_image.dim() == 5:
                    # 合并批次维度 [batch, sub_batch, channels, height, width]
                    batch_size, sub_batch = decoded_image.shape[:2]
                    decoded_image = decoded_image.reshape(
                        batch_size * sub_batch, *decoded_image.shape[2:]
                    )
                else:
                    return None, f"Unexpected decoded image shape: {decoded_image.shape}"

                # 确保图像值在合理范围内
                decoded_image = torch.clamp(decoded_image, -1.0, 1.0)

            return decoded_image, "Image decoded successfully"

        except Exception as e:
            return None, f"Failed to decode image: {str(e)}"

    def process(self, vae_name="vae_model", vae=None, latent=None, save_vae=False, save_latent=False, decode_image=False):
        """主处理函数"""
        status_messages = []
        result_image = None

        # 输入验证
        if decode_image and (vae is None or latent is None):
            return ("Cannot decode: Both VAE and latent inputs are required", None)

        # 保存VAE模型
        if save_vae:
            if vae is not None:
                vae_save_path = os.path.join(self.output_dir, "saved_vaes")
                vae_status = self.save_vae(vae, vae_name, vae_save_path)
                status_messages.append(vae_status)
            else:
                status_messages.append("Skip VAE save: No VAE input")
                print("    - VAE save: Skipped (no VAE input)")

        # 保存latent数据
        if save_latent:
            if latent is not None:
                latent_save_path = os.path.join(self.output_dir, "saved_latents")
                latent_status = self.save_latent(latent, latent_save_path)
                status_messages.append(latent_status)
            else:
                status_messages.append("Skip latent save: No latent input")
                print("   - Latent save: Skipped (no latent input)")

        # 解码图像
        if decode_image:
            if vae is not None and latent is not None:
                decoded_image, decode_status = self.decode_latent_to_image(vae, latent)
                status_messages.append(decode_status)
                if decoded_image is not None:
                    result_image = decoded_image
            else:
                missing = []
                if vae is None: missing.append("VAE")
                if latent is None: missing.append("latent")
                status_msg = f" Skip decode: Missing {', '.join(missing)}"
                status_messages.append(status_msg)

        final_status = " | ".join(status_messages)
        print(f"VAE Latent Decoder Status: {final_status}")

        return (final_status, result_image)

class MeVAELoader:
    CATEGORY = "Me/load vae"
    NODE_DISPLAY_NAME = "VAE Loader (Custom)"
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取文件列表（只允许 .pth / .safetensors）
        vae_files = folder_paths.get_filename_list("saved_vaes")
        vae_files = [f for f in vae_files if f.endswith((".pth", ".safetensors"))]
        
        # 如果没有文件，提供字符串输入
        if not vae_files:
            return {
                "required": {
                    "vae_file": ("STRING", {"default": "","tooltip":"simple vae loader"}),
                }
            }
        
        return {
            "required": {
                "vae_file": (vae_files,),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    
    def load_vae(self, vae_file):
        if not vae_file:
            raise ValueError("VAE file name cannot be empty")
            
        output_dir = folder_paths.get_output_directory()
        vae_dir = os.path.join(output_dir, "saved_vaes")
        vae_path = os.path.join(vae_dir, vae_file)

        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE file not found: {vae_path}")

        try:
            # 加载VAE权重
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
            return (vae,)
            
        except Exception as e:
            raise Exception(f" Failed to load VAE from {vae_file}: {str(e)}")

class MeLatentLoader:
    CATEGORY = "Me/load latent"
    NODE_DISPLAY_NAME = "Latent Loader (Custom)"
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取文件列表（只允许 .pt 文件）
        latent_files = folder_paths.get_filename_list("saved_latents")
        latent_files = [f for f in latent_files if f.endswith(".pt")]
        
        # 如果没有文件，提供字符串输入
        if not latent_files:
            return {
                "required": {
                    "latent_file": ("STRING", {"default": ""}),
                }
            }
        
        return {
            "required": {
                "latent_file": (latent_files,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load_latent"
    
    def load_latent(self, latent_file):
        if not latent_file:
            raise ValueError("Latent file name cannot be empty")
            
        output_dir = folder_paths.get_output_directory()
        latent_dir = os.path.join(output_dir, "saved_latents")
        latent_path = os.path.join(latent_dir, latent_file)

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")

        try:
            # 加载latent数据
            latent = torch.load(latent_path, map_location='cpu')
            
            # 确保返回正确的 LATENT 格式
            if isinstance(latent, dict):
                if 'latent' in latent:
                    latent_tensor = latent['latent']
                elif 'samples' in latent:
                    latent_tensor = latent['samples']
                else:
                    # 如果字典中没有明确的键，尝试找到第一个张量
                    tensor_keys = [k for k, v in latent.items() if torch.is_tensor(v)]
                    if tensor_keys:
                        latent_tensor = latent[tensor_keys[0]]
                    else:
                        raise ValueError("No tensor found in latent file")
            else:
                latent_tensor = latent

            # 确保是张量
            if not torch.is_tensor(latent_tensor):
                raise ValueError("Loaded data is not a tensor")
                
            # 返回 ComfyUI 标准的 LATENT 格式
            result = {"samples": latent_tensor}
            
            print(f" Latent successfully loaded from: {latent_file} (shape: {list(latent_tensor.shape)})")
            return (result,)
            
        except Exception as e:
            raise Exception(f" Failed to load latent from {latent_file}: {str(e)}")
