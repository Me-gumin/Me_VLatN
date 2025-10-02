import os
import torch
import folder_paths
import comfy.utils
import comfy.sd

# 注册目录
output_dir = folder_paths.get_output_directory()
vae_dir = os.path.join(output_dir, "saved_vaes")
latent_dir = os.path.join(output_dir, "saved_latents")
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(latent_dir, exist_ok=True)
folder_paths.add_model_folder_path("saved_vaes", vae_dir)
folder_paths.add_model_folder_path("saved_latents", latent_dir)

class MeLoader:
    CATEGORY = "Me/loader"
    NODE_DISPLAY_NAME = "Me Loader"
    
    @classmethod
    def INPUT_TYPES(cls):
        vae_path = folder_paths.get_filename_list("saved_vaes")
        vae_files = ["None"] + [f for f in vae_path if f.endswith((".pth", ".safetensors"))]
        latent_path = folder_paths.get_filename_list("saved_latents")
        latent_files = ["None"] + [f for f in latent_path if f.endswith(".pt")]
        return {
            "optional": {
                "vae_file": (vae_files,) if vae_files else ("STRING", {"default": "None"}),
                "latent_file": (latent_files,) if latent_files else ("STRING", {"default": "None"}),
            }
        }

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("latent", "vae")
    OUTPUT_OPTIONAL = {
        "latent": True,
        "vae": True,
    }
    FUNCTION = "MeLoader"
    
    def load_latent(self, latent_file):
        if not latent_file or latent_file == "None":
            return (None,)
        latent_dir = os.path.join(folder_paths.get_output_directory(), "saved_latents")
        latent_path = os.path.join(latent_dir, latent_file)

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")

        latent = torch.load(latent_path, map_location='cpu')
        if isinstance(latent, dict):
            if 'latent' in latent:
                latent_tensor = latent['latent']
            elif 'samples' in latent:
                latent_tensor = latent['samples']
            else:
                tensor_keys = [k for k, v in latent.items() if torch.is_tensor(v)]
                if tensor_keys:
                    latent_tensor = latent[tensor_keys[0]]
                else:
                    raise ValueError("No tensor found in latent file")
        else:
            latent_tensor = latent

        if not torch.is_tensor(latent_tensor):
            raise ValueError("Loaded data is not a tensor")

        return ({"samples": latent_tensor},)
        
    def load_vae(self, vae_file):
        if not vae_file or vae_file == "None":
            return (None,)
        vae_dir = os.path.join(folder_paths.get_output_directory(), "saved_vaes")
        vae_path = os.path.join(vae_dir, vae_file)

        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE file not found: {vae_path}")

        sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return (vae,)
    
    def MeLoader(self, vae_file="None", latent_file="None"):
        latent = self.load_latent(latent_file=latent_file)[0] if latent_file != "None" else None
        vae = self.load_vae(vae_file=vae_file)[0] if vae_file != "None" else None
        return (latent, vae)
