# from .vae_latent_rip import (
#     CheckpointVAEExtractor,
#     MeVAELoader,
#     KSampleLatentProcessor,
#     MeLatentLoader,
# )


# # 将所有节点注册到ComfyUI
# NODE_CLASS_MAPPINGS = {
#     "CheckpointVAEExtractor": CheckpointVAEExtractor,
#     "MeVAELoader": MeVAELoader,
#     "KSampleLatentProcessor": KSampleLatentProcessor,
#     "MeLatentLoader": MeLatentLoader,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "CheckpointVAEExtractor": "Checkpoint VAE Extractor",
#     "MeVAELoader": "Me VAE Loader", 
#     "KSampleLatentProcessor": "KSample Latent Processor",
#     "MeLatentLoader": "Me Latent Loader",
# }

# __all__=['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS']

from .c112 import (
    VaeLatentDecoderNode,
    MeVAELoader,
    MeLatentLoader,
)
# 将所有节点注册到ComfyUI
# 节点注册
NODE_CLASS_MAPPINGS = {
    "VaeLatentDecoderNode": VaeLatentDecoderNode,
    "MeVAELoader": MeVAELoader,
    "MeLatentLoader": MeLatentLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VaeLatentDecoderNode": "VAE Latent Saver & Decoder",
    "MeVAELoader": "Me VAE Loader",
    "MeLatentLoader": "Me Latent Loader",
}

__all__=['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS']