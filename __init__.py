from .c113 import VaeLatentDecoderNode
from .c114 import MeLoader
# 将所有节点注册到ComfyUI
# 节点注册
NODE_CLASS_MAPPINGS = {
    "VaeLatentDecoderNode": VaeLatentDecoderNode,
    "MeLoader":MeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VaeLatentDecoderNode": "VAE Latent Saver & Decoder",
    "MeLoader":"Me Loader",
}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]