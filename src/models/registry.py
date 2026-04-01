from .aasist_wrapper import AASISTWrapper
from .rawnet2_wrapper import RawNet2Wrapper
from .ssl_probe_wrapper import SSLProbeWrapper  

MODEL_REGISTRY = {
    "aasist": AASISTWrapper,
    "aasist-l": AASISTWrapper,
    "rawnet2": RawNet2Wrapper,
    "wav2vec2":  SSLProbeWrapper,   # ADD
    "hubert":    SSLProbeWrapper,   # ADD
    "wavlm":     SSLProbeWrapper,   # ADD
}



def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
