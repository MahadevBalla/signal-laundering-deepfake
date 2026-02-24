from .aasist_wrapper import AASISTWrapper
from .rawnet2_wrapper import RawNet2Wrapper

MODEL_REGISTRY = {
    "aasist": AASISTWrapper,
    "aasist-l": AASISTWrapper,
    "rawnet2": RawNet2Wrapper,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
