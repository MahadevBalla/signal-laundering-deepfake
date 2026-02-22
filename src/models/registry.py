from .aasist_wrapper import AASISTWrapper

MODEL_REGISTRY = {
    "aasist": AASISTWrapper,
    "aasist-l": AASISTWrapper,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
