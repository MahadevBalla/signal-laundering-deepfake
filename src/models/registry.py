"""Model registry that maps user-facing names to wrapper instances."""

from __future__ import annotations

from typing import Any

from .aasist_wrapper import AASISTWrapper
from .rawnet2_wrapper import RawNet2Wrapper, HuBERTRawNet2Wrapper
from .ssl_eval_wrapper import SSLEvalWrapper

_SSL_CONFIGS = {
    "wav2vec2": "configs/wav2vec2_probe.yaml",
    "hubert":   "configs/hubert_probe.yaml",
    "wavlm":    "configs/wavlm_probe.yaml",
}

_AASIST_CONFIGS = {
    "aasist":         "external/aasist/config/AASIST.conf",
    "aasist-l":       "external/aasist/config/AASIST-L.conf",
    "rawnet2":        "external/aasist/config/RawNet2_baseline.conf",
    "hubert-rawnet2": "external/aasist/config/RawNet2_baseline.conf",
}


def get_model(name: str, **kwargs: Any):
    """Instantiate and return a model wrapper for a registry key."""
    if name in ("aasist", "aasist-l"):
        kwargs.setdefault("config_path", _AASIST_CONFIGS[name])
        return AASISTWrapper(**kwargs)

    if name == "rawnet2":
        kwargs.setdefault("config_path", _AASIST_CONFIGS["rawnet2"])
        return RawNet2Wrapper(**kwargs)

    if name == "hubert-rawnet2":
        kwargs.setdefault("config_path", _AASIST_CONFIGS["hubert-rawnet2"])
        return HuBERTRawNet2Wrapper(**kwargs)

    for ssl_name, cfg_path in _SSL_CONFIGS.items():
        if name == ssl_name:
            kwargs.setdefault("config_path", cfg_path)
            kwargs.setdefault("backend_mode", "weighted")
            kwargs.setdefault("backend_type", "ffn")
            return SSLEvalWrapper(**kwargs)

        prefix = f"{ssl_name}-layer"
        if name.startswith(prefix):
            try:
                layer = int(name[len(prefix):])
            except ValueError:
                raise ValueError(f"Cannot parse layer index from '{name}'")
            kwargs.setdefault("config_path", cfg_path)
            kwargs["backend_mode"] = "single"
            kwargs["backend_type"] = "ffn"
            kwargs["layer"] = layer
            return SSLEvalWrapper(**kwargs)

        if name == f"{ssl_name}_aasist":
            kwargs.setdefault("config_path", cfg_path)
            kwargs["backend_mode"] = "weighted"
            kwargs["backend_type"] = "aasist"
            return SSLEvalWrapper(**kwargs)

        if name in (f"{ssl_name}_rawnet2", f"{ssl_name}_rawnet2_ssl"):
            kwargs.setdefault("config_path", cfg_path)
            kwargs["backend_mode"] = "weighted"
            kwargs["backend_type"] = "rawnet2"
            return SSLEvalWrapper(**kwargs)

    raise ValueError(
        f"Unknown model '{name}'.\n"
        "Waveform: aasist, aasist-l, rawnet2, hubert-rawnet2\n"
        "SSL+FFN weighted: wav2vec2, hubert, wavlm\n"
        "SSL+FFN single: wav2vec2-layer<N>, hubert-layer<N>, wavlm-layer<N>\n"
        "SSL+AASIST: wav2vec2_aasist, hubert_aasist, wavlm_aasist\n"
        "SSL+RawNet2: wav2vec2_rawnet2, hubert_rawnet2_ssl, wavlm_rawnet2"
    )
