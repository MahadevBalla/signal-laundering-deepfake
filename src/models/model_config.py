"""Central model configuration registry.

This module keeps model-to-config and model-to-weight mappings in one place.
It is shared by run scripts and evaluation scripts to avoid copy-pasted maps.
"""

from __future__ import annotations

from typing import Optional

_SSL_YAML = {
    "wav2vec2": "configs/wav2vec2_probe.yaml",
    "hubert":   "configs/hubert_probe.yaml",
    "wavlm":    "configs/wavlm_probe.yaml",
}

CONFIGS: dict[str, str] = {
    "aasist":             "external/aasist/config/AASIST.conf",
    "aasist-l":           "external/aasist/config/AASIST-L.conf",
    "rawnet2":            "external/aasist/config/RawNet2_baseline.conf",
    "hubert-rawnet2":     "external/aasist/config/RawNet2_baseline.conf",
    "wav2vec2":           _SSL_YAML["wav2vec2"],
    "hubert":             _SSL_YAML["hubert"],
    "wavlm":              _SSL_YAML["wavlm"],
    "wav2vec2_aasist":    _SSL_YAML["wav2vec2"],
    "hubert_aasist":      _SSL_YAML["hubert"],
    "wavlm_aasist":       _SSL_YAML["wavlm"],
    "wav2vec2_rawnet2":   _SSL_YAML["wav2vec2"],
    "hubert_rawnet2_ssl": _SSL_YAML["hubert"],
    "wavlm_rawnet2":      _SSL_YAML["wavlm"],
}

WEIGHTS: dict[str, Optional[str]] = {
    "aasist":             "external/aasist/models/weights/AASIST.pth",
    "aasist-l":           "external/aasist/models/weights/AASIST-L.pth",
    "rawnet2":            "external/aasist/models/weights/RawNet2.pth",
    "hubert-rawnet2":     None,
    "wav2vec2":           "models/wav2vec2_ffn_weighted.pth",
    "hubert":             "models/hubert_ffn_weighted.pth",
    "wavlm":              "models/wavlm_ffn_weighted.pth",
    "wav2vec2_aasist":    "models/wav2vec2_aasist_weighted.pth",
    "hubert_aasist":      "models/hubert_aasist_weighted.pth",
    "wavlm_aasist":       "models/wavlm_aasist_weighted.pth",
    "wav2vec2_rawnet2":   "models/wav2vec2_rawnet2_weighted.pth",
    "hubert_rawnet2_ssl": "models/hubert_rawnet2_weighted.pth",
    "wavlm_rawnet2":      "models/wavlm_rawnet2_weighted.pth",
}

SSL_MODELS = frozenset({
    "wav2vec2", "hubert", "wavlm",
    "wav2vec2_aasist",    "hubert_aasist",     "wavlm_aasist",
    "wav2vec2_rawnet2",   "hubert_rawnet2_ssl", "wavlm_rawnet2",
})
