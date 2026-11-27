"""Standard Mamba backend for the released RCDIFO call signature."""

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.modules.mamba_simple import Mamba as _Mamba
except (ImportError, OSError) as exc:
    raise ImportError(
        "RCDIFO requires the CUDA extensions installed by install_cuda_extensions.py. "
        + str(exc)
    ) from exc


def Mamba(dim, bimamba_type=None, **kwargs):
    if bimamba_type is not None:
        raise ValueError("This adapter uses bimamba_type=None.")
    return _Mamba(dim, **kwargs)
