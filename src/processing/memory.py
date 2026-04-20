"""Best-effort PyTorch / CUDA memory release between heavy pipeline steps."""

import gc


def release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
