import os
import torch.distributed


def get_organization_name(organization_id: str) -> str:
    if "org-e" in organization_id:
        return "dcevals-kokotajlo"
    elif "org-U" in organization_id:
        return "situational-awareness"
    else:
        raise ValueError


def is_main_process():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        # Not using distributed training, so this is the main process
        return True

    # Check for PyTorch distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    # If nothing else, assume this is the main process
    return True
