from .LoadTempCheckpoint import LoadTempCheckpoint
from .LoadTempLoRA import LoadTempLoRA

NODE_CLASS_MAPPINGS = {
    "LoadTempCheckpoint": LoadTempCheckpoint,
    "LoadTempLoRA": LoadTempLoRA
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTempCheckpoint": "Load Checkpoint (Temporary)",
    "LoadTempLoRA": "Load LoRA (Temporary)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
