from .LoadTempCheckpoint import LoadTempCheckpoint
from .LoadTempLoRA import LoadTempLoRA
from .LoadTempMultiLoRA import LoadTempMultiLoRA

NODE_CLASS_MAPPINGS = {
    "LoadTempCheckpoint": LoadTempCheckpoint,
    "LoadTempLoRA": LoadTempLoRA,
    "LoadTempMultiLoRA": LoadTempMultiLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTempCheckpoint": "Checkpoint Loader (Temporary)",
    "LoadTempLoRA": "Load LoRA (Temporary)",
    "LoadTempMultiLoRA": "Load Multi LoRA (Temporary)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
