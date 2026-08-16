"""GPU/CPU device detection. Never hard-code a device (Section 31)."""
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device: str
    name: str
    total_memory_gb: float


def get_device() -> DeviceInfo:
    try:
        import torch
    except ImportError:
        return DeviceInfo(device="cpu", name="cpu (torch not installed)", total_memory_gb=0.0)

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            device="cuda",
            name=props.name,
            total_memory_gb=round(props.total_memory / (1024 ** 3), 2),
        )
    return DeviceInfo(device="cpu", name="cpu", total_memory_gb=0.0)


if __name__ == "__main__":
    info = get_device()
    print(f"device={info.device} name={info.name} memory_gb={info.total_memory_gb}")
