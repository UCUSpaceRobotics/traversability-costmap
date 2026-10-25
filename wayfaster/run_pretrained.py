import torch
from pathlib import Path

source = Path("pretrained/wayfaster_best.ckpt")
destination = Path("pretrained/wayfaster_weights_only.pt")

# WARNING: Only use with a trusted checkpoint.
# Unrestricted pickle deserialization can execute arbitrary code.
checkpoint = torch.load(
    source,
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint keys:", list(checkpoint.keys()))

# Extract model parameters and buffers
state_dict = checkpoint["state_dict"]

assert state_dict, "Checkpoint contains no model weights"
assert all(
    isinstance(k, str) and isinstance(v, torch.Tensor)
    for k, v in state_dict.items()
), "Unexpected entries in state_dict"

print(f"Number of entries: {len(state_dict)}")

for key in list(state_dict)[:5]:
    print(key, state_dict[key].shape)

# Save only tensors, without Lightning metadata
torch.save(
    {"state_dict": state_dict},
    destination,
)

# Verify that the cleaned checkpoint loads safely
verified = torch.load(
    destination,
    map_location="cpu",
    weights_only=True,
)

assert verified["state_dict"].keys() == state_dict.keys()

print("\nClean checkpoint saved successfully!")
print(f"Path: {destination}")