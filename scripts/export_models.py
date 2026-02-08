import torch
import torchreid

model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    pretrained=True,
)
model.eval()

dummy = torch.randn(1, 3, 256, 128)
traced = torch.jit.trace(model, dummy)
traced.save("models/reid.ts")
print("Saved models/reid.ts")