"""Download + verify a MiDaS monocular depth model via torch.hub (GitHub)."""
import torch

torch.manual_seed(0)
for name in ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]:
    try:
        model = torch.hub.load(
            "isl-org/MiDaS", name, pretrained=True, trust_repo=True
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.rand(1, 3, 256, 256))
        print(f"MODEL_OK {name} params={sum(p.numel() for p in model.parameters())} "
              f"out={tuple(out.shape)}")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL_FAIL {name}: {type(exc).__name__} {str(exc)[:120]}")
