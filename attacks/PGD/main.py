from PIL import Image
img = Image.open("../../../ferrari.png").convert("RGB")
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from PIL import Image

eps   = 8/255 
alpha = 2/255 
steps = 10   

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT').to(device).train()  

image_path = "../../../"
image_name = "ferrari.png"
image_file = image_path + image_name

img = Image.open(image_file).convert("RGB")
x = F.to_tensor(img).unsqueeze(0).to(device) 

_, H, W = x.shape[1:]
targets = [{
    "boxes": torch.tensor([[0., 0., W, H]], device=device),
    "labels": torch.tensor([1],   device=device)
}]

x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)


for i in range(steps):
    if x_adv.grad is not None:
        x_adv.grad.zero_()

    losses = model(x_adv, targets)
    loss = sum(losses.values())
    
    loss.backward()
    
    with torch.no_grad():
        x_adv += alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

        x_adv = torch.clamp(x_adv, 0, 1)
    
        x_adv.requires_grad_(True)
# Save the adversarial image
# x_adv = x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
# x_adv = (x_adv * 255).astype("uint8")
# Image.fromarray(x_adv).save("adv.png")

model.eval()
with torch.no_grad():
    pred_clean = model(x)[0]
    pred_adv   = model(x_adv)[0]

print(f"Clean boxes: {len(pred_clean['boxes'])}, Adv boxes: {len(pred_adv['boxes'])}")

try:
    save_image(x_adv.detach().cpu(), f"adv_image_{image_name}")
except Exception as e:
    adv_pil = F.to_pil_image(x_adv.squeeze(0).cpu())
    adv_pil.save(f"adv_image_{image_name}")