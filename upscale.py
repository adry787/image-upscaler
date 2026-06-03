#!/usr/bin/env python3
import torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as T
class SRNet(nn.Module):
    def __init__(self,s=4):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,3*s*s,3,padding=1),nn.PixelShuffle(s))
    def forward(self,x): return self.net(x)
class Upscaler:
    def __init__(self,s=4):
        self.dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=SRNet(s).to(self.dev).eval()
    def upscale(self,inp,out):
        img=Image.open(inp).convert("RGB"); x=T.ToTensor()(img).unsqueeze(0).to(self.dev)
        with torch.no_grad(): result=T.ToPILImage()(self.model(x).squeeze().clamp(0,1))
        result.save(out); print(f"Saved {out}")
if __name__=="__main__":
    import sys; Upscaler().upscale(sys.argv[1],"output_4x.png")
