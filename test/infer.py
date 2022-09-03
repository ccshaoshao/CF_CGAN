
import os
import numpy as np
import torch
import cv2
import hydra
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

class CFGAN():
    pad_mod = 8

    def __init__(self,config, device='cpu',model_path=r"D:\CFGAN\experiment\1\best.ckpt"):

        self.device = device
        self.model_path=model_path
        self.config=config

        self.init_model()
    def init_model(self):
        model = load_checkpoint(self.config, self.model_path, strict=False, map_location='cuda')
        model.freeze()
        model.eval()
        self.model = model



    def forward(self, image, mask):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        batch={
            'image':image,
            'mask':mask
        }
        inpainted_image = self.model(batch)

        cur_res = inpainted_image['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res

@hydra.main(config_path=r"D:\CFGAN\experiment\1", config_name='config.yaml')
def main(config: OmegaConf):

    cfgan=CFGAN(config)
    images=[os.path.join('D:/CFGAN/test/test_images',filename) for filename in os.listdir('D:/CFGAN/test/test_images')]
    masks=[os.path.join('D:/CFGAN/test/test_masks',filename) for filename in os.listdir('D:/CFGAN/test/test_masks')]
    images=[cv2.imread(image) for image in images]
    masks=[cv2.imread(mask) for mask in masks]

    batch=[]
    for i,image in enumerate(images):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results=[]
        for j,mask in enumerate(masks):
            print(i,j)
            result=cfgan.forward(image,mask[:,:,0])
            results.append(result)
        result=np.concatenate(results,axis=1)
        batch.append(result)

    all_mask = np.concatenate(masks, axis=1)
    batch.insert(0,all_mask)
    final=np.concatenate(batch,axis=0)
    cv2.imwrite(f'D:/CFGAN/test/inpainted/result.png',final)
if __name__ == '__main__':
    main()