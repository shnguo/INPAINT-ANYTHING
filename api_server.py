from kserve import Model, ModelServer,model_server, InferRequest, InferOutput, InferResponse
from typing import Dict, Union
import numpy as np
import argparse
from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
import torch
import cv2
from utils import show_anns,img_2_base64,dilate_mask,save_array_to_img,show_points,show_mask
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import numpy
from omegaconf import OmegaConf
import os
import yaml
import sys
from pathlib import Path
from FastSAM.fastsam import FastSAM, FastSAMPrompt


sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

class Inpaiting_Anything(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.dispatch = {
            "sam_all": self.sam_all_solver,
            "sam_point":self.sam_point_solver,
            'remove_anything':self.remove_anything_solver,
            'fastsam_all':self.fastsam_all_solver,
        }
       self.load_sam()
       self.load_lama()
       self.load_fastsam()
       self.ready = True

    def load_sam(self):
        sam = sam_model_registry['vit_h'](checkpoint='./pretrained_models/sam_vit_h_4b8939.pth')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(torch.cuda.is_available())
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.sam_predictor = SamPredictor(sam)

    def load_fastsam(self):
        self.fastsam_model = FastSAM('FastSAM/weights/FastSAM.pt') 

    def load_lama(self):
        config_p = 'lama/configs/prediction/default.yaml'
        ckpt_p = 'pretrained_models/big-lama'
        predict_config = OmegaConf.load(config_p)
        predict_config.model.path = ckpt_p
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = torch.device(predict_config.device)
        self.device = torch.device(device)

        train_config_path = os.path.join(
            predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        self.model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()
        if not predict_config.get('refine', False):
            self.model.to(self.device)
        self.predict_config = predict_config

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
            return self.dispatch[payload['mode']](payload,headers)
    
    def ori_img_input(self,payload):
        image = None
        if 'img_path' in payload["instances"][0]['image']:
            image = cv2.imread( payload["instances"][0]['image']['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif 'img_base64' in payload["instances"][0]['image']:
            img_data = payload["instances"][0]["image"]["img_base64"]
            raw_img_data = base64.b64decode(img_data)
            pil_image = Image.open(io.BytesIO(raw_img_data)).convert('RGB')
            pil_image.save('results/tmp.png')
            image = numpy.array(pil_image)
            # Convert RGB to BGR
            # image = image[:, :, ::-1].copy()
        return image
    
    def mask_input(self,payload):
        mask = None
        if 'mask_path' in payload["instances"][0]['mask']:
            mask = cv2.imread( payload["instances"][0]['mask']['mask_path'])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            mask =  mask[:, :, -1].copy()
        elif 'mask_base64' in payload["instances"][0]['mask']:
            mask_data = payload["instances"][0]["mask"]["mask_base64"]
            raw_mask_data = base64.b64decode(mask_data)
            pil_mask = Image.open(io.BytesIO(raw_mask_data)).convert('HSV')
            mask = numpy.array(pil_mask)
            # Convert RGB to BGR
            mask = mask[:, :, -1].copy()
        return mask

    def sam_all_solver(self,payload,headers):
        if not hasattr(self,'mask_generator'):
            self.load_sam()

        image = self.ori_img_input(payload)
        masks = self.mask_generator.generate(image)
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        plt.axis('off')
        show_anns(plt.gca(), masks)
        final_mask = np.zeros((masks[0]['crop_box'][-1],masks[0]['crop_box'][-2]))
        for i , mask in enumerate(masks):
            final_mask  = final_mask+mask['segmentation']*(i+1)
        my_base64_pngData = self.get_plt_base64()
        plt.close()
        print(final_mask)
        return {'final_mask':final_mask.tolist(),'img_base64':my_base64_pngData}
    
    def fastsam_all_solver(self,payload,headers):
        image = self.ori_img_input(payload)
        everything_results = self.fastsam_model(image, device='cuda',retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(image, everything_results,device='cuda')
        ann = prompt_process.everything_prompt()
        final_mask = np.zeros((image.shape[0],image.shape[1]))
        ann_array = ann.detach().cpu().numpy()
        for i , mask in enumerate(ann_array):
            final_mask  = final_mask+mask*(i+1)
        return {'final_mask':final_mask.tolist()}



    def get_plt_base64(self):
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='png')
        my_stringIObytes.seek(0)
        my_base64_pngData = base64.b64encode(my_stringIObytes.read())
        return my_base64_pngData

    def sam_point_solver(self,payload,headers):
        if not hasattr(self,'sam_predictor'):
            self.load_sam()
        image = self.ori_img_input(payload)
        point_coords =  np.array(payload["instances"][0]['point_coords'])
        point_labels = np.array(payload["instances"][0]['point_labels'])
        self.sam_predictor.set_image(image)
        masks, scores, logits = self.sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )   
        masks = masks.astype(np.uint8) * 255
        masks = [dilate_mask(mask, 15) for mask in masks]
        result = {'mask_base64':[],'img_base64':[]}
        px = 1/plt.rcParams['figure.dpi']
        for idx, mask in enumerate(masks):
            mask_base64 = img_2_base64(Image.fromarray(mask.astype(np.uint8)))
            result['mask_base64'].append(mask_base64)    
            dpi = plt.rcParams['figure.dpi']
            height, width = image.shape[:2]
            plt.figure(figsize=(width*px, height*px))
            plt.imshow(image)
            plt.axis('off')
            show_points(plt.gca(), point_coords, point_labels,
                    size=(width*0.04)**2/2)
            show_mask(plt.gca(), mask, random_color=False)
            img_base64 = self.get_plt_base64()
            result['img_base64'].append(img_base64)
            plt.close()

        return result
    
    @torch.no_grad()
    def remove_anything_solver(self,payload,headers):
        # mask = self.sam_point_solver(payload,headers)
        mask = self.mask_input(payload)
        mod=8
        img = self.ori_img_input(payload)
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()
        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = self.model(batch)
        cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        result_img = Image.fromarray(cur_res.astype(np.uint8))
        result_img.save('results/result_img.png')
        img_base64 = img_2_base64(result_img)
        return {'img_base64':img_base64}




         
    

parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name", help="The name that the model is served under.", default="custom-model"
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = Inpaiting_Anything("aigc_inpaiting_anything")
    ModelServer().start([model]
                        )
