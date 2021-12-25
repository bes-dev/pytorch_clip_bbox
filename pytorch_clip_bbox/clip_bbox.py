"""
Copyright 2021 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_clip_guided_loss import get_clip_guided_loss
from pytorch_clip_guided_loss.clip_guided_loss import CLIPPrompt


class ClipBBOX(nn.Module):
    """ Implementation of the CLIP guided bbox refinement for Object Detection.
    Arguments:
        clip_type (str): type of the CLIP model.
    """
    def __init__(
            self,
            clip_type: str = "clip_vit_b32"
    ):
        super().__init__()
        # CLIP guided loss
        self.clip_loss = get_clip_guided_loss(clip_type, input_range=(0.0, 1.0))
        self.input_size = self.clip_loss.image_processor[0].size
        # utils
        self.register_buffer("device_info", torch.tensor(0))

    def add_prompt(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[str] = None,
            weight: float = 1.0,
            label: Optional[str] = None,
            store_src: bool = True
    ) -> str:
        """Add prompt to loss function.
        Arguments:
            image (torch.Tensor): input image [Optional].
            text (str): input text [Optional].
            weight (float): importance of the prompt.
            label (str): label of the prompt [Optional].
            store_src (bool): store source data of the prompt.
        Returns:
            label (src): label of the prompt.
        """
        return self.clip_loss.add_prompt(image, text, weight, label, store_src)

    def get_prompt(self, label: str) -> Optional[CLIPPrompt]:
        """Get prompt if available.
        Arguments:
            label (str): label of the prompt [Optional].
        Returns:
            prompt (CLIPPrompt or None): prompt [Optional].
        """
        return self.clip_loss.get_prompt(label)

    def get_prompts_list(self) -> List[str]:
        """Get list of all available prompts.
        Returns:
            prompts (list<str>): list of prompts labels.
        """
        return self.clip_loss.get_prompts_list()

    def delete_prompt(self, label: Optional[str] = None) -> None:
        """Add prompt to loss function.
        Arguments:
            label (str): label of the prompt to delete [Optional].
        """
        return self.clip_loss.delete_prompt(label)

    def clear_prompts(self) -> None:
        """Delete all available prompts."""
        return self.clip_loss.clear_prompts()

    @torch.no_grad()
    def forward(
            self,
            img: np.array,
            boxes: List[Tuple[int, int, int, int]],
            is_rgb: bool = False,
            top_k: int = 1,
            batch_size: int = 128
    ):
        """ CLIP guided filter for input bounding boxes.
        Argument:
            img (np.array): input image.
            boxes (List[Tuple[int, int, int, int]]): input bounding boxes in format [(xmin, ymin, xmax, ymax)]
            is_rgb (bool): is imput image in RGB/BGR format.
            top_k (int): top k best matches will be returned.
                         Use top_k = -1 to return all boxes in ranked order.
            batch_size (int): batch size.
        Returns:
            outputs (List[Dict]): predicts in format:
                                  [{"rect": [x, y, w, h], "loss": loss_val}]
        """
        _img = img.copy() if is_rgb else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch = self._prepare_batch(_img, boxes).to(self.device_info.device)
        loss = self._predict_clip_loss(batch, batch_size)
        outputs = self._generate_output(boxes, loss, top_k)
        return outputs

    def _prepare_batch(self, img: np.array, boxes: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """ Crop region proposals and generate batch
        Argument:
            img (np.array): input image in opencv format (H, W, 3).
            boxes (List[Tuple[int, int, int, int]]): input boxes in format [(xmin, ymin, xmax, ymax)]
        Returns:
            batch (torch.Tensor): output batch (B, C, H, W).
        """
        batch = []
        for x1, y1, x2, y2 in boxes:
            crop = cv2.resize(img[y1:y2, x1:x2], self.input_size)
            batch.append(torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0))
        batch = torch.cat(batch, dim=0)
        # normalize batch
        batch = batch / (batch.max() + 1e-8)
        return batch

    def _predict_clip_loss(
            self,
            batch_full: torch.Tensor,
            batch_size: int = 128
    ) -> torch.Tensor:
        """ Predict CLIP loss for region proposals using user's prompts.
        Argument:
            batch_full (torch.Tensor): input batch (B, C, H, W).
            prompt_label (str): prompt label that uses for ranking.
            batch_size (int): batch size.
        Returns:
            loss (Dict[str, torch.Tensor]): output batch {"prompt_label": (B, )}.
        """
        loss = {}
        id_start = 0
        while id_start < batch_full.size(0):
            id_stop = min(id_start + batch_size, batch_full.size(0))
            batch = batch_full[id_start:id_stop]
            predicted_loss = self.clip_loss.image_loss(image=batch, reduce=None)
            for key, val in predicted_loss.items():
                if not key in loss:
                    loss[key] = []
                loss[key].append(val.cpu())
            id_start = id_stop
        for key, val in loss.items():
            loss[key] = torch.cat(val, dim=0)
        return loss

    def _generate_output(
            self,
            boxes: List[Tuple[int, int, int, int]],
            loss: Dict[str, torch.Tensor],
            top_k: int = 1
    ) -> List[Dict]:
        """ Generate top_k predictions as an output of the model.
        Argument:
            boxes (List[Tuple[int, int, int, int]]): bounding boxes in format [(xmin, ymin, xmax, ymax)]
            loss (Dict[str, torch.Tensor]): predicted CLIP loss in format {"prompt_label": (B, )}.
            top_k (int): top k best matches will be returned.
                         Use top_k = -1 to return all boxes in ranked order.
        Returns:
            outputs (List[Dict]): predicts in format:
                    {
                        <prompt_label>: {
                            "src": <source_prompt>,
                            "ranking": [{"loss": loss, "idx": idx, "rect": [xmin, ymin, xmax, ymax]}]
                        }
                    }
        """
        output = {}
        top_k = min(top_k, len(boxes)) if top_k > 0 else len(boxes)
        for key, val in loss.items():
            output[key] = {}
            if key in self.get_prompts_list():
                output[key]["src"] = self.get_prompt(key).src
                if not isinstance(output[key]["src"], str):
                    output[key]["src"] = key
            ranking = []
            vals, ids = val.sort()
            for i in range(top_k):
                ranking.append({
                    "loss": vals[i],
                    "idx": ids[i],
                    "rect": boxes[ids[i]]
                })
            output[key]["ranking"] = ranking
        return output
