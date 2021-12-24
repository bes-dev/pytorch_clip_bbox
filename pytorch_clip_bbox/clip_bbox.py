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


class ClipBBOX(nn.Module):
    """ Implementation of the CLIP guided bbox refinement for Object Detection.
    Arguments:
        clip_type (str): type of the CLIP model.
        batch_size (int): batch size.
    """
    def __init__(
            self,
            clip_type: str = "clip_vit_b32",
            batch_size: int = 128
    ):
        super().__init__()
        # CLIP guided loss
        self.clip_loss = get_clip_guided_loss(clip_type, input_range=(0.0, 1.0))
        self.input_size = self.clip_loss.image_processor[0].size
        self.batch_size = batch_size
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
            prompt_label: str = "loss"
    ):
        """ CLIP guided filter for input bounding boxes.
        Argument:
            img (np.array): input image.
            boxes (List[Tuple[int, int, int, int]]): input bounding boxes in format [(x, y, w, h)]
            is_rgb (bool): is imput image in RGB/BGR format.
            top_k (int): top k best matches will be returned.
                         Use top_k = -1 to return all boxes in ranked order.
            prompt_label (str): prompt label that uses for ranking.
        Returns:
            outputs (List[Dict]): predicts in format:
                                  [{"rect": [x, y, w, h], "loss": loss_val}]
        """
        _img = img.copy() if is_rgb else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch = self._prepare_batch(_img, boxes).to(self.device_info.device)
        loss = self._predict_clip_loss(batch, prompt_label)
        outputs = self._generate_output(boxes, loss, top_k)
        return outputs

    def _prepare_batch(self, img: np.array, proposals: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """ Crop region proposals and generate batch
        Argument:
            img (np.array): input image in opencv format (H, W, 3).
            proposals (List[Tuple[int, int, int, int]]): output proposals in format [(x, y, w, h)]
        Returns:
            batch (torch.Tensor): output batch (B, C, H, W).
        """
        batch = []
        for x, y, w, h in proposals:
            crop = cv2.resize(img[y:y+h, x:x+w], self.input_size)
            batch.append(torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0))
        batch = torch.cat(batch, dim=0)
        # normalize batch
        batch = batch / (batch.max() + 1e-8)
        return batch

    def _predict_clip_loss(
            self,
            batch_full: torch.Tensor,
            prompt_label: str = "loss"
    ) -> torch.Tensor:
        """ Predict CLIP loss for region proposals using user's prompts.
        Argument:
            batch_full (torch.Tensor): input batch (B, C, H, W).
            prompt_label (str): prompt label that uses for ranking.
        Returns:
            loss (torch.Tensor): output batch (B, ).
        """
        loss = []
        id_start = 0
        while id_start < batch_full.size(0):
            id_stop = min(id_start + self.batch_size, batch_full.size(0))
            batch = batch_full[id_start:id_stop]
            loss.append(self.clip_loss.image_loss(image=batch, reduce=None)["loss"].cpu())
            id_start = id_stop
        loss = torch.cat(loss, dim=0)
        return loss

    def _generate_output(
            self,
            boxes: List[Tuple[int, int, int, int]],
            loss: torch.Tensor,
            top_k: int = 1
    ) -> List[Dict]:
        """ Generate top_k predictions as an output of the model.
        Argument:
            boxes (List[Tuple[int, int, int, int]]): bounding boxes in format [(x, y, w, h)]
            loss (torch.Tensor): predicted CLIP loss in format (B,).
            top_k (int): top k best matches will be returned.
                         Use top_k = -1 to return all boxes in ranked order.
        Returns:
            outputs (List[Dict]): predicts in format:
                                  [{"rect": [x, y, w, h], "loss": loss_val, "idx": idx}]
        """
        output = []
        vals, ids = loss.sort()
        top_k = min(top_k, len(boxes)) if top_k > 0 else len(boxes)
        for i in range(top_k):
            output.append({
                "rect": boxes[ids[i]],
                "loss": vals[i],
                "idx": ids[i]
            })
        return output
