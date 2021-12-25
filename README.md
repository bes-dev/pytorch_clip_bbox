# pytorch_clip_bbox: Implementation of the CLIP guided bbox ranking for Object Detection.

<p align="center">
  <img src="resources/preds.jpg"/>
</p>

Pytorch based library to rank predicted bounding boxes using text/image user's prompts.

Usually, object detection models trains to detect common classes of objects such as "car", "person", "cup", "bottle".
But sometimes we need to detect more complex classes such as "lady in the red dress", "bottle of whiskey", or "where is my red cup" instead of "person", "bottle", "cup" respectively.
One way to solve this problem is to train more complex detectors that can detect more complex classes,
but we propose to use text-driven object detection that allows detecting any complex classes that can be described by natural language.
This library is written to rank predicted bounding boxes using text/image descriptions of complex classes.

## Install package

```bash
pip install pytorch_clip_bbox
```

## Install the latest version

```bash
pip install --upgrade git+https://github.com/bes-dev/pytorch_clip_bbox.git
```

## Features
- The library supports multiple prompts (images or texts) as targets for filtering.
- The library automatically detects the language of the input text, and multilingual translate it via google translate.
- The library supports the original CLIP model by OpenAI and ruCLIP model by SberAI.
- Simple integration with different object detection models.

## Usage

We provide examples to integrate our library with different popular object detectors like: [YOLOv5](examples/yolov5.py), [MaskRCNN](examples/maskrcnn.py).
Please, follow to [examples](examples/) to find more examples.

### Simple example to integrate pytorch_clip_bbox with MaskRCNN model

```bash
$ pip install -r wheel cython opencv-python numpy torch torchvision pytorch_clip_bbox
```

```python
import argparse
import random
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from pytorch_clip_bbox import ClipBBOX

def get_coloured_mask(mask):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    c = colours[random.randrange(0,10)]
    r[mask == 1], g[mask == 1], b[mask == 1] = c
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask, c

def main(args):
    # build detector
    detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(args.device)
    clip_bbox = ClipBBOX(clip_type=args.clip_type).to(args.device)
    # add prompts
    if args.text_prompt is not None:
        for prompt in args.text_prompt.split(","):
            clip_bbox.add_prompt(text=prompt)
    if args.image_prompt is not None:
        image = cv2.cvtColor(cv2.imread(args.image_prompt), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = img / 255.0
        clip_bbox.add_prompt(image=image)
    image = cv2.imread(args.image)
    pred = detector([
        T.ToTensor()(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(args.device)
    ])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_threshold = [pred_score.index(x) for x in pred_score if x > args.confidence][-1]
    boxes = [[int(b) for b in box] for box in list(pred[0]['boxes'].detach().cpu().numpy())][:pred_threshold + 1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()[:pred_threshold + 1]
    ranking = clip_bbox(image, boxes, top_k=args.top_k)
    for key in ranking.keys():
        if key == "loss":
            continue
        for box in ranking[key]["ranking"]:
            mask, color = get_coloured_mask(masks[box["idx"]])
            image = cv2.addWeighted(image, 1, mask, 0.5, 0)
            x1, y1, x2, y2 = box["rect"]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 6)
            cv2.rectangle(image, (x1, y1), (x2, y1-100), color, -1)
            cv2.putText(image, ranking[key]["src"], (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), thickness=5)
    if args.output_image is None:
        cv2.imshow("image", image)
        cv2.waitKey()
    else:
        cv2.imwrite(args.output_image, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Input image.")
    parser.add_argument("--device", type=str, default="cuda:0", help="inference device.")
    parser.add_argument("--confidence", type=float, default=0.7, help="confidence threshold [MaskRCNN].")
    parser.add_argument("--text-prompt", type=str, default=None, help="Text prompt.")
    parser.add_argument("--image-prompt", type=str, default=None, help="Image prompt.")
    parser.add_argument("--clip-type", type=str, default="clip_vit_b32", help="Type of CLIP model [ruclip, clip_vit_b32, clip_vit_b16].")
    parser.add_argument("--top-k", type=int, default=1, help="top_k predictions will be returned.")
    parser.add_argument("--output-image", type=str, default=None, help="Output image name.")
    args = parser.parse_args()
    main(args)
```