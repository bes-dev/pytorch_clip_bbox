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
import argparse
import cv2
import torch
from pytorch_clip_bbox import ClipBBOX

def extract_boxes(detections):
    boxes = []
    for i in range(detections.xyxy[0].size(0)):
        x1, y1, x2, y2, confidence, idx = detections.xyxy[0][i]
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

def main(args):
    # build detector
    detector = torch.hub.load("ultralytics/yolov5", "yolov5s").to(args.device)
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
    detections = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes = extract_boxes(detections)
    ranking = clip_bbox(image, boxes, top_k=args.top_k)
    for key in ranking.keys():
        if key == "loss":
            continue
        for box in ranking[key]["ranking"]:
            x1, y1, x2, y2 = box["rect"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6)
            cv2.rectangle(image, (x1, y1), (x2, y1-100), (0, 255, 0), -1)
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
    parser.add_argument("--text-prompt", type=str, default=None, help="Text prompt.")
    parser.add_argument("--image-prompt", type=str, default=None, help="Image prompt.")
    parser.add_argument("--clip-type", type=str, default="clip_vit_b32", help="Type of CLIP model [ruclip, clip_vit_b32, clip_vit_b16].")
    parser.add_argument("--top-k", type=int, default=1, help="top_k predictions will be returned.")
    parser.add_argument("--output-image", type=str, default=None, help="Output image name.")
    args = parser.parse_args()
    main(args)
