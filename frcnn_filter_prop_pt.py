#!/opt/local/bin/python3
#!/usr/bin/python3

import os
## try to stay off the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import torch
import torchvision
import numpy as np
import pickle

from torch import Tensor
from typing import List, Tuple

with open('frcnn_filter_prop.pickle', 'rb') as handle:
    (proposals, objectness, orig_boxes, orig_scores) = pickle.load(handle)

proposals = torch.from_numpy(proposals)
objectness = torch.from_numpy(objectness)
InputImgSize = 1024
num_anchors_per_level = [15360]

################################################################################
################################################################################

def filter_proposals(
        #self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:


        ## from class external
        min_size = 0.001
        score_thresh = 0.0
        nms_thresh = 0.7
        post_nms_top_n = 1000

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        #print("DEBUG: filter_proposals : levels.shape, objectness.shape :",
        #     levels.shape, objectness.shape)

        # select top_n boxes independently per level before applying nms
        top_n_idx = _get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        #print(proposals.shape, batch_idx, top_n_idx.shape)
        #print("DEBUG: filter_proposals : batch_idx, top_n_idx.shape :", batch_idx, top_n_idx.shape)
        #print("DEBUG: filter_proposals : num_anchors_per_level:", num_anchors_per_level)

        #print("DEBUG: filter_proposals : objectness pre :", objectness.shape)

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        #print("DEBUG: filter_proposals : objectness post :", objectness.shape)

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = remove_small_boxes(boxes, min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, lvl, nms_thresh)

            # keep only topk scoring predictions
            #keep = keep[: self.post_nms_top_n()]
            keep = keep[: post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores

################################################################################
################################################################################

def _get_top_n_idx(#self, 
    objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
    
        ## from class external
        pre_nms_top_n = 1000
        
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            ## Note: this returns current train/eval top_n value.
            #pre_nms_top_n = _topk_min(ob, self.pre_nms_top_n(), 1)
            pre_nms_top_n = _topk_min(ob, pre_nms_top_n, 1)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

################################################################################
################################################################################

def _topk_min(input: Tensor, orig_kval: int, axis: int) -> int:
    if not torch.jit.is_tracing():
        return min(orig_kval, input.size(axis))
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    return _fake_cast_onnx(min_kval)

################################################################################
################################################################################

## from opts/boxes.py
def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size `size`.
    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image
    Returns:
        Tensor[N, 4]: clipped boxes
    """
    #if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #    _log_api_usage_once(clip_boxes_to_image)
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)

################################################################################
################################################################################

def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove boxes which contains at least one side smaller than min_size.
    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        min_size (float): minimum size
    Returns:
        Tensor[K]: indices of the boxes that have both sides
        larger than min_size
    """
    #if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #    _log_api_usage_once(remove_small_boxes)
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep

################################################################################
################################################################################

def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
  
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 20000) and not torchvision._is_tracing():
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)

################################################################################
################################################################################

def _batched_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    #keep = nms(boxes_for_nms, scores, iou_threshold)
    #return keep
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

################################################################################
################################################################################

print("DEBUG: proposals :",
              #type(proposals), 
              proposals.shape,
              proposals.dtype,
              np.sqrt(np.mean(proposals.numpy()**2))
             )

print("DEBUG: objectness :",
              #type(objectness), 
              objectness.shape,
              objectness.dtype,
              np.sqrt(np.mean(objectness.numpy()**2))
             )

print("DEBUG: img_size :", InputImgSize)

print("DEBUG: num_anch.. :",
              type(num_anchors_per_level),
              len(num_anchors_per_level),
              type(num_anchors_per_level[0]),
              num_anchors_per_level[0]
             )

boxes, scores = filter_proposals(proposals,
                                 objectness,
                                 [(InputImgSize, InputImgSize)],
                                 num_anchors_per_level)

print("DEBUG: boxes :", 
              type(boxes),
              len(boxes),
              #type(boxes[0]),
              boxes[0].shape,
              boxes[0].dtype,
              np.sqrt(np.mean(boxes[0].numpy()**2))
             )

print("DEBUG: scores :",
              type(scores),
              len(scores),
              #type(scores[0]),
              scores[0].shape,
              scores[0].dtype,
              np.sqrt(np.mean(scores[0].numpy()**2)))

################################################################################
################################################################################
