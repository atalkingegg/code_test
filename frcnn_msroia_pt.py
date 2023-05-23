#!/usr/bin/python3

import torch
import pickle

################################################################################

def pt_initLevelMapper(
    k_min,
    k_max,
    canonical_scale=224,
    canonical_level=4,
    eps=1e-6
):
    return pt_LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class pt_LevelMapper:

    def __init__(
        self,
        k_min,
        k_max,
        canonical_scale= 224,
        canonical_level= 4,
        eps= 1e-6
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


def pt_infer_scale(feature, original_size):
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    return possible_scales[0]

def pt_setup_scales(features, image_shapes, canonical_scale, canonical_level):
    if not image_shapes:
        raise ValueError("images list should not be empty")
    max_x = 0
    max_y = 0
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
    original_input_shape = (max_x, max_y)

    scales = [pt_infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

    map_levels = pt_initLevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return scales, map_levels

def pt_filter_input(x, featmap_names):
    x_filtered = []
    for k, v in x.items():
        if k in featmap_names:
            x_filtered.append(v)
    return x_filtered

## from torchvision/ops/_utils.py
def pt_check_roi_boxes_shape(boxes):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            torch._assert(
                _tensor.size(1) == 4, "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
            )
    elif isinstance(boxes, torch.Tensor):
        torch._assert(boxes.size(1) == 5, "The boxes tensor shape is not correct as Tensor[K, 5]")
    else:
        torch._assert(False, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]")
    return

def pt_convert_to_roi_format(boxes):
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [torch.full_like(b[:, :1],
                         i,
                         dtype=dtype,
                         layout=torch.strided,
                         device=device) for i, b in enumerate(boxes)],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois

def ppt_bilinear_interpolate_v1(
    input,  # [N, C, H, W]
    roi_batch_ind,  # [K]
    y,  # [K, PH, IY]
    x,  # [K, PW, IX]
    ymask,  # [K, IY]
    xmask,  # [K, IX]
):
    _, channels, height, width = input.size()

    # deal with inverse element out of feature map boundary
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    # do bilinear interpolation, but respect the masking!
    # TODO: It's possible the masking here is unnecessary if y and
    # x were clamped appropriately; hard to tell
    def masked_index(
        y,  # [K, PH, IY]
        x,  # [K, PW, IX]
    ):
        if ymask is not None:
            assert xmask is not None
            y = torch.where(ymask[:, None, :], y, 0)
            x = torch.where(xmask[:, None, :], x, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None].long(),
            torch.arange(channels, device=input.device)[None, :, None, None, None, None].long(),
            y[:, None, :, None, :, None].long(),  # prev [K, PH, IY]
            x[:, None, None, :, None, :].long(),  # prev [K, PW, IX]
        ]  # [K, C, PH, PW, IY, IX]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    # all ws preemptively [K, C, PH, PW, IY, IX]
    def outer_prod(y, x):
        return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val

def ppt_maybe_cast(tensor):
    if torch.is_autocast_enabled() and tensor.is_cuda and tensor.dtype != torch.double:
        return tensor.float()
    else:
        return tensor

## Pure Pytorch version 1.0
# https://github.com/pytorch/vision/blob/main/torchvision/ops/roi_align.py

def ppt_roi_align(
    input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    orig_dtype = input.dtype

    input = ppt_maybe_cast(input)
    rois = ppt_maybe_cast(rois)

    _, _, height, width = input.size()

    ph = torch.arange(pooled_height, device=input.device)  # [PH]
    pw = torch.arange(pooled_width, device=input.device)  # [PW]

    # input: [N, C, H, W]
    # rois: [K, 5]

    roi_batch_ind = rois[:, 0].int()  # [K]
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
    roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
    roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
    roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

    roi_width = roi_end_w - roi_start_w  # [K]
    roi_height = roi_end_h - roi_start_h  # [K]
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)  # [K]
        roi_height = torch.clamp(roi_height, min=1.0)  # [K]

    bin_size_h = roi_height / pooled_height  # [K]
    bin_size_w = roi_width / pooled_width  # [K]

    exact_sampling = sampling_ratio > 0

    roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(roi_height / pooled_height)  # scalar or [K]
    roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(roi_width / pooled_width)  # scalar or [K]

    """
    iy, ix = dims(2)
    """

    if exact_sampling:
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)  # scalar
        iy = torch.arange(roi_bin_grid_h, device=input.device)  # [IY]
        ix = torch.arange(roi_bin_grid_w, device=input.device)  # [IX]
        ymask = None
        xmask = None
    else:
        count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)  # [K]
        # When doing adaptive sampling, the number of samples we need to do
        # is data-dependent based on how big the ROIs are.  This is a bit
        # awkward because first-class dims can't actually handle this.
        # So instead, we inefficiently suppose that we needed to sample ALL
        # the points and mask out things that turned out to be unnecessary
        iy = torch.arange(height, device=input.device)  # [IY]
        ix = torch.arange(width, device=input.device)  # [IX]
        ymask = iy[None, :] < roi_bin_grid_h[:, None]  # [K, IY]
        xmask = ix[None, :] < roi_bin_grid_w[:, None]  # [K, IX]

    def from_K(t):
        return t[:, None, None]

    y = (
        from_K(roi_start_h)
        + ph[None, :, None] * from_K(bin_size_h)
        + (iy[None, None, :] + 0.5) * from_K(bin_size_h / roi_bin_grid_h)
    )  # [K, PH, IY]
    x = (
        from_K(roi_start_w)
        + pw[None, :, None] * from_K(bin_size_w)
        + (ix[None, None, :] + 0.5) * from_K(bin_size_w / roi_bin_grid_w)
    )  # [K, PW, IX]
    ## V1
    val = ppt_bilinear_interpolate_v1(input, roi_batch_ind, y, x, ymask, xmask)  # [K, C, PH, PW, IY, IX]
    ## V2
    #n, c, ph, pw = dims(4)
    #offset_rois = rois[n]
    #roi_batch_ind = offset_rois[0].int()
    #offset_input = input[roi_batch_ind.long()][c]
    #val = ppt_bilinear_interpolate(offset_input, height, width, y, x, ymask, xmask)


        # Mask out samples that weren't actually adaptively needed
    if not exact_sampling:
        val = torch.where(ymask[:, None, None, None, :, None], val, 0)
        val = torch.where(xmask[:, None, None, None, None, :], val, 0)

    output = val.sum((-1, -2))  # remove IY, IX ~> [K, C, PH, PW]
    if isinstance(count, torch.Tensor):
        output /= count[:, None, None, None]
    else:
        output /= count

    output = output.to(orig_dtype)

    return output


def pt_roi_align(
    input,
    boxes,
    output_size,
    spatial_scale= 1.0,
    sampling_ratio= -1,
    aligned=False,
):

    pt_check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = torch.nn.modules.utils._pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = pt_convert_boxes_to_roi_format(rois)

    ## This gets exact matching results.
    if UseOpsVer:
        return torch.ops.torchvision.roi_align(
            input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
        )
    else:
    ## New python / torch only version, 98% matching
    # https://github.com/pytorch/vision/blob/main/torchvision/ops/roi_align.py
        return ppt_roi_align(
            input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
        )

def pt_multiscale_roi_align(
    x_filtered,
    boxes,
    output_size,
    sampling_ratio,
    scales,
    mapper):

    if scales is None or mapper is None:
        raise ValueError("scales and mapper should not be None")

    num_levels = len(x_filtered)
    rois = pt_convert_to_roi_format(boxes)

    if num_levels == 1:
        return pt_roi_align(
            x_filtered[0],
            rois,
            output_size=output_size,
            spatial_scale=scales[0],
            sampling_ratio=sampling_ratio,
        )

    levels = mapper(boxes)

    num_rois = len(rois)
    num_channels = x_filtered[0].shape[1]

    dtype, device = x_filtered[0].dtype, x_filtered[0].device
    result = torch.zeros(
        (
            num_rois,
            num_channels,
        )
        + output_size,
        dtype=dtype,
        device=device,
    )

    tracing_results = []
    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = torch.where(levels == level)[0]
        rois_per_level = rois[idx_in_level]

        result_idx_in_level = roi_align(
            per_level_feature,
            rois_per_level,
            output_size=output_size,
            spatial_scale=scale,
            sampling_ratio=sampling_ratio,
        )

        result[idx_in_level] = result_idx_in_level.to(result.dtype)

    return result

## From https://pytorch.org/vision/0.12/_modules/torchvision/ops/poolers.html
## in_tools/models/vision/torchvision/ops/poolers.py

class pt_MultiScaleRoIAlign(torch.nn.Module):

    def __init__(
        self,
        featmap_names,
        output_size,
        sampling_ratio,
        canonical_scale=224,
        canonical_level=4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(
        self,
        x,
        boxes,
        image_shapes,
    ):

        x_filtered = pt_filter_input(x, self.featmap_names)
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = pt_setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )

        return pt_multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )


################################################################################
################################################################################

## This brings up local pytorch version!
pt_roi_pooler = pt_MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

## load the multi-scale region of interest align layer input values captured from live model.
with open('frcnn_msroia_inputs_orig.pickle', 'rb') as handle:
    features, boxes, image_shapes, scales, map_levels, output_size, sampling_ratio, canonical_scale, canonical_level = pickle.load(handle)

UseOpsVer = True ## use roi_align from torch intrinsics

pt_roi_out_v1 = pt_roi_pooler(features, boxes, image_shapes)
print("pt_roi_out_v1:", type(pt_roi_out_v1), pt_roi_out_v1.shape,
      torch.sqrt(torch.mean(pt_roi_out_v1 ** 2)).numpy())

UseOpsVer = False ## use python code version

pt_roi_out_v2 = pt_roi_pooler(features, boxes, image_shapes)
print("pt_roi_out_v2:", type(pt_roi_out_v2), pt_roi_out_v2.shape,
      torch.sqrt(torch.mean(pt_roi_out_v2 ** 2)).numpy())


if torch.all(pt_roi_out_v1.eq(pt_roi_out_v2)):
    print("Output matches exactly!")
else:
    print("Output is NOT the same!")

A = torch.count_nonzero(pt_roi_out_v1 - pt_roi_out_v2).numpy()
B = pt_roi_out_v1.numel()
print(A, "/", B, "=", str(round((100 * (B - A) / B), 3)) + "% Matching!")
