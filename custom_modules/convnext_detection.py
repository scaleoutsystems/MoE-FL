import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from .convnext_moe import make_last_block_moe_factory
from .convnext_moe_blocks import MoEConvNeXtWrapper, MoECNBlock

#Detectron2 standard: random short side per training image, 800 at test time, long side <= 1333
TRAIN_MIN_SIZES = tuple(range(640, 801, 32))

#Under bf16 autocast, anchors and decoded boxes inherit bf16 (4-8 px spacing at ~1000 px), so run the heads in fp32
def _float(features):
    return {k: v.float() for k, v in features.items()}

class FP32RPN(RegionProposalNetwork):
    def forward(self, images, features, targets=None):
        with torch.autocast(device_type=next(iter(features.values())).device.type, enabled=False):
            return super().forward(images, _float(features), targets)

class FP32RoIHeads(RoIHeads):
    def forward(self, features, proposals, image_shapes, targets=None):
        with torch.autocast(device_type=next(iter(features.values())).device.type, enabled=False):
            return super().forward(_float(features), [p.float() for p in proposals], image_shapes, targets)

def _faster_rcnn(features, num_classes, min_size, max_size):
    #convnext_tiny stage outputs (features 1/3/5/7, strides 4-32) -> FPN P2-P6 -> Faster R-CNN
    backbone = BackboneWithFPN(
        features,
        return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )
    #1000 proposals per image after NMS in training, as in Detectron2's FPN config (torchvision's default is 2000)
    detector = FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size,
                          rpn_post_nms_top_n_train=1000)
    #Class swap keeps parameter names and deepcopy intact
    detector.rpn.__class__ = FP32RPN
    detector.roi_heads.__class__ = FP32RoIHeads
    return detector

def _load_imagenet_features(features):
    #ImageNet-1k ConvNeXt-Tiny weights (backbone only, FPN and heads stay random). In MoE blocks the dwconv,
    #norm and layer scale are copied as is, and every expert starts from the MLP of the block it replaces
    #(needs mlp_ratio=4). The routers stay at their init.
    pretrained = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.state_dict()
    moe_blocks = {name: m for name, m in features.named_modules() if isinstance(m, MoECNBlock)}
    state = {}
    for key, value in pretrained.items():
        stage, idx, rest = key.split(".", 2)
        block = f"{stage}.{idx}"
        if block not in moe_blocks or rest == "layer_scale":
            state[key] = value
        elif rest.startswith(("block.0.", "block.2.")):  # depthwise conv, LayerNorm
            state[f"{block}.dwconv.{rest[6:]}"] = value
        else:  # block.3 / block.5: the MLP's two Linear layers
            linear, param = rest[6:].split(".")
            for e in range(moe_blocks[block].num_experts):
                state[f"{block}.experts.{e}.block.{0 if linear == '3' else 2}.{param}"] = value
    missing, unexpected = features.load_state_dict(state, strict=False)
    assert not unexpected and all(".router." in k for k in missing), (missing, unexpected)
    return features

# Model factories, num_classes includes background
def convnext_detector_fn(num_classes=4, min_size=TRAIN_MIN_SIZES, max_size=1333, pretrained=False):
    features = convnext_tiny(weights=None).features
    if pretrained:
        _load_imagenet_features(features)
    return _faster_rcnn(features, num_classes, min_size, max_size)

def convnext_moe_detector_fn(num_experts=5, top_k=1, mlp_ratio=4, capacity_ratio=1.0,
                             collect_expert_stats=False, num_classes=4, bpr=True, context_aware=False,
                             min_size=TRAIN_MIN_SIZES, max_size=1333, pretrained=False):
    block_factory = make_last_block_moe_factory(
        num_experts=num_experts,
        top_k=top_k,
        mlp_ratio=mlp_ratio,
        capacity_ratio=capacity_ratio,
        bpr=bpr,
        context_aware=context_aware,
    )
    features = convnext_tiny(weights=None, block=block_factory).features
    if pretrained:
        _load_imagenet_features(features)
    return MoEConvNeXtWrapper(_faster_rcnn(features, num_classes, min_size, max_size), collect_expert_stats)
