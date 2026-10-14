from torchvision.models import convnext_tiny
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from .convnext_moe import make_last_block_moe_factory
from .convnext_moe_blocks import MoEConvNeXtWrapper

#Detectron2 standard: random short side per training image, 800 at test time, long side <= 1333
TRAIN_MIN_SIZES = tuple(range(640, 801, 32))

def _faster_rcnn(features, num_classes, min_size, max_size):
    #convnext_tiny stage outputs (features 1/3/5/7, strides 4-32) -> FPN P2-P6 -> Faster R-CNN
    backbone = BackboneWithFPN(
        features,
        return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )
    return FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size)

# Model factories, num_classes includes background
def convnext_detector_fn(num_classes=4, min_size=TRAIN_MIN_SIZES, max_size=1333):
    return _faster_rcnn(convnext_tiny(weights=None).features, num_classes, min_size, max_size)

def convnext_moe_detector_fn(num_experts=5, top_k=1, mlp_ratio=2, capacity_ratio=1.0,
                             collect_expert_stats=False, num_classes=4, bpr=True, context_aware=False,
                             min_size=TRAIN_MIN_SIZES, max_size=1333):
    block_factory = make_last_block_moe_factory(
        num_experts=num_experts,
        top_k=top_k,
        mlp_ratio=mlp_ratio,
        capacity_ratio=capacity_ratio,
        bpr=bpr,
        context_aware=context_aware,
    )
    features = convnext_tiny(weights=None, block=block_factory).features
    return MoEConvNeXtWrapper(_faster_rcnn(features, num_classes, min_size, max_size), collect_expert_stats)
