from torchvision.datasets import ImageNet
from torch.utils.data import Subset


SUBSET_CLASSES = [
    # Animals
    "bald eagle", "great grey owl", "American alligator", "African crocodile",
    "Komodo dragon", "diamondback", "scorpion", "tarantula", "flamingo",
    "king penguin", "albatross", "sea lion", "African hunting dog", "red fox",
    "Arctic fox", "snow leopard", "cheetah", "lion", "tiger", "brown bear",
    "gorilla", "zebra", "hippopotamus", "bison", "Arabian camel", "impala",
    "gazelle", "monarch", "dragonfly", "ostrich", "vulture", "timber wolf",
    "coyote", "ram", "ibex",
    # Landscape/vegetation
    "alp", "cliff", "geyser", "lakeside", "promontory", "sandbar", "seashore",
    "valley", "volcano", "hay", "rapeseed", "daisy", "corn", "acorn", "buckeye",
    # Vehicles
    "aircraft carrier", "airliner", "airship", "ambulance", "beach wagon",
    "bicycle-built-for-two", "bobsled", "canoe", "container ship", "fire engine",
    "freight car", "garbage truck", "go-kart", "harvester", "jeep",
    "mountain bike", "motor scooter", "school bus", "speedboat", "tractor",
    # Buildings/structures
    "barn", "boathouse", "castle", "cliff dwelling", "dam", "dome",
    "drilling platform", "greenhouse", "beacon", "monastery", "mosque", "pier",
    "stone wall", "stupa", "suspension bridge", "totem pole", "triumphal arch",
    "viaduct", "water tower", "yurt",
    # Objects
    "flagpole", "fountain", "mailbox", "park bench", "parking meter",
    "picket fence", "solar dish", "sundial", "traffic light", "street sign",
]


class ImageNetSubset(Subset):
    """ImageNet filtered to a 100-class outdoor subset, with labels remapped to 0-99."""

    def __init__(self, root, split="train", transform=None):
        base = ImageNet(root=root, split=split, transform=transform)

        # Match class names to ImageNet indices
        matched = {}
        missing = []
        for name in SUBSET_CLASSES:
            if name in base.class_to_idx:
                matched[name] = base.class_to_idx[name]
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"Could not find {len(missing)} classes in ImageNet: {missing}")
        
        target_indices = set(matched.values())
        self.label_map = {old: new for new, old in enumerate(sorted(target_indices))}
        self.classes = dict(sorted({self.label_map[idx]: name for name, idx in matched.items()}.items()))
        self.num_classes = len(self.classes)
        self.targets = [self.label_map[label] for _, label in base.samples if label in target_indices]
        filtered = [i for i, (_, label) in enumerate(base.samples) if label in target_indices]
        
        super().__init__(base, filtered)

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, self.label_map[label]