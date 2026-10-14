import os
import json
from collections import defaultdict
import torch
from PIL import Image
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset, Subset
import torchvision.transforms.functional as TF

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

    
class ImageNetSubset(ImageNet):
    """ImageNet filtered to a 100-class outdoor subset, with labels remapped to 0-99."""

    def __init__(self, root, split="train", transform=None):
        super().__init__(root=root, split=split, transform=transform)

        matched = {}
        missing = []
        for name in SUBSET_CLASSES:
            if name in self.class_to_idx:
                matched[name] = self.class_to_idx[name]
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"Could not find {len(missing)} classes in ImageNet: {missing}")

        target_indices = set(matched.values())
        self.label_map = {old: new for new, old in enumerate(sorted(target_indices))}
        self.classes = dict(sorted({self.label_map[idx]: name for name, idx in matched.items()}.items()))
        self.num_classes = len(self.classes)

        # Filter samples and imgs in-place
        self.samples = [(p, l) for p, l in self.samples if l in target_indices]
        self.imgs = [(p, self.label_map[l]) for p, l in self.samples]
        self.targets = [self.label_map[l] for _, l in self.samples]

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, self.label_map[label]
    
    
class WeatherImageNetSubset(ImageNetSubset):
    def __init__(self, root, split="train", transform=None,return_weather=False):
        super().__init__(root=root, split=split, transform=transform)
        #The available weather augmentations
        self.weather_types = ["sunny", "fog", "rain", "snow"]
        #The weather labels for each data point
        self.weathers = [self._parse_weather(p) for p, _ in self.imgs]
        self.return_weather = return_weather

    def _parse_weather(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.rsplit("_", maxsplit=1)
        return parts[1]

    def get_weather_subset(self, weather: str) -> Subset:
        indices = [i for i, w in enumerate(self.weathers) if w == weather]
        return Subset(self, indices)

    def get_all_weather_subsets(self): #-> dict[str, Subset]:
        return {w: self.get_weather_subset(w) for w in self.weather_types}
    
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        if not self.return_weather:
            return img, label
        weather = self.weather_types.index(self.weathers[idx])
        return img, label, weather

class ZOD(Dataset):
    """ZOD Frames 2D detection from the devkit COCO export, with the road type as context.

    Returns (image, target, context): image is a float tensor at short side `short_side`
    (long side capped at `max_size`), target holds boxes (xyxy, in that image's pixels),
    labels and the COCO image id. Crowd ("Unclear") boxes are left out, they are only
    ignore regions for evaluation against the original COCO JSON.
    """

    #Context index = position in this list = expert id 
    road_types = ["city", "arterial-urban", "highway", "arterial-rural", "smaller-rural"]
    #Detection label = position + 1, 0 is background
    classes = ["Vehicle", "VulnerableVehicle", "Pedestrian"]

    def __init__(self, coco_json, dataset_root, train=False, short_side=800, max_size=1333,
                 version="full"):
        self.coco_json = coco_json  # ground truth for evaluate_zod
        with open(coco_json) as f:
            coco = json.load(f)
        cat_to_label = {c["id"]: self.classes.index(c["name"]) + 1 for c in coco["categories"]}

        self.images = coco["images"]
        boxes = defaultdict(list)
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            if not a["iscrowd"] and w > 0 and h > 0:  # degenerate boxes crash Faster R-CNN
                boxes[a["image_id"]].append((x, y, x + w, y + h, cat_to_label[a["category_id"]]))
        self.boxes = [boxes[img["id"]] for img in self.images]
        self.contexts = self._load_road_types(dataset_root, [img["id"] for img in self.images], version)

        self.train = train
        self.short_side = short_side
        self.max_size = max_size

    def _load_road_types(self, dataset_root, frame_ids, version):
        #Imported here so classification runs don't need the zod devkit
        from zod import ZodFrames
        zod_frames = ZodFrames(dataset_root, version)
        contexts = []
        for frame_id in frame_ids:
            road_type = zod_frames[frame_id].metadata.road_type.lower().replace("_", "-").replace(" ", "-")
            if road_type not in self.road_types:
                raise ValueError(f"Frame {frame_id}: unknown road type {road_type!r}, expected one of {self.road_types}")
            contexts.append(self.road_types.index(road_type))
        return contexts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        w, h = info["width"], info["height"]
        scale = min(self.short_side / min(w, h), self.max_size / max(w, h))
        new_w, new_h = round(w * scale), round(h * scale)

        img = Image.open(info["file_name"])
        #Let the JPEG decoder skip work: decodes at the smallest of 1/2, 1/4, 1/8 scale
        #that is still >= the target size, then resize the rest of the way on the CPU
        img.draft("RGB", (new_w, new_h))
        img = img.convert("RGB").resize((new_w, new_h), Image.BILINEAR)
        image = TF.to_tensor(img)

        boxes = torch.tensor([b[:4] for b in self.boxes[idx]], dtype=torch.float32).reshape(-1, 4)
        boxes *= torch.tensor([new_w / w, new_h / h, new_w / w, new_h / h])
        labels = torch.tensor([b[4] for b in self.boxes[idx]], dtype=torch.int64)

        if self.train and torch.rand(()) < 0.5:
            image = image.flip(-1)
            boxes[:, [0, 2]] = new_w - boxes[:, [2, 0]]

        target = {"boxes": boxes, "labels": labels, "image_id": info["id"]}
        return image, target, self.contexts[idx]

    #DataLoader collate_fn: detection batches are lists, images differ in size after augmentation
    @staticmethod
    def collate(batch):
        images, targets, contexts = zip(*batch)
        return list(images), list(targets), torch.tensor(contexts)
