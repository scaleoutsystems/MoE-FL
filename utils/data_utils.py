import os
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
    """ZOD Frames 2D detection from the devkit COCO export, with the weather as context.

    Expects a JSON written by zod_resize.py (images at short side `short_side`, long side
    capped at `max_size`). Returns (image, target, context): image is a float tensor,
    target holds boxes (xyxy, scaled from the original pixels to the image's),
    labels and the COCO image id. Crowd ("Unclear") boxes are left out, they are only
    ignore regions for evaluation.
    """

    #Context index = position in this list = expert id 
    weathers = ["clear", "cloudy", "rain", "fog", "snow"]
    #Detection label = position + 1, 0 is background
    classes = ["Vehicle", "VulnerableVehicle", "Pedestrian"]

    def __init__(self, coco_json, dataset_root, train=False, short_side=800, max_size=1333,
                 drop_cars=("bravo",)):
        self.coco_json = coco_json  # ground truth for evaluate_zod
        #Image paths in the JSON are relative to its directory (absolute ones are kept as they are)
        self.image_dir = os.path.dirname(os.path.abspath(coco_json))
        with open(coco_json) as f:
            coco = json.load(f)
        cat_to_label = {c["id"]: self.classes.index(c["name"]) + 1 for c in coco["categories"]}

        #Weather, car and day per frame from the cached metadata (see load_metadata)
        meta = {int(r["id"]): r for r in self.load_metadata(dataset_root)}
        #Frames of dropped cars (bravo: too few frames for a client) are excluded everywhere
        splits = {meta[img["id"]]["split"] for img in coco["images"]}
        assert len(splits) == 1, f"{coco_json} mixes the splits {splits}"
        self.images = [img for img in coco["images"] if meta[img["id"]]["car"] not in drop_cars]
        self.contexts = [self.weathers.index(meta[img["id"]]["weather"]) for img in self.images]
        self.cars = [meta[img["id"]]["car"] for img in self.images]
        self.days = [meta[img["id"]]["time"][:10] for img in self.images]

        size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
        boxes = defaultdict(list)
        for a in coco["annotations"]:
            if a["iscrowd"]:
                continue
            x, y, w, h = a["bbox"]
            #Clip to the image as Detectron2 does (annotations of cut-off objects extend beyond it);
            #empty boxes crash Faster R-CNN. Evaluation still uses the unclipped ground truth
            img_w, img_h = size[a["image_id"]]
            x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, img_w), min(y + h, img_h)
            if x2 > x1 and y2 > y1:
                boxes[a["image_id"]].append((x1, y1, x2, y2, cat_to_label[a["category_id"]]))
        self.boxes = [boxes[img["id"]] for img in self.images]

        self.train = train
        self.short_side = short_side
        self.max_size = max_size

    @staticmethod
    def load_metadata(root):
        """One row per frame (id, split, time, car, country, weather), read from
        single_frames/<id>/metadata.json without the devkit and cached in <root>/zod_metadata.json."""
        cache = os.path.join(root, "zod_metadata.json")
        if os.path.exists(cache):
            with open(cache) as f:
                return json.load(f)
        with open(os.path.join(root, "trainval-frames-full.json")) as f:
            split_of = {frame["id"]: split for split, frames in json.load(f).items() for frame in frames}
        def read(frame_id):
            with open(os.path.join(root, "single_frames", frame_id, "metadata.json")) as f:
                m = json.load(f)
            return {"id": frame_id, "split": split_of[frame_id], "time": m["time"], "car": m["collection_car"],
                    "country": m["country_code"], "weather": ZOD.weather_category(m["scraped_weather"])}
        with ThreadPoolExecutor(32) as ex:
            rows = list(ex.map(read, sorted(split_of)))
        with open(cache, "w") as f:
            json.dump(rows, f)
        return rows

    @staticmethod
    def weather_category(scraped_weather):
        """ZOD paper's 5 categories"""
        weather = scraped_weather.removesuffix("-day").removesuffix("-night")
        weather = "clear" if weather in ("partly-cloudy", "wind") else weather
        if weather not in ZOD.weathers:
            raise ValueError(f"Unknown weather {scraped_weather!r}")
        return weather

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        w, h = info["width"], info["height"]
        scale = min(self.short_side / min(w, h), self.max_size / max(w, h))
        new_w, new_h = round(w * scale), round(h * scale)

        #Images are pre-resized by zod_resize.py, the boxes are still in original pixels
        img = Image.open(os.path.join(self.image_dir, info["file_name"])).convert("RGB")
        assert img.size == (new_w, new_h), f"{info['file_name']} is {img.size}, expected {(new_w, new_h)}: run zod_resize.py"
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
