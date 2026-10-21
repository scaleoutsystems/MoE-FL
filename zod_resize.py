"""One-time preprocessing: resize the images of a devkit COCO export to the training size
(short side 800, long side <= 1333, i.e. 1333x751 for ZOD) and write a copy of the JSON
pointing to them. Annotations, width and height stay in original pixels: ZOD scales the
boxes, and evaluation uses the original sizes. Skips images that already exist, so an
interrupted run can simply be restarted.

    python zod_resize.py <devkit coco json> <output image dir> <output json>

The devkit JSON's file_names are relative to its own directory (or absolute).

file_name in the output JSON is relative to the JSON's own directory, so the data can be
moved (e.g. to a cluster) as long as the JSON and the image dir keep their relative positions.
"""
import os
import sys
import json
from multiprocessing import Pool
from PIL import Image

#Must match ZOD(short_side, max_size)
SHORT_SIDE, MAX_SIZE = 800, 1333

def resize(job):
    src, dst, width, height = job
    if os.path.exists(dst):
        return
    img = Image.open(src)
    w, h = img.size
    assert (w, h) == (width, height), f"{src} is {w}x{h}, the JSON says {width}x{height}"
    scale = min(SHORT_SIDE / min(w, h), MAX_SIZE / max(w, h))
    size = (round(w * scale), round(h * scale))
    img.draft("RGB", size)  # let the JPEG decoder skip work
    #Write to a temporary file first: a killed run must not leave a truncated image that a restart skips
    tmp = dst + ".tmp"
    img.convert("RGB").resize(size, Image.BILINEAR).save(tmp, format="JPEG", quality=95)
    os.replace(tmp, dst)

if __name__ == "__main__":
    src_json, out_dir, out_json = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    with open(src_json) as f:
        coco = json.load(f)

    src_dir = os.path.dirname(os.path.abspath(src_json))  # source file_names are relative to this
    json_dir = os.path.dirname(os.path.abspath(out_json))
    os.makedirs(json_dir, exist_ok=True)
    jobs = []
    for img in coco["images"]:
        dst = os.path.join(out_dir, f"{img['id']:06d}.jpg")
        jobs.append((os.path.join(src_dir, img["file_name"]), dst, img["width"], img["height"]))
        img["file_name"] = os.path.relpath(os.path.abspath(dst), json_dir)

    with Pool() as pool:
        for i, _ in enumerate(pool.imap_unordered(resize, jobs, chunksize=64)):
            if i % 5000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    missing = [dst for _, dst, _, _ in jobs if not os.path.exists(dst)]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[0]}"

    with open(out_json + ".tmp", "w") as f:
        json.dump(coco, f)
    os.replace(out_json + ".tmp", out_json)
    print(f"Done: {len(jobs)} images in {out_dir}, annotations in {out_json}")
