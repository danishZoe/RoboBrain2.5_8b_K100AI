"""Offline smoke samples (no external network, official demo images unavailable).

Hand-drawn synthetic scenes covering general VQA / grounding / pointing / trajectory.
These are placeholder smoke samples, not an official benchmark distribution.
"""
import json
import os

from PIL import Image, ImageDraw

WORK = os.environ.get("WORK_DIR", "/root/private_data/workspace/robo")
IMG_DIR = os.path.join(WORK, "samples", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def scene_red_hat():
    """Red hat + white cup on a table (grounding / pointing)."""
    im = Image.new("RGB", (640, 480), (205, 198, 190))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 360, 640, 480), fill=(150, 112, 72))
    d.ellipse((250, 130, 390, 250), fill=(205, 40, 40))
    d.rectangle((205, 240, 435, 272), fill=(165, 25, 25))
    d.rectangle((455, 250, 530, 355), fill=(245, 245, 245))
    d.ellipse((495, 250, 530, 275), fill=(235, 235, 235))
    d.ellipse((455, 340, 530, 365), fill=(235, 235, 235))
    d.arc((520, 280, 570, 340), start=270, end=90, fill=(240, 240, 240), width=12)
    return im


def scene_objects():
    """Table with banana on plate, cup, apple (trajectory / general)."""
    im = Image.new("RGB", (640, 480), (215, 210, 200))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 330, 640, 480), fill=(160, 120, 80))
    d.ellipse((80, 300, 300, 380), fill=(238, 238, 245))
    d.ellipse((110, 310, 270, 365), fill=(228, 228, 235))
    d.rounded_rectangle((150, 250, 260, 330), radius=30, fill=(240, 205, 60))
    d.rounded_rectangle((340, 250, 420, 340), radius=16, fill=(200, 60, 55))
    d.ellipse((470, 270, 540, 340), fill=(190, 45, 50))
    return im


def scene_navigation():
    """Room: sofa + doorway + coffee table (navigation / pointing)."""
    im = Image.new("RGB", (640, 480), (225, 222, 215))
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 600, 440), outline=(120, 115, 105), width=6)
    d.rectangle((70, 120, 300, 300), fill=(90, 120, 165))
    d.rectangle((90, 140, 280, 200), fill=(110, 140, 185))
    d.rectangle((420, 60, 540, 150), fill=(150, 140, 125))
    d.ellipse((380, 300, 460, 380), fill=(150, 135, 110))
    return im


IMAGES = {
    "red_hat.png": scene_red_hat,
    "objects.png": scene_objects,
    "navigation.png": scene_navigation,
}
for name, fn in IMAGES.items():
    path = os.path.join(IMG_DIR, name)
    fn().save(path)
    print("wrote", path)

TASKS = [
    {"id": "general_01", "task": "general", "image": "red_hat.png",
     "prompt": "What is shown in this image?", "max_new_tokens": 96,
     "expect": "描述画面中有一顶红帽和杯子"},
    {"id": "grounding_01", "task": "grounding", "image": "red_hat.png",
     "prompt": "the red hat", "max_new_tokens": 64, "do_sample": False,
     "expect": "返回红帽所在位置"},
    {"id": "pointing_01", "task": "pointing", "image": "red_hat.png",
     "prompt": "the affordance area for holding the cup", "max_new_tokens": 64,
     "do_sample": False, "expect": "返回杯子可抓取区域"},
    {"id": "trajectory_01", "task": "trajectory", "image": "objects.png",
     "prompt": "reach for the banana on the plate", "max_new_tokens": 128,
     "do_sample": False,
     "expect": "返回抓取香蕉的关键点序列"},
    {"id": "general_02", "task": "general", "image": "navigation.png",
     "prompt": "Describe the scene in this image.", "max_new_tokens": 96,
     "expect": "描述房间内有沙发、门和茶几"},
    {"id": "navigation_01", "task": "pointing", "image": "navigation.png",
     "prompt": "Identify spot within the sofa in the house", "max_new_tokens": 64,
     "expect": "返回沙发上的目标点"},
]
with open(os.path.join(WORK, "samples", "tasks.json"), "w", encoding="utf-8") as f:
    json.dump(TASKS, f, ensure_ascii=False, indent=2)
print("wrote", os.path.join(WORK, "samples", "tasks.json"), "(%d tasks)" % len(TASKS))