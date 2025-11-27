from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list[str]:
    """
    Generate caption(s) for a specific view.

    We keep this deliberately simple and robust:
      - Parse the detection list for this view from the info.json
      - Count how many karts are present (class_id == 1)
      - Produce a short natural language caption describing the scene

    Returns:
        A list of caption strings for this (image, view).
    """
    info_path = Path(info_path)
    with open(info_path) as f:
        info = json.load(f)

    # Default caption if anything goes wrong
    base_caption = "A SuperTuxKart racing scene with karts on a track."

    # Try to make the caption a bit more informative
    try:
        detections = info.get("detections", [])
        if len(detections) == 0 or view_index >= len(detections):
            return [base_caption]

        frame_detections = detections[view_index]

        # Count how many karts (class_id == 1)
        num_karts = 0
        for det in frame_detections:
            # Each detection is [class_id, track_id, x1, y1, x2, y2]
            class_id = int(det[0])
            if class_id == 1:
                num_karts += 1

        if num_karts <= 0:
            caption = base_caption
        elif num_karts == 1:
            caption = "A SuperTuxKart race scene with a single kart driving on the track."
        elif num_karts == 2:
            caption = "A SuperTuxKart race scene with two karts racing on the track."
        else:
            caption = f"A SuperTuxKart race scene with {num_karts} karts racing on the track."

        return [caption]
    except Exception:
        # Be safe: if parsing fails for any reason, still return something
        return [base_caption]


def check_caption(info_file: str, view_index: int):
    """
    Visualize one frame with detections and show the generated caption.
    """
    info_path = Path(info_file)

    # Recover image file name from info file + view index
    base_name = info_path.stem.replace("_info", "")  # e.g. 00000_info -> 00000
    image_pattern = f"{base_name}_{view_index:02d}_im.jpg"
    image_files = list(info_path.parent.glob(image_pattern))
    if not image_files:
        raise FileNotFoundError(f"Could not find image matching pattern {image_pattern} next to {info_file}")
    image_file = image_files[0]

    # Draw detections using helper from generate_qa.py
    annotated_image = draw_detections(str(image_file), str(info_file))

    # Generate caption(s) for this view
    captions = generate_caption(str(info_file), view_index)
    caption = captions[0] if captions else ""

    # Extract frame_id for nicer title (optional)
    frame_id, view_idx = extract_frame_info(str(image_file))

    # Show the result
    plt.figure(figsize=(8, 5))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {frame_id}, View {view_idx}\n{caption}")
    plt.tight_layout()
    plt.show()


def build_captions(split: str = "train", data_dir: str = "data", max_views: int = 8):
    """
    Build caption pairs for a given split (e.g. train, valid) and save to ..._captions.json.

    It will:
      - Look for all *_info.json files in data/<split>/
      - For each info file, iterate over possible view indices (0..max_views-1)
      - If the corresponding image exists, generate caption(s)
      - Save a list of { "image_file": "...", "caption": "..." } dicts into
        data/<split>/<split>_captions.json

    The 'image_file' field is stored relative to the root data directory,
    so CaptionDataset can reconstruct the full path as:
        os.path.join(data_dir, caption["image_file"])
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split

    info_files = sorted(split_dir.glob("*_info.json"))
    if not info_files:
        raise FileNotFoundError(f"No *_info.json files found in {split_dir}")

    all_captions = []

    for info_path in info_files:
        base_name = info_path.stem.replace("_info", "")  # e.g. 00000

        for view_index in range(max_views):
            image_name = f"{base_name}_{view_index:02d}_im.jpg"
            image_path = split_dir / image_name
            if not image_path.exists():
                # Some frames might not have all view indices; skip gracefully
                continue

            captions = generate_caption(str(info_path), view_index)
            for cap in captions:
                all_captions.append(
                    {
                        # Store path relative to root data dir, like "train/00000_00_im.jpg"
                        "image_file": f"{split}/{image_name}",
                        "caption": cap,
                    }
                )

    output_path = split_dir / f"{split}_captions.json"
    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Saved {len(all_captions)} caption pairs to {output_path}")


"""
Usage Example: Visualize captions for a specific file and view:
   python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0

Build caption dataset for training:
   python -m homework.generate_captions build --split train --data_dir data
"""


def main():
    fire.Fire({"check": check_caption, "build": build_captions})


if __name__ == "__main__":
    main()