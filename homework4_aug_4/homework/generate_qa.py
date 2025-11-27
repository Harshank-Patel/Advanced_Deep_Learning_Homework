import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> dict:
    """
    Extract kart objects from the info.json file.
    
    Returns:
        dict: {
            "all_karts": list of all kart objects,
            "ego_kart": the kart object with track_id 0 (or None),
            "other_karts": list of opponent karts
        }
    """
    info_path = Path(info_path)
    with open(info_path) as f:
        info = json.load(f)

    if view_index >= len(info.get("detections", [])):
        return {"all_karts": [], "ego_kart": None, "other_karts": []}

    frame_detections = info["detections"][view_index]
    
    # NOTE: The info.json coordinates are in 600x400 space. 
    # We should use ORIGINAL_WIDTH/HEIGHT for validity checks to be accurate to the data source.
    valid_w, valid_h = ORIGINAL_WIDTH, ORIGINAL_HEIGHT

    all_karts = []
    ego_kart = None

    for det in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = det
        class_id, track_id = int(class_id), int(track_id)
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # Only Karts (class_id 1)
        if class_id != 1:
            continue

        # Check visibility
        if x2 < 0 or x1 > valid_w or y2 < 0 or y1 > valid_h:
            continue
            
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        kart_name = "the ego car" if track_id == 0 else f"Kart {track_id}"

        kart_obj = {
            "track_id": track_id,
            "kart_name": kart_name,
            "center_x": center_x,
            "center_y": center_y,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        }
        
        all_karts.append(kart_obj)
        if track_id == 0:
            ego_kart = kart_obj
            
    other_karts = [k for k in all_karts if k["track_id"] != 0]
    
    return {
        "all_karts": all_karts,
        "ego_kart": ego_kart,
        "other_karts": other_karts
    }


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.
    """
    with open(info_path) as f:
        info = json.load(f)
    
    track_name = info.get("track", "unknown track").strip()
    
    # Clean up names like "stk_cocoa_temple.sarb" -> "Cocoa Temple"
    if track_name.endswith(".sarb"):
        track_name = track_name[:-5]
    if track_name.startswith("stk_"):
        track_name = track_name[4:]
    
    return track_name.replace("_", " ").title()


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.
    """
    # Note: We ignore the passed img_width/height for filtering because
    # extract_kart_objects needs to use the JSON's native resolution (600x400)
    kart_data = extract_kart_objects(info_path, view_index)
    
    all_karts = kart_data["all_karts"]
    ego_kart = kart_data["ego_kart"]
    other_karts = kart_data["other_karts"]
    track_name = extract_track_info(info_path)
    
    qa_pairs = []

    # 1. Ego car question
    if ego_kart:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": ego_kart["kart_name"]
        })

    # 2. Total karts question
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(all_karts))
    })

    # 3. Track information questions
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # 4. Relative position questions for each kart
    if ego_kart and other_karts:
        ex, ey = ego_kart["center_x"], ego_kart["center_y"]
        
        for k in other_karts:
            kx, ky = k["center_x"], k["center_y"]
            name = k["kart_name"]
            
            # Left/Right
            pos_h = "left" if kx < ex else "right"
            qa_pairs.append({
                "question": f"Is {name} to the left or right of the ego car?",
                "answer": pos_h
            })
            
            # Front/Behind (Image coordinates: smaller Y is higher up = "in front")
            pos_v = "in front of" if ky < ey else "behind"
            qa_pairs.append({
                "question": f"Is {name} in front of or behind the ego car?",
                "answer": pos_v
            })

    # 5. Counting questions
    if ego_kart:
        ex, ey = ego_kart["center_x"], ego_kart["center_y"]
        
        count_left = sum(1 for k in other_karts if k["center_x"] < ex)
        count_right = sum(1 for k in other_karts if k["center_x"] >= ex)
        count_front = sum(1 for k in other_karts if k["center_y"] < ey)
        count_behind = sum(1 for k in other_karts if k["center_y"] >= ey)
        
        qa_pairs.append({
            "question": "How many karts are to the left of the ego car?",
            "answer": str(count_left)
        })
        qa_pairs.append({
            "question": "How many karts are to the right of the ego car?",
            "answer": str(count_right)
        })
        qa_pairs.append({
            "question": "How many karts are in front of the ego car?",
            "answer": str(count_front)
        })
        qa_pairs.append({
            "question": "How many karts are behind the ego car?",
            "answer": str(count_behind)
        })

    return qa_pairs


def build_qa_pairs(split: str = "train", data_dir: str = "data", max_views: int = 8):
    """
    Build QA pairs for a given split and save to json.
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    info_files = sorted(split_dir.glob("*_info.json"))
    if not info_files:
        print(f"No info files found in {split_dir}")
        return

    all_qa_pairs = []
    
    for info_path in info_files:
        base_name = info_path.stem.replace("_info", "")
        
        for view_index in range(max_views):
            image_name = f"{base_name}_{view_index:02d}_im.jpg"
            image_path = split_dir / image_name
            
            if not image_path.exists():
                continue
                
            qa_pairs = generate_qa_pairs(str(info_path), view_index)
            
            for qa in qa_pairs:
                all_qa_pairs.append({
                    "image_file": f"{split}/{image_name}",
                    "question": qa["question"],
                    "answer": qa["answer"]
                })
                
    # Save as {split}_qa_pairs.json so VQADataset picks it up
    out_path = split_dir / f"{split}_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
        
    print(f"Saved {len(all_qa_pairs)} QA pairs to {out_path}")


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def main():
    fire.Fire({"check": check_qa_pairs, "build": build_qa_pairs})


if __name__ == "__main__":
    main()