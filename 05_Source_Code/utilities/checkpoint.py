import os
import json

CHECKPOINT_FILE = None


def initialize(project_root):
    global CHECKPOINT_FILE

    CHECKPOINT_FILE = os.path.join(
        project_root,
        "06_Models",
        "quantum_checkpoint.json"
    )


def save_checkpoint(image_index):

    data = {
        "last_completed_image": image_index
    }

    with open(CHECKPOINT_FILE, "w") as file:
        json.dump(data, file, indent=4)


def load_checkpoint():

    if not os.path.exists(CHECKPOINT_FILE):
        return 0

    with open(CHECKPOINT_FILE, "r") as file:
        data = json.load(file)

    return data.get("last_completed_image", 0)


def checkpoint_exists():

    return os.path.exists(CHECKPOINT_FILE)


def clear_checkpoint():

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)