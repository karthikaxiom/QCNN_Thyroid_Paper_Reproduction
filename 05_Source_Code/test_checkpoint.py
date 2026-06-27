import os

from utilities import checkpoint

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

checkpoint.initialize(PROJECT_ROOT)

checkpoint.save_checkpoint(25)

print(checkpoint.load_checkpoint())

checkpoint.clear_checkpoint()

print(checkpoint.checkpoint_exists())