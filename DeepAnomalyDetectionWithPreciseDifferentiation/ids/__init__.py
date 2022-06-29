import os

from ids.utils import read_system_config
from ids.generate_default_config import SYSTEM_CONFIG_FILE_NAME

read_system_config(os.path.basename(SYSTEM_CONFIG_FILE_NAME))