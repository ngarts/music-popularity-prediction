import logging
import os
import constants

# Ensure the logs directory exists
os.makedirs(constants.LOG_DIR, exist_ok=True)

# Create logger
logger = logging.getLogger("music_analysis")
logger.setLevel(logging.INFO)  # Set log level (change to DEBUG if needed)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console Handler (optional, for logging in the terminal)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File Handler (ensures logs are saved in a file)
file_handler = logging.FileHandler(constants.LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
