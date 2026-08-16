import sys
from loguru import logger

# Remove default logger
logger.remove()

# Add console logger with beautiful formatting
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Optional: Add file logger if we want to save logs later
# logger.add("logs/app.log", rotation="10 MB", level="DEBUG")
