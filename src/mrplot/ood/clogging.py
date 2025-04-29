import logging
import sys

# Configure the root logger for the package
logger = logging.getLogger("mrplot.ood")
logger.setLevel(logging.WARNING)  # Default level

# Add a handler to output to stderr if no handlers are configured
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Example usage (can be removed or kept for testing)
# if __name__ == "__main__":
#     logger.setLevel(logging.DEBUG)  # Example: Set to DEBUG for verbose output
#     logger.debug("This is a debug message.")
#     logger.info("This is an info message.")
#     logger.warning("This is a warning message.")
#     logger.error("This is an error message.")
#     logger.critical("This is a critical message.")

