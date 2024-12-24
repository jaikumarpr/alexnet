import logging

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG) # Set the log level

# Create handlers
file_handler = logging.FileHandler('ops.log')
stdout_handler = logging.StreamHandler()

# Create formatters and set it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


def init():
    return logger