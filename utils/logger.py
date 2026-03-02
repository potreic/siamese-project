import logging
import sys

def setup_logger(log_file="experiment.log"):
    logger = logging.getLogger("SiameseExperiment")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
