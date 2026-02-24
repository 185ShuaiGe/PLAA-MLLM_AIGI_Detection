
import logging
import os
from typing import Optional
from datetime import datetime
from configs.path_config import PathConfig


class Logger:
    def __init__(self, name: str = "PLAA_MLLM", log_dir: Optional[str] = None):
        self.logger = None
        self.log_dir = log_dir
        self._setup_logger(name)

    def _setup_logger(self, name: str) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(self.log_dir, f'{name}_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def debug(self, msg: str) -> None:
        if self.logger:
            self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        if self.logger:
            self.logger.warning(msg)

    def error(self, msg: str) -> None:
        if self.logger:
            self.logger.error(msg)

    def critical(self, msg: str) -> None:
        if self.logger:
            self.logger.critical(msg)

