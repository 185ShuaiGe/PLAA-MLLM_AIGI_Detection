
import logging
import os
from typing import Optional
from datetime import datetime
from configs.path_config import PathConfig


class Logger:
    def __init__(self, name: str = "PLAA_MLLM", log_dir: Optional[str] = None):
        self.logger = None
        self.log_dir = log_dir

    def info(self, msg: str) -&gt; None:
        """
        记录 INFO 级别日志

        Args:
            msg: 日志消息
        """
        pass

    def debug(self, msg: str) -&gt; None:
        """
        记录 DEBUG 级别日志

        Args:
            msg: 日志消息
        """
        pass

    def warning(self, msg: str) -&gt; None:
        """
        记录 WARNING 级别日志

        Args:
            msg: 日志消息
        """
        pass

    def error(self, msg: str) -&gt; None:
        """
        记录 ERROR 级别日志

        Args:
            msg: 日志消息
        """
        pass

    def critical(self, msg: str) -&gt; None:
        """
        记录 CRITICAL 级别日志

        Args:
            msg: 日志消息
        """
        pass

    def _setup_logger(self) -&gt; None:
        """
        配置日志处理器和格式

        设置控制台输出和文件输出，指定日志格式
        """
        pass
