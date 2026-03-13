import logging
import os
from datetime import datetime

class Logger:
    # 关键修改 1：定义一个类变量，用于全局共享同一个文件写入句柄
    _shared_file_handler = None  

    def __init__(self, name, log_dir=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # 阻止日志向上一级传递，避免在控制台重复打印两次
        self.logger.propagate = False 

        # 确保每个命名的 logger 只添加一次 handlers
        if not self.logger.handlers:
            # 定义统一的日志格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            # 1. 挂载控制台 Handler（所有 logger 都在控制台输出）
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # 2. 如果传入了 log_dir 且全局 File Handler 还没有创建过，则创建它
            if log_dir is not None and Logger._shared_file_handler is None:
                os.makedirs(log_dir, exist_ok=True)
                # 可以自定义你的日志文件名，这里以时间戳命名为例
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 统一保存为 DS_MoME_Training_xxx.log
                log_file = os.path.join(log_dir, f"DS_MoME_Training_{timestamp}.log") 
                
                # 创建文件处理器
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setFormatter(formatter)
                
                # 赋值给类变量，供全局共享
                Logger._shared_file_handler = file_handler

            # 3. 关键修改 2：只要全局共享的 File Handler 存在，就挂载到当前实例上
            if Logger._shared_file_handler is not None:
                self.logger.addHandler(Logger._shared_file_handler)

    # 代理 logging 的基本方法
    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
        
    def debug(self, msg):
        self.logger.debug(msg)