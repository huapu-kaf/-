"""
调试工具
"""

import sys
import os
import time
import traceback

class DebugLogger:
    """调试日志记录器，确保即使缓冲区不刷新也能记录日志"""
    
    def __init__(self, log_file='debug.log'):
        self.log_file = log_file
        self.start_time = time.time()
        
        # 记录初始信息
        self.log("调试日志开始记录")
        self.log(f"Python版本: {sys.version}")
        self.log(f"工作目录: {os.getcwd()}")
        self.log(f"命令行参数: {sys.argv}")
    
    def log(self, message):
        """记录消息到日志文件和标准输出"""
        timestamp = time.time() - self.start_time
        formatted_message = f"[{timestamp:.3f}s] {message}"
        
        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message + '\n')
            f.flush()  # 确保立即写入文件
        
        # 输出到标准输出
        print(formatted_message, flush=True)
    
    def log_exception(self, e=None):
        """记录异常信息"""
        if e:
            self.log(f"异常: {type(e).__name__}: {e}")
        
        # 获取完整的异常跟踪
        trace = traceback.format_exc()
        self.log(f"异常跟踪:\n{trace}")

def redirect_output():
    """重定向标准输出和错误到文件"""
    sys.stdout = open('stdout.log', 'w', encoding='utf-8', buffering=1)
    sys.stderr = open('stderr.log', 'w', encoding='utf-8', buffering=1)

def restore_output():
    """恢复标准输出和错误"""
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    if hasattr(sys.stderr, 'close'):
        sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def main():
    """演示调试工具的使用"""
    logger = DebugLogger()
    
    try:
        logger.log("测试调试日志")
        # 制造一个异常
        x = 1 / 0
    except Exception as e:
        logger.log_exception(e)
    
    logger.log("脚本执行完毕")

if __name__ == "__main__":
    main() 