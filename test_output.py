"""
测试输出脚本
"""

import os
import sys
import time

def main():
    print("测试输出脚本开始执行...")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"命令行参数: {sys.argv}")
    
    # 测试标准输出
    print("这是标准输出")
    
    # 测试标准错误
    sys.stderr.write("这是标准错误\n")
    sys.stderr.flush()
    
    # 测试延迟输出
    for i in range(5):
        print(f"延迟输出 {i+1}")
        sys.stdout.flush()
        time.sleep(1)
    
    print("测试输出脚本执行完毕")

if __name__ == "__main__":
    main() 