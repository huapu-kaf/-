"""
带调试功能的环境测试脚本
"""

import os
import sys
import time
import numpy as np
import traceback

# 设置线程数为1，避免多线程问题
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 直接输出到文件，避免缓冲区问题
log_file = open('debug_env.log', 'w', encoding='utf-8')

def log(message):
    """记录日志"""
    timestamp = time.strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    log_file.write(full_message + '\n')
    log_file.flush()

log("脚本开始执行...")
log(f"Python版本: {sys.version}")
log(f"当前工作目录: {os.getcwd()}")
log(f"命令行参数: {sys.argv}")

try:
    from advanced_train import AdvancedLaserVehicleEnv
    log("成功导入AdvancedLaserVehicleEnv")
except ImportError as e:
    log(f"导入AdvancedLaserVehicleEnv失败: {e}")
    traceback.print_exc(file=log_file)
    sys.exit(1)

def run_safe_test():
    """运行安全的测试，捕获异常"""
    log("准备创建环境...")
    
    try:
        # 创建环境
        env = AdvancedLaserVehicleEnv(render_mode='direct')  # 使用direct模式，避免GUI问题
        log("环境创建成功")
        
        # 重置环境
        log("重置环境...")
        obs = env.reset()
        log(f"初始观测形状: {obs[0].shape}")
        
        # 运行一些随机动作
        log("开始执行随机动作...")
        for i in range(10):  # 只运行10步，避免太长
            action = np.array([-0.5, 0.5])  # 简单的左转动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            log(f"步数: {i}, 奖励: {reward}")
            
            if terminated or truncated:
                log("回合结束")
                obs = env.reset()
            
            time.sleep(0.1)
        
        # 关闭环境
        log("关闭环境...")
        env.close()
        log("环境已关闭")
        
        return True
    except Exception as e:
        log(f"发生错误: {e}")
        traceback.print_exc(file=log_file)
        return False

if __name__ == "__main__":
    success = run_safe_test()
    log(f"测试{'成功' if success else '失败'}")
    log_file.close() 