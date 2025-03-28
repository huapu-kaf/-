"""
简单的环境测试脚本
"""

import os
import sys
import time
import numpy as np

print("脚本开始执行...")

try:
    from advanced_train import AdvancedLaserVehicleEnv
    print("成功导入AdvancedLaserVehicleEnv")
except ImportError as e:
    print(f"导入AdvancedLaserVehicleEnv失败: {e}")
    sys.exit(1)

def main():
    print("创建环境...")
    
    try:
        # 创建环境
        env = AdvancedLaserVehicleEnv(render_mode='gui')
        print("环境创建成功")
        
        # 重置环境
        print("重置环境...")
        obs = env.reset()
        print(f"初始观测形状: {obs[0].shape}")
        
        # 运行一些随机动作
        print("开始执行随机动作...")
        for i in range(100):
            action = np.array([-0.5, 0.5])  # 简单的左转动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i % 10 == 0:
                print(f"步数: {i}, 奖励: {reward}")
            
            if terminated or truncated:
                print("回合结束")
                obs = env.reset()
            
            time.sleep(0.05)
        
        # 关闭环境
        print("关闭环境...")
        env.close()
        print("环境已关闭")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print("脚本执行完毕") 