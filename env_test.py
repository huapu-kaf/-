import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
import traceback

# 尝试导入环境
try:
    from advanced_train import AdvancedLaserVehicleEnv
    print("成功导入环境类")
except Exception as e:
    print(f"导入环境类时出错: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_env(render_mode='gui', max_steps=500):
    """测试环境功能"""
    print(f"开始测试环境，渲染模式: {render_mode}")
    
    # 创建环境
    try:
        env = AdvancedLaserVehicleEnv(render_mode=render_mode)
        print("环境创建成功")
    except Exception as e:
        print(f"创建环境时出错: {e}")
        traceback.print_exc()
        return
    
    # 查看空间
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 重置环境
    try:
        print("重置环境...")
        obs, info = env.reset()
        print(f"观测形状: {obs.shape}")
        print(f"初始信息: {info}")
    except Exception as e:
        print(f"重置环境时出错: {e}")
        traceback.print_exc()
        env.close()
        return
    
    # 运行几步
    try:
        for step in range(max_steps):
            # 生成随机动作
            action = env.action_space.sample()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 50 == 0:
                print(f"步数: {step}, 奖励: {reward}")
            
            # 如果游戏结束则重置
            if terminated or truncated:
                print(f"环境结束，步数: {step}")
                obs, info = env.reset()
            
            # 降低速度以便观察
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("用户中断测试")
    except Exception as e:
        print(f"环境运行时出错: {e}")
        traceback.print_exc()
    finally:
        # 关闭环境
        env.close()
        print("环境已关闭")

if __name__ == "__main__":
    # 解析参数
    render_mode = 'gui'
    if len(sys.argv) > 1:
        render_mode = sys.argv[1]
    
    test_env(render_mode) 