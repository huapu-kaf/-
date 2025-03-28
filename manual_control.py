import os
import time
import argparse
import numpy as np
import pybullet as p

from advanced_train import AdvancedLaserVehicleEnv

# 定义按键常量
KEY_ESC = 27
KEY_SPACE = 32
KEY_R = 114
KEY_W = 119
KEY_A = 97
KEY_S = 115
KEY_D = 100
KEY_UP = 65297
KEY_DOWN = 65298
KEY_LEFT = 65295
KEY_RIGHT = 65296

# 定义按键状态常量
KEY_IS_DOWN = 1
KEY_WAS_TRIGGERED = 2
KEY_WAS_RELEASED = 4

def parse_args():
    parser = argparse.ArgumentParser(description='手动控制激光车')
    parser.add_argument('--render_mode', type=str, default='gui', choices=['gui', 'direct'],
                        help='渲染模式: gui或direct')
    parser.add_argument('--difficulty', type=str, default='medium', 
                        choices=['easy', 'medium', 'hard', 'extreme'],
                        help='难度级别')
    return parser.parse_args()

def print_controls():
    """打印控制说明"""
    print("\n--- 激光车手动控制 ---")
    print("W/↑: 前进")
    print("S/↓: 后退")
    print("A/←: 左转")
    print("D/→: 右转")
    print("空格: 发射激光")
    print("R: 重置环境")
    print("ESC: 退出")
    print("----------------------\n")

def get_action_from_keys(keys):
    """根据按键获取动作"""
    # 默认动作: [前进/后退, 左转/右转]
    action = np.array([0.0, 0.0])
    
    # 前进/后退控制 (W/S 或 上/下箭头)
    if KEY_W in keys or KEY_UP in keys:
        action[0] = 1.0  # 前进
    elif KEY_S in keys or KEY_DOWN in keys:
        action[0] = -1.0  # 后退
    
    # 左转/右转控制 (A/D 或 左/右箭头)
    if KEY_A in keys or KEY_LEFT in keys:
        action[1] = 1.0  # 左转
    elif KEY_D in keys or KEY_RIGHT in keys:
        action[1] = -1.0  # 右转
    
    return action

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 将难度字符串转换为数值
    difficulty_map = {
        'easy': 0.3,
        'medium': 0.5,
        'hard': 0.7,
        'extreme': 1.0
    }
    difficulty_value = difficulty_map.get(args.difficulty, 0.5)
    
    # 打印控制说明
    print_controls()
    
    try:
        # 创建环境
        env = AdvancedLaserVehicleEnv(
            render_mode=args.render_mode,
            difficulty=difficulty_value
        )
        
        # 初始化环境
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        # 主循环
        last_info_time = time.time()
        reset_requested = False
        exit_requested = False
        
        while not exit_requested:
            # 获取按键状态
            keys = p.getKeyboardEvents()
            
            # 检查ESC键退出
            if KEY_ESC in keys and keys[KEY_ESC] & KEY_WAS_TRIGGERED:
                print("\n检测到ESC键，准备退出...")
                exit_requested = True
                continue
            
            # 检查是否需要重置环境
            if KEY_R in keys and keys[KEY_R] & KEY_WAS_TRIGGERED and not reset_requested:
                print("\n重置环境...")
                obs, _ = env.reset()
                done = False
                truncated = False
                episode_reward = 0
                step_count = 0
                reset_requested = True
            elif KEY_R not in keys:
                reset_requested = False
            
            # 如果回合结束，重置环境
            if done or truncated:
                print(f"\n回合结束，总步数: {step_count}, 总奖励: {episode_reward:.2f}")
                obs, _ = env.reset()
                done = False
                truncated = False
                episode_reward = 0
                step_count = 0
            
            # 获取按键动作
            action = get_action_from_keys(keys)
            
            # 检查是否发射激光
            if KEY_SPACE in keys and keys[KEY_SPACE] & KEY_IS_DOWN:
                # 这里我们通过环境直接控制车辆发射激光
                if hasattr(env, 'arena') and hasattr(env.arena, 'vehicle'):
                    env.arena.vehicle.fire_laser()
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # 每秒打印一次状态信息
            current_time = time.time()
            if current_time - last_info_time >= 1.0:
                print(f"步骤: {step_count}, 奖励: {reward:.2f}, 累计奖励: {episode_reward:.2f}")
                if 'reward_components' in info:
                    for k, v in info['reward_components'].items():
                        if v != 0:
                            print(f"  - {k}: {v:.2f}")
                last_info_time = current_time
            
            # 控制循环速率
            time.sleep(0.05)
        
    except Exception as e:
        print(f"运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保环境正确关闭
        if 'env' in locals():
            env.close()
            print("环境已关闭")

if __name__ == "__main__":
    main() 