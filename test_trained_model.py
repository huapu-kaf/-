import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
import pybullet as p

from advanced_train import AdvancedLaserVehicleEnv

def parse_args():
    parser = argparse.ArgumentParser(description='测试训练好的激光车模型')
    parser.add_argument('--model_path', type=str, default='./advanced_logs/interrupted_model',
                        help='模型路径')
    parser.add_argument('--render_mode', type=str, default='gui', choices=['gui', 'direct'],
                        help='渲染模式: gui或direct')
    parser.add_argument('--episodes', type=int, default=5,
                        help='测试回合数')
    parser.add_argument('--difficulty', type=float, default=0.5, 
                        help='难度级别 (0.1-1.0)')
    parser.add_argument('--debug', action='store_true',
                        help='是否开启调试模式')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查模型文件是否存在
    model_path = args.model_path
    if not model_path.endswith('.zip'):
        model_path += '.zip'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    print(f"正在加载模型: {model_path}")
    
    try:
        # 创建环境
        env = AdvancedLaserVehicleEnv(
            render_mode=args.render_mode,
            difficulty=args.difficulty
        )
        
        # 加载模型
        model = PPO.load(model_path)
        print("模型加载成功!")
        
        # 测试模型
        total_rewards = []
        
        for episode in range(args.episodes):
            print(f"\n开始测试回合 {episode+1}/{args.episodes}")
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step_count = 0
            
            while not (done or truncated):
                # 使用模型预测动作
                action, _states = model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # 打印当前步骤信息
                if step_count % 10 == 0:
                    print(f"步骤: {step_count}, 奖励: {reward:.2f}, 累计奖励: {episode_reward:.2f}")
                    if 'reward_components' in info:
                        for k, v in info['reward_components'].items():
                            if v != 0:
                                print(f"  - {k}: {v:.2f}")
                
                # 如果在GUI模式下，添加一些延迟以便观察
                if args.render_mode == 'gui':
                    time.sleep(0.01)
            
            print(f"回合 {episode+1} 结束, 总步数: {step_count}, 总奖励: {episode_reward:.2f}")
            total_rewards.append(episode_reward)
        
        print("\n测试完成!")
        print(f"平均奖励: {np.mean(total_rewards):.2f}")
        print(f"奖励标准差: {np.std(total_rewards):.2f}")
        print(f"最高奖励: {np.max(total_rewards):.2f}")
        print(f"最低奖励: {np.min(total_rewards):.2f}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保环境正确关闭
        if 'env' in locals():
            env.close()
            print("环境已关闭")

if __name__ == "__main__":
    main() 