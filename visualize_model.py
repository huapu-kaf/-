"""
激光车辆强化学习 - 模型可视化脚本
用于加载训练好的模型并在GUI模式下可视化其行为
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym
import traceback
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 添加调试信息
print("脚本开始执行...")
print(f"Python版本: {os.sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(f"命令行参数: {os.sys.argv}")

try:
    from advanced_train import AdvancedLaserVehicleEnv, CustomCNN
    print("成功导入自定义模块")
except ImportError as e:
    print(f"导入自定义模块失败: {e}")
    print("请确保你在正确的目录下运行此脚本")
    exit(1)

def try_find_model(model_path):
    """
    尝试查找模型文件，处理各种路径和权限问题
    
    参数:
        model_path: 模型路径
        
    返回:
        str: 有效的模型路径，如果找不到则返回None
    """
    # 标准化路径
    model_path = Path(model_path)
    
    # 检查是否有.zip扩展名的路径
    zip_path = Path(str(model_path) + '.zip')
    if zip_path.exists() and os.access(zip_path, os.R_OK):
        print(f"找到可读取的模型文件: {zip_path}")
        return str(model_path)
    
    print(f"无法访问模型文件: {zip_path}")
    
    # 检查是否有其他名称的模型
    parent_dir = model_path.parent
    if parent_dir.exists():
        # 查找有效的模型文件
        potential_models = list(parent_dir.glob('*.zip'))
        for model in potential_models:
            if os.access(model, os.R_OK):
                # 去掉.zip后缀
                valid_path = str(model)[:-4]
                print(f"找到替代模型: {valid_path}")
                return valid_path
    
    # 如果有checkpoints目录，检查那里
    checkpoints_dir = Path('./advanced_logs/checkpoints')
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob('*.zip'))
        if checkpoint_files:
            for model in sorted(checkpoint_files, key=lambda p: p.stat().st_mtime, reverse=True):
                if os.access(model, os.R_OK):
                    # 去掉.zip后缀
                    valid_path = str(model)[:-4]
                    print(f"找到检查点模型: {valid_path}")
                    return valid_path
    
    # 尝试找accessible_model
    accessible_model = Path('./advanced_logs/accessible_model.zip')
    if accessible_model.exists() and os.access(accessible_model, os.R_OK):
        print(f"找到可访问模型: {accessible_model}")
        return str(accessible_model)[:-4]
    
    print("未找到任何可用模型文件")
    return None

def visualize_environment(max_steps=1000, render_mode='gui'):
    """
    仅可视化环境，不加载模型
    
    参数:
        max_steps: 最大步数
        render_mode: 渲染模式，默认为'gui'
    """
    print("仅可视化环境模式")
    
    # 创建环境
    try:
        env = AdvancedLaserVehicleEnv(render_mode=render_mode)
        env = DummyVecEnv([lambda: env])
        print("环境创建成功")
    except Exception as e:
        print(f"创建环境时出错: {e}")
        traceback.print_exc()
        return
    
    try:
        print("\n开始环境可视化")
        # 适配新版和旧版reset接口
        try:
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
                print("使用旧版接口：env.reset() 只返回观测值")
        except Exception as e:
            print(f"重置环境时出错: {e}")
            traceback.print_exc()
            return
        
        for step in range(max_steps):
            # 随机动作
            action = np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1)]])
            
            # 执行动作
            try:
                step_result = env.step(action)
                if len(step_result) == 5:  # 新版接口 (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                elif len(step_result) == 4:  # 旧版接口 (obs, reward, done, info)
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                    print("使用旧版接口：env.step() 返回4个值")
            except Exception as e:
                print(f"执行动作时出错: {e}")
                traceback.print_exc()
                continue
            
            # 显示当前状态
            if step % 100 == 0:
                print(f"步数: {step+1}/{max_steps}")
            
            # 检查是否结束
            done_flag = terminated[0] if isinstance(terminated, np.ndarray) else terminated
            truncated_flag = truncated[0] if isinstance(truncated, np.ndarray) else truncated
            
            if done_flag or truncated_flag:
                print(f"回合结束，步数: {step+1}")
                # 重置环境，适配新旧接口
                try:
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple) and len(reset_result) == 2:
                        obs, _ = reset_result
                    else:
                        obs = reset_result
                except Exception as e:
                    print(f"重置环境时出错: {e}")
                    traceback.print_exc()
                    break
            
            # 添加一点延迟，使可视化更容易观察
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n用户中断可视化")
    except Exception as e:
        print(f"\n可视化过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        # 关闭环境
        try:
            env.close()
            print("环境已关闭")
        except Exception as e:
            print(f"关闭环境时出错: {e}")

def visualize_model(model_path, episodes=5, max_steps=1000, render_mode='gui', force_load=False):
    """
    可视化模型行为
    
    参数:
        model_path: 模型路径
        episodes: 运行的回合数
        max_steps: 每回合最大步数
        render_mode: 渲染模式，默认为'gui'
        force_load: 是否强制加载模型（忽略观测空间不匹配）
    """
    print(f"正在加载模型: {model_path}")
    
    # 创建环境
    try:
        env = AdvancedLaserVehicleEnv(render_mode=render_mode)
        env = DummyVecEnv([lambda: env])
        print("环境创建成功")
        
        # 打印观测空间信息
        print(f"环境观测空间: {env.observation_space}")
    except Exception as e:
        print(f"创建环境时出错: {e}")
        traceback.print_exc()
        return
    
    # 尝试找到有效的模型文件
    valid_model_path = try_find_model(model_path)
    if not valid_model_path:
        print("无法找到有效的模型文件，将使用随机动作")
        # 回退到环境可视化模式
        visualize_environment(max_steps=max_steps, render_mode=render_mode)
        return
    
    # 加载模型
    try:
        if force_load:
            print("强制加载模型（忽略观测空间不匹配）")
            # 使用自定义加载方式
            model = PPO.load(valid_model_path)
            # 手动设置环境
            model.set_env(env)
        else:
            model = PPO.load(valid_model_path, env=env)
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("如果是观测空间不匹配错误，请尝试使用 --force_load 选项")
        print("或者使用 --env_only 选项仅查看环境")
        traceback.print_exc()
        env.close()
        # 回退到环境可视化模式
        print("\n切换到环境可视化模式...")
        visualize_environment(max_steps=max_steps, render_mode=render_mode)
        return
    
    # 运行多个回合
    total_rewards = []
    
    try:
        for episode in range(episodes):
            print(f"\n开始回合 {episode+1}/{episodes}")
            
            # 适配新版和旧版reset接口
            try:
                reset_result = env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    obs, _ = reset_result
                else:
                    obs = reset_result
                    print("使用旧版接口：env.reset() 只返回观测值")
            except Exception as e:
                print(f"重置环境时出错: {e}")
                traceback.print_exc()
                continue
            
            episode_reward = 0
            
            for step in range(max_steps):
                # 预测动作
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except Exception as e:
                    print(f"预测动作时出错: {e}")
                    print("尝试使用随机动作...")
                    action = np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1)]])
                
                # 执行动作
                try:
                    step_result = env.step(action)
                    if len(step_result) == 5:  # 新版接口
                        obs, reward, terminated, truncated, info = step_result
                    elif len(step_result) == 4:  # 旧版接口
                        obs, reward, done, info = step_result
                        terminated = done
                        truncated = False
                        print("使用旧版接口：env.step() 返回4个值")
                except Exception as e:
                    print(f"执行动作时出错: {e}")
                    traceback.print_exc()
                    continue
                
                # 获取奖励值，适配不同格式
                if isinstance(reward, np.ndarray) and len(reward) > 0:
                    reward_value = reward[0]
                else:
                    reward_value = reward
                
                episode_reward += reward_value
                
                # 显示当前状态
                if step % 100 == 0 or step == max_steps - 1:
                    print(f"步数: {step+1}/{max_steps}, 当前奖励: {episode_reward:.2f}")
                
                # 检查是否结束
                done_flag = terminated[0] if isinstance(terminated, np.ndarray) else terminated
                truncated_flag = truncated[0] if isinstance(truncated, np.ndarray) else truncated
                
                if done_flag or truncated_flag:
                    print(f"回合结束，步数: {step+1}, 总奖励: {episode_reward:.2f}")
                    break
                
                # 添加一点延迟，使可视化更容易观察
                time.sleep(0.01)
            
            total_rewards.append(episode_reward)
            print(f"回合 {episode+1} 完成，总奖励: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\n用户中断可视化")
    except Exception as e:
        print(f"\n可视化过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        # 关闭环境
        try:
            env.close()
            print("环境已关闭")
        except Exception as e:
            print(f"关闭环境时出错: {e}")
    
    # 显示统计信息
    if total_rewards:
        print("\n统计信息:")
        print(f"平均奖励: {np.mean(total_rewards):.2f}")
        print(f"最高奖励: {np.max(total_rewards):.2f}")
        print(f"最低奖励: {np.min(total_rewards):.2f}")
        print(f"奖励标准差: {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="激光车辆强化学习模型可视化")
    parser.add_argument('--model_path', type=str, default='./advanced_logs/final_model',
                      help='模型路径')
    parser.add_argument('--episodes', type=int, default=5,
                      help='运行的回合数')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='每回合最大步数')
    parser.add_argument('--render_mode', type=str, default='gui',
                      choices=['gui', 'direct'],
                      help='渲染模式')
    parser.add_argument('--force_load', action='store_true',
                      help='强制加载模型（忽略观测空间不匹配）')
    parser.add_argument('--env_only', action='store_true',
                      help='仅可视化环境，不加载模型')
    
    args = parser.parse_args()
    
    # 仅可视化环境模式
    if args.env_only:
        visualize_environment(
            max_steps=args.max_steps,
            render_mode=args.render_mode
        )
        exit(0)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path + '.zip'):
        print(f"警告: 模型文件 {args.model_path}.zip 不存在，尝试查找替代模型...")
        
    # 开始可视化
    visualize_model(
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        force_load=args.force_load
    ) 