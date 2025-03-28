 # 激光车辆强化学习训练系统

## 项目简介

这个项目实现了一个基于PyBullet物理引擎的激光车辆竞技场环境，使用强化学习(PPO算法)训练车辆进行自主导航、激光对战和障碍物规避。通过多维度奖励机制和课程学习，智能体可以逐步掌握复杂的战斗策略。

## 快速开始

### 安装依赖

```bash
pip install numpy pybullet stable-baselines3 torch gymnasium
```

### 训练模型

```bash
# 标准训练
python advanced_train.py --render_mode=direct --total_timesteps=1000000

# GUI模式训练(可视化，但训练较慢)
python advanced_train.py --render_mode=gui --total_timesteps=100000
```

### 测试训练好的模型

```bash
python test_trained_model.py --model_path=./advanced_logs/final_model
```

### 手动控制激光车

```bash
python manual_control.py
```

## 参数调整指南

### 基础训练参数

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|----------|
| `total_timesteps` | 总训练步数 | 1,000,000 | 500,000-5,000,000 |
| `n_envs` | 并行环境数 | 8 | 1-16 |
| `learning_rate` | 学习率 | 5e-4 | 1e-5-1e-3 |
| `batch_size` | 批次大小 | 512 | 64-2048 |
| `n_steps` | 每次收集的步数 | 2048 | 1024-8192 |
| `gamma` | 折扣因子 | 0.99 | 0.9-0.999 |
| `ent_coef` | 熵系数 | 0.01 | 0.001-0.05 |

### 如何选择参数

**入门级配置** (稳定学习，适合初学者)：
```bash
python advanced_train.py --total_timesteps=500000 --n_envs=4 --batch_size=256 --learning_rate=3e-4
```

**标准配置** (平衡速度和性能)：
```bash
python advanced_train.py --total_timesteps=1000000 --n_envs=8 --batch_size=512 --learning_rate=5e-4
```

**高级配置** (最佳性能，需要强大硬件)：
```bash
python advanced_train.py --total_timesteps=2000000 --n_envs=16 --batch_size=1024 --n_steps=4096 --learning_rate=3e-4 --gamma=0.995 --gae_lambda=0.98
```

## 奖励系统调整

奖励系统对模型的行为有决定性影响。以下是主要的奖励组件及其作用：

| 奖励组件 | 默认权重 | 作用 |
|----------|----------|------|
| 基础生存奖励 | 1.0 | 鼓励车辆生存更长时间 |
| 移动奖励 | 3.0 | 鼓励车辆探索环境 |
| 旋转奖励 | 1.5 | 鼓励车辆转向和瞄准 |
| 碰撞惩罚 | 8.0 | 惩罚车辆与障碍物碰撞 |
| 激光命中奖励 | 5.0 | 奖励成功命中对手 |
| 目标奖励 | 15.0 | 奖励完成主要目标 |
| 时间惩罚 | 0.05 | 鼓励车辆高效行动 |

### 如何修改奖励配置

在`advanced_train.py`文件中找到以下代码段并修改权重值：

```python
self.reward_config = {
    'base': {'weight': 1.0},      # 基础生存奖励
    'distance': {'weight': 3.0},  # 移动奖励
    'rotation': {'weight': 1.5},  # 旋转奖励
    'collision': {'weight': 8.0}, # 碰撞惩罚
    'laser': {'weight': 5.0},     # 激光命中奖励
    'goal': {'weight': 15.0},     # 达成目标奖励
    'time': {'weight': 0.05}      # 时间惩罚
}
```

### 奖励调整策略

- **攻击型车辆**：增加`laser`和`goal`权重
- **防御型车辆**：增加`collision`权重，减小`time`惩罚
- **灵活型车辆**：增加`distance`和`rotation`权重

## 常见问题与解决方案

### 训练问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 训练不稳定 | 学习率过高 | 降低`learning_rate`值 |
| 训练太慢 | 批次大小小 | 增加`batch_size`和`n_envs` |
| 模型不探索 | 熵系数低 | 增加`ent_coef`值 |
| 训练崩溃 | 物理模拟问题 | 减少`n_envs`值，使用`direct`模式 |

### 行为问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 过度碰撞 | 碰撞惩罚不足 | 增加`collision`的权重 |
| 不积极攻击 | 激光奖励不足 | 增加`laser`和`goal`的权重 |
| 原地打转 | 移动奖励不足 | 增加`distance`的权重 |
| 行为随机 | 训练不足 | 增加训练步数，降低`ent_coef` |

## 高级功能

### 继续训练

从现有模型继续训练：

```bash
python advanced_train.py --continue_training --model_path=./advanced_logs/checkpoints/model_100000_steps
```

### 环境参数调整

除了训练参数外，还可以调整环境本身的参数：

```bash
# 修改难度和最大步数
python advanced_train.py --render_mode=direct --difficulty=0.5 --max_steps=2000
```

## 性能优化建议

1. **硬件资源**：使用GPU训练，CPU核心越多能支持的并行环境越多
2. **内存管理**：根据可用内存调整`batch_size`和`n_steps`
3. **渲染模式**：训练时使用`direct`模式，只在测试时使用`gui`模式
4. **并行度**：尽可能增加`n_envs`值，但要根据CPU核心数合理设置
5. **保存频率**：减少模型保存频率可以提高训练效率

希望这个指南能帮助你训练出性能优异的激光车辆模型！