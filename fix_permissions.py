import os
import stat
import shutil
import glob
import sys
import traceback

def fix_permissions():
    """修复权限，允许读写advanced_logs目录及其子文件"""
    print("开始修复权限问题...")
    
    # 检查advanced_logs目录是否存在
    if not os.path.exists('./advanced_logs'):
        print("创建advanced_logs目录")
        try:
            os.makedirs('./advanced_logs', exist_ok=True)
        except Exception as e:
            print(f"创建目录时出错: {e}")
            traceback.print_exc()
            return False
    
    # 检查tensorboard目录
    tb_dir = './advanced_logs/tensorboard'
    if not os.path.exists(tb_dir):
        print(f"创建目录: {tb_dir}")
        try:
            os.makedirs(tb_dir, exist_ok=True)
        except Exception as e:
            print(f"创建tensorboard目录时出错: {e}")
    
    # 检查checkpoints目录
    ckpt_dir = './advanced_logs/checkpoints'
    if not os.path.exists(ckpt_dir):
        print(f"创建目录: {ckpt_dir}")
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
        except Exception as e:
            print(f"创建checkpoints目录时出错: {e}")
    
    # 设置目录权限
    try:
        print(f"设置目录权限: ./advanced_logs")
        os.chmod('./advanced_logs', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 读写执行权限0777
        
        # 设置子目录权限
        for subdir in ['tensorboard', 'checkpoints']:
            subdir_path = f'./advanced_logs/{subdir}'
            if os.path.exists(subdir_path):
                print(f"设置目录权限: {subdir_path}")
                os.chmod(subdir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except Exception as e:
        print(f"设置目录权限时出错: {e}")
        traceback.print_exc()
    
    # 检查现有文件
    try:
        for root, dirs, files in os.walk('./advanced_logs'):
            # 设置子目录权限
            for d in dirs:
                dir_path = os.path.join(root, d)
                print(f"设置子目录权限: {dir_path}")
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            
            # 设置文件权限
            for f in files:
                file_path = os.path.join(root, f)
                print(f"设置文件权限: {file_path}")
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)  # 读写权限0666
    except Exception as e:
        print(f"设置文件权限时出错: {e}")
        traceback.print_exc()
    
    # 查找并处理checkpoint文件
    try:
        # 检查final_model是否存在
        final_model_path = './advanced_logs/final_model'
        if os.path.exists(final_model_path):
            print(f"找到final_model: {final_model_path}")
            try:
                os.chmod(final_model_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                print(f"成功设置final_model权限")
            except Exception as e:
                print(f"设置final_model权限时出错: {e}")
            
            # 创建备份
            accessible_copy = './advanced_logs/final_model_copy.zip'
            try:
                print(f"创建可访问的模型副本: {accessible_copy}")
                shutil.copy2(final_model_path, accessible_copy)
                os.chmod(accessible_copy, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                print(f"成功创建可访问的模型副本")
            except Exception as e:
                print(f"创建模型副本时出错: {e}")
                traceback.print_exc()
        else:
            print("未找到final_model文件")
            
            # 查找最新的checkpoint文件
            checkpoint_files = glob.glob('./advanced_logs/checkpoints/*.zip')
            if checkpoint_files:
                checkpoint_files.sort(key=os.path.getmtime, reverse=True)
                latest_checkpoint = checkpoint_files[0]
                print(f"找到最新的checkpoint: {latest_checkpoint}")
                
                try:
                    os.chmod(latest_checkpoint, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                    print(f"成功设置checkpoint权限")
                except Exception as e:
                    print(f"设置checkpoint权限时出错: {e}")
                
                # 创建一个可访问的模型副本
                accessible_copy = './advanced_logs/latest_model_copy.zip'
                try:
                    print(f"创建可访问的模型副本: {accessible_copy}")
                    shutil.copy2(latest_checkpoint, accessible_copy)
                    os.chmod(accessible_copy, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                    print(f"成功创建可访问的模型副本")
                except Exception as e:
                    print(f"创建模型副本时出错: {e}")
                    traceback.print_exc()
            else:
                print("未找到任何checkpoint文件")
    except Exception as e:
        print(f"处理checkpoint文件时出错: {e}")
        traceback.print_exc()
    
    # 创建测试模型，验证权限
    try:
        test_model_path = './advanced_logs/test_model.zip'
        print(f"创建测试模型文件: {test_model_path}")
        with open(test_model_path, 'wb') as f:
            f.write(b'test model file')
        os.chmod(test_model_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        print(f"测试模型文件创建成功")
        
        # 验证权限
        if os.access(test_model_path, os.R_OK | os.W_OK):
            print("权限测试通过：文件可读可写")
        else:
            print("权限测试失败：文件不可读写")
            
    except Exception as e:
        print(f"创建测试模型文件时出错: {e}")
        traceback.print_exc()
    
    print("权限修复完成")
    return True

if __name__ == "__main__":
    success = fix_permissions()
    if success:
        print("成功修复权限问题")
        sys.exit(0)
    else:
        print("修复权限时出现问题")
        sys.exit(1) 