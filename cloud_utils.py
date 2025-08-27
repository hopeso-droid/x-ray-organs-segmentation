"""
云端部署工具模块
Cloud deployment utilities module
"""
import os
import sys
import platform
import tempfile
import logging
from pathlib import Path
import streamlit as st
import gc
import psutil

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()

def abs_path(path, path_type="current"):
    """
    云端兼容的路径处理函数，替代 QtFusion.path.abs_path
    
    Args:
        path (str): 相对路径
        path_type (str): 路径类型，固定为 "current"
    
    Returns:
        str: 绝对路径
    """
    if path_type == "current":
        base_path = get_project_root()
        full_path = base_path / path
        return str(full_path.resolve())
    else:
        return os.path.abspath(path)

def is_cloud_environment():
    """检测是否在云端环境运行"""
    # 检测常见的云端环境变量
    cloud_indicators = [
        'STREAMLIT_SHARING',  # Streamlit Cloud
        'HEROKUAPP',          # Heroku
        'GITHUB_ACTIONS',     # GitHub Actions
        'AWS_LAMBDA_FUNCTION_NAME',  # AWS Lambda
        'GOOGLE_CLOUD_PROJECT',      # Google Cloud
    ]
    
    return any(os.getenv(indicator) for indicator in cloud_indicators)

def setup_cloud_directories():
    """设置云端所需的目录结构"""
    project_root = get_project_root()
    
    # 创建必要的目录
    directories = [
        'tempDir',
        'icon',
        'weights',
        'uploads',
        'logs'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
    
    return project_root

def get_camera_names_cloud():
    """云端环境的摄像头名称获取（禁用摄像头功能）"""
    return ["摄像头检测关闭"]

def save_uploaded_file_cloud(uploaded_file):
    """云端环境的文件保存函数"""
    if uploaded_file is None:
        return None
    
    project_root = get_project_root()
    upload_dir = project_root / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

class CloudLogger:
    """云端日志记录器"""
    
    def __init__(self, log_file="app.log"):
        self.log_file = get_project_root() / "logs" / log_file
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log(self, message, level="INFO"):
        """记录日志"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"日志记录失败: {e}")
    
    def info(self, message):
        self.log(message, "INFO")
    
    def error(self, message):
        self.log(message, "ERROR")
    
    def warning(self, message):
        self.log(message, "WARNING")

# 全局云端日志器
cloud_logger = CloudLogger()

def init_cloud_environment():
    """初始化云端环境"""
    try:
        # 设置目录结构
        setup_cloud_directories()
        
        # 记录环境信息
        cloud_logger.info(f"Python版本: {sys.version}")
        cloud_logger.info(f"操作系统: {platform.system()} {platform.release()}")
        cloud_logger.info(f"项目根目录: {get_project_root()}")
        cloud_logger.info(f"云端环境检测: {is_cloud_environment()}")
        
        return True
    except Exception as e:
        print(f"云端环境初始化失败: {e}")
        return False
