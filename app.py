"""
细胞组织分割系统 - Streamlit Cloud 版本
Cell Tissue Segmentation System - Streamlit Cloud Version
"""
import os
import sys
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入云端工具
from cloud_utils import init_cloud_environment, is_cloud_environment, abs_path, cloud_logger

# 初始化云端环境
init_cloud_environment()

# 设置页面配置
import streamlit as st

st.set_page_config(
    page_title="AI细胞组织分割系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # AI细胞组织分割系统
        
        **版本**: 1.0.0  
        **开发**: 合溪生物科技  
        **用途**: 科研和教学使用
        
        本系统基于深度学习技术，提供细胞组织的智能分割和分析功能。
        """
    }
)

try:
    # 导入核心模块
    from web_cloud import Detection_UI_Cloud
    
    def main():
        """主函数"""
        try:
            # 记录应用启动
            cloud_logger.info("应用启动")
            cloud_logger.info(f"运行环境: {'云端' if is_cloud_environment() else '本地'}")
            
            # 创建应用实例
            app = Detection_UI_Cloud()
            
            # 运行主界面
            app.setupMainWindow()
            
        except Exception as e:
            cloud_logger.error(f"应用运行错误: {str(e)}")
            st.error(f"⚠️ 应用运行错误: {str(e)}")
            st.error("请刷新页面重试，或联系技术支持。")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    cloud_logger.error(f"模块导入失败: {str(e)}")
    st.error("⚠️ 系统模块加载失败")
    st.error("请确保所有依赖包已正确安装。")
    st.info("如果问题持续存在，请联系技术支持。")

except Exception as e:
    cloud_logger.error(f"系统错误: {str(e)}")
    st.error("⚠️ 系统初始化失败")
    st.error(f"错误详情: {str(e)}")
    st.info("请刷新页面重试。")
