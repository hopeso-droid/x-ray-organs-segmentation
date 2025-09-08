"""
AI 胸部X光病症检测系统 - Streamlit 版本
AI Chest X-ray Disease Detection System - Streamlit Version
"""
import os
import sys
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置页面配置
import streamlit as st

st.set_page_config(
    page_title="AI 胸部X光病症检测系统",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # AI 胸部X光病症检测系统
        
        **版本**: 1.0.0  
        **开发**: 合溪生物科技  
        **用途**: 医学影像科研和教学使用
        
        本系统基于深度学习技术，提供胸部X光的智能病症检测和分析功能。
        """
    }
)

try:
    # 导入胸部X光病症检测系统的web.py
    from web import Detection_UI
    
    def main():
        """主函数"""
        try:
            # 创建应用实例
            app = Detection_UI()
            
            # 运行主界面
            app.setupMainWindow()
            
        except Exception as e:
            st.error(f"⚠️ 应用运行错误: {str(e)}")
            st.error("请刷新页面重试，或联系技术支持。")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    st.error("⚠️ 系统模块加载失败")
    st.error(f"模块导入失败: {str(e)}")
    st.info("如果问题持续存在，请联系技术支持。")

except Exception as e:
    st.error("⚠️ 系统初始化失败")
    st.error(f"错误详情: {str(e)}")
    st.info("请刷新页面重试。")
