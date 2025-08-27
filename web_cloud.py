"""
云端优化的细胞组织分割系统界面
Cloud-optimized Cell Tissue Segmentation System Interface
"""
import random
import tempfile
import time
import os
import cv2
import numpy as np
import streamlit as st
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
from hashlib import md5

# 导入云端工具
from cloud_utils import abs_path, is_cloud_environment, get_camera_names_cloud, save_uploaded_file_cloud, cloud_logger

# 定义默认值，防止导入失败
Label_list = ["细胞核", "细胞质", "组织结构", "血管", "细胞", "细胞膜"]
Chinese_to_English = {
    "细胞核": "Nucleus",
    "细胞质": "Cytoplasm", 
    "组织结构": "Tissue",
    "血管": "Vessel",
    "细胞": "Cell",
    "细胞膜": "Membrane"
}

# 导入本地模块
try:
    from log import ResultLogger, LogTable
    from model import Web_Detector
    
    # 尝试导入标签列表，如果失败则使用默认值
    try:
        from chinese_name_list import Label_list as _Label_list, Chinese_to_English as _Chinese_to_English
        Label_list = _Label_list
        Chinese_to_English = _Chinese_to_English
    except ImportError:
        cloud_logger.warning("使用默认的标签列表")
    
    try:
        from ui_style import def_css_hitml
    except ImportError:
        # 如果导入失败，创建一个空函数
        def def_css_hitml():
            pass
        cloud_logger.warning("UI样式模块未找到，使用默认样式")
    
    try:
        from utils import concat_results
    except ImportError:
        # 如果导入失败，创建一个简单的替代函数
        def concat_results(result, location, confidence, time_str):
            import pandas as pd
            return pd.DataFrame({
                '检测结果': [result],
                '位置': [location], 
                '置信度': [confidence],
                '用时': [time_str]
            })
        cloud_logger.warning("utils模块未找到，使用默认函数")
        
except ImportError as e:
    cloud_logger.error(f"导入关键模块失败: {e}")
    st.error(f"⚠️ 关键模块导入失败: {e}")
    st.error("请检查依赖文件是否完整")
    st.stop()

def load_default_image():
    """
    加载适合细胞组织分割系统的默认图片
    """
    try:
        # 优先使用细胞图像作为默认图片
        cell_image_path = abs_path("icon/cell_ini_image.jpg", path_type="current")
        if os.path.exists(cell_image_path):
            return Image.open(cell_image_path)
    except Exception as e:
        cloud_logger.warning(f"Failed to load cell_ini_image.jpg: {e}")
    
    try:
        # 备选方案：使用其他现有图片
        ini_image_path = abs_path("icon/ini-image.png", path_type="current")
        if os.path.exists(ini_image_path):
            return Image.open(ini_image_path)
    except Exception as e:
        cloud_logger.warning(f"Failed to load ini-image.png: {e}")
    
    try:
        # 如果都不可用，创建一个自定义的默认图片
        width, height = 600, 400
        
        # 创建一个深灰色背景
        img_array = np.ones((height, width, 3), dtype=np.uint8) * 45
        
        # 添加一个圆形区域模拟显微镜视场
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        # 在圆形区域内创建稍亮的背景
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        img_array[mask] = [65, 65, 65]
        
        # 添加圆形边界
        cv2.circle(img_array, (center_x, center_y), radius, (120, 120, 120), 2)
        
        # 添加十字线
        cv2.line(img_array, (center_x - 20, center_y), (center_x + 20, center_y), (100, 100, 100), 1)
        cv2.line(img_array, (center_x, center_y - 20), (center_x, center_y + 20), (100, 100, 100), 1)
        
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (150, 150, 150)
        thickness = 2
        
        # 主标题
        text1 = "AI Cell Tissue Segmentation"
        text_size1 = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text_x1 = (width - text_size1[0]) // 2
        text_y1 = center_y - 50
        cv2.putText(img_array, text1, (text_x1, text_y1), font, font_scale, color, thickness)
        
        # 副标题
        text2 = "Waiting for Microscope Image..."
        font_scale2 = 0.6
        text_size2 = cv2.getTextSize(text2, font, font_scale2, thickness)[0]
        text_x2 = (width - text_size2[0]) // 2
        text_y2 = center_y + 30
        cv2.putText(img_array, text2, (text_x2, text_y2), font, font_scale2, (120, 120, 120), thickness)
        
        # 底部信息
        text3 = "Upload microscope images for AI analysis"
        font_scale3 = 0.4
        text_size3 = cv2.getTextSize(text3, font, font_scale3, 1)[0]
        text_x3 = (width - text_size3[0]) // 2
        text_y3 = height - 30
        cv2.putText(img_array, text3, (text_x3, text_y3), font, font_scale3, (100, 100, 100), 1)
        
        # 转换为 PIL Image
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
        
    except Exception as e:
        cloud_logger.error(f"Error creating default image: {e}")
        # 最后的备选方案：创建一个简单的纯色图片
        simple_img = np.ones((400, 600, 3), dtype=np.uint8) * 128
        return Image.fromarray(simple_img)

def drawRectBox(image, bbox, alpha=0.2, addText='', color=(0, 255, 0), thickness=2):
    """
    自定义的 drawRectBox 函数
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 如果有文本要添加
        if addText:
            # 计算文本大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(addText, font, font_scale, text_thickness)
            
            # 绘制文本背景
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # 绘制文本
            cv2.putText(image, addText, (x1, y1 - 5), font, font_scale, (255, 255, 255), text_thickness)
        
        return image
    except Exception as e:
        cloud_logger.error(f"Error in drawRectBox: {e}")
        return image

def calculate_polygon_area(points):
    """计算多边形面积的函数"""
    return cv2.contourArea(points.astype(np.float32))

def generate_color_based_on_name(name):
    """使用哈希函数生成稳定的颜色"""
    hash_object = md5(name.encode())
    hex_color = hash_object.hexdigest()[:6]
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV 使用BGR格式

def adjust_parameter(image_size, base_size=1000):
    """计算自适应参数"""
    max_size = max(image_size)
    return max_size / base_size

def draw_detections(image, info, alpha=0.2):
    """绘制检测结果"""
    name, bbox, conf, cls_id, mask = info['class_name'], info['bbox'], info['score'], info['class_id'], info['mask']
    
    # 将中文标签转换为英文标签
    english_name = Chinese_to_English.get(name, name)
    
    adjust_param = adjust_parameter(image.shape[:2])
    
    if mask is None:
        x1, y1, x2, y2 = bbox
        aim_frame_area = (x2 - x1) * (y2 - y1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=int(5 * adjust_param))
        
        # 使用英文标签
        label_text = f"{english_name} {conf:.2f}"
        
        # 绘制黑色背景
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * adjust_param, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
        
        # 绘制白色文字
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * adjust_param, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        mask_points = np.concatenate(mask)
        aim_frame_area = calculate_polygon_area(mask_points)
        mask_color = generate_color_based_on_name(name)
        try:
            overlay = image.copy()
            cv2.fillPoly(overlay, [mask_points.astype(np.int32)], mask_color)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.drawContours(image, [mask_points.astype(np.int32)], -1, (0, 0, 255), thickness=int(8 * adjust_param))

            # 绘制类别名称
            x, y = np.min(mask_points, axis=0).astype(int)
            english_name = Chinese_to_English.get(name, name)
            label_text = f"{english_name} {conf:.2f}"
            
            # 绘制黑色背景
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * adjust_param, 1)
            cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
            
            # 绘制白色文字
            cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6 * adjust_param, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            cloud_logger.error(f"An error occurred in mask drawing: {e}")

    return image, aim_frame_area

def format_time(seconds):
    """格式化时间"""
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return "{:02}:{:02}:{:02}".format(int(hrs), int(mins), int(secs))

class Detection_UI_Cloud:
    """
    云端检测系统类
    """

    def __init__(self):
        """初始化云端检测系统"""
        cloud_logger.info("初始化云端检测系统")
        
        # 初始化类别标签列表
        self.cls_name = Label_list
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.cls_name))]

        # 设置页面标题
        self.title = "AI细胞组织分割系统 - Cell Tissue Segmentation System"
        self.setup_page()
        def_css_hitml()

        # 初始化检测相关的配置参数
        self.model_type = None
        self.conf_threshold = 0.15
        self.iou_threshold = 0.5

        # 禁用摄像头功能（云端不支持）
        self.selected_camera = "摄像头检测关闭"
        self.file_type = None
        self.uploaded_file = None
        self.uploaded_video = None
        self.custom_model_file = None

        # 初始化检测结果相关变量
        self.detection_result = None
        self.detection_location = None
        self.detection_confidence = None
        self.detection_time = None

        # 初始化UI显示相关变量
        self.display_mode = None
        self.close_flag = None
        self.close_placeholder = None
        self.image_placeholder = None
        self.image_placeholder_res = None
        self.table_placeholder = None
        self.progress_bar = None

        # 初始化日志数据保存路径
        self.saved_log_data = abs_path("tempDir/log_table_data.csv", path_type="current")

        # 创建LogTable实例
        if 'logTable' not in st.session_state:
            st.session_state['logTable'] = LogTable(self.saved_log_data)

        self.logTable = st.session_state['logTable']

        # 加载模型
        if 'model' not in st.session_state:
            st.session_state['model'] = Web_Detector()

        self.model = st.session_state['model']
        
        # 尝试加载模型权重
        self.load_model_weights()
        
        # 设置侧边栏
        self.setup_sidebar()

    def load_model_weights(self):
        """加载模型权重"""
        try:
            # 优先使用自定义训练模型
            default_model_path = abs_path("tempDir/best.pt", path_type="current")
            if os.path.exists(default_model_path):
                self.model.load_model(model_path=default_model_path)
                cloud_logger.info(f"成功加载模型: {default_model_path}")
                return
            
            # 备用模型
            backup_paths = [
                abs_path("weights/yolov8s.pt", path_type="current"),
                abs_path("yolo11s.pt", path_type="current"),
                abs_path("yolo11s-seg.pt", path_type="current")
            ]
            
            for backup_path in backup_paths:
                if os.path.exists(backup_path):
                    self.model.load_model(model_path=backup_path)
                    cloud_logger.info(f"成功加载备用模型: {backup_path}")
                    return
            
            # 如果没有本地模型，尝试下载默认模型
            cloud_logger.warning("未找到本地模型文件，将使用默认模型")
            
        except Exception as e:
            cloud_logger.error(f"模型加载失败: {e}")
            st.error("⚠️ 模型加载失败，某些功能可能不可用")

    def setup_page(self):
        """设置页面布局"""
        # 专业化的标题和介绍
        st.markdown(
            f"""
            <div style="text-align: center; background: linear-gradient(90deg, #2d5016 0%, #3e7b27 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <div style="color: #e8f5e8; margin-bottom: 10px; font-size: 0.9em; font-weight: bold;">
                    合溪生物科技 | Hexi Biotechnology Co., Ltd.
                </div>
                <h1 style="color: white; margin: 0; font-size: 2.5em;">🔬 {self.title}</h1>
                <p style="color: #e8f5e8; margin: 10px 0 0 0; font-size: 1.1em;">
                    基于深度学习的细胞组织智能分割与分析系统 (云端版)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 云端部署说明
        if is_cloud_environment():
            st.info("🌐 您正在使用云端版本。为了保护隐私和安全，摄像头功能已禁用，请上传图像文件进行分析。")
        
        # 科研免责声明
        col_disclaimer, col_instructions = st.columns([1, 1])
        
        with col_disclaimer:
            st.markdown(
                """
                <div style="background-color: #fff8e1; border: 1px solid #ffcc02; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
                    <h4 style="color: #ff6f00; margin-top: 0;">⚠️ 科研使用声明</h4>
                    <ul style="margin-bottom: 0; color: #ff6f00;">
                        <li>本系统仅供生物医学研究和教学使用</li>
                        <li>不可用于临床诊断或医疗决策</li>
                        <li>分析结果需要专业研究人员验证</li>
                        <li>细胞组织分割结果仅供科研参考</li>
                        <li>云端版本不存储用户数据</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_instructions:
            st.markdown(
                """
                <div style="background-color: #e8f5e8; border: 1px solid #66bb6a; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
                    <h4 style="color: #2e7d32; margin-top: 0;">🔬 分析说明</h4>
                    <ul style="margin-bottom: 0; color: #2e7d32;">
                        <li><strong>分析类型：</strong>细胞组织智能分割</li>
                        <li><strong>支持格式：</strong>JPG, PNG, JPEG, TIFF</li>
                        <li><strong>最佳图像：</strong>高分辨率显微镜图像</li>
                        <li><strong>分析指标：</strong>细胞边界、组织结构、形态特征</li>
                        <li><strong>云端优势：</strong>无需安装，随时访问</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    def setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.markdown("### 🔬 AI 分析参数配置")
        
        # 置信度阈值
        self.conf_threshold = float(st.sidebar.slider(
            "置信度阈值 (Confidence Threshold)", 
            min_value=0.0, max_value=1.0, value=0.3,
            help="较低的值会分割更多细胞区域，较高的值只分割明确的细胞结构"
        ))
        
        # IOU阈值
        self.iou_threshold = float(st.sidebar.slider(
            "重叠度阈值 (IoU Threshold)", 
            min_value=0.0, max_value=1.0, value=0.25,
            help="用于消除重复分割区域的阈值"
        ))
        
        # 模型配置
        st.sidebar.header("🤖 AI 模型配置")
        self.model_type = st.sidebar.selectbox(
            "分析模式", 
            ["检测任务 (Detection)", "分割任务 (Segmentation)"],
            help="检测模式：标记细胞位置；分割模式：精确描绘细胞边界"
        )

        # 模型文件选择（云端简化版）
        st.sidebar.markdown("**模型状态**")
        if hasattr(self, 'model') and self.model is not None:
            st.sidebar.success("✅ AI模型已就绪")
        else:
            st.sidebar.error("❌ AI模型未加载")

        st.sidebar.markdown("---")

        # 图像输入配置
        st.sidebar.header("🔬 显微镜图像输入")
        self.file_type = st.sidebar.selectbox("图像类型", ["细胞切片图像", "组织学视频"])
        
        if self.file_type == "细胞切片图像":
            self.uploaded_file = st.sidebar.file_uploader(
                "上传显微镜图像", 
                type=["jpg", "png", "jpeg", "tiff", "tif"],
                help="支持 JPEG、PNG、TIFF 格式的显微镜图像"
            )
        elif self.file_type == "组织学视频":
            self.uploaded_video = st.sidebar.file_uploader(
                "上传显微镜视频", 
                type=["mp4", "avi", "mov"],
                help="支持 MP4、AVI、MOV 格式的显微镜视频"
            )

        # 操作指南
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 操作指南")
        if self.file_type == "细胞切片图像":
            st.sidebar.info("🔬 请上传显微镜图像，然后点击'开始分析'按钮进行AI细胞分割")
        if self.file_type == "组织学视频":
            st.sidebar.info("🎥 请上传显微镜视频，然后点击'开始分析'按钮进行批量分析")

    def process_uploaded_file(self):
        """处理上传的文件"""
        if self.uploaded_file is not None:
            cloud_logger.info(f"开始处理上传的图像: {self.uploaded_file.name}")
            
            self.logTable.clear_frames()
            self.progress_bar.progress(0)
            
            # 读取上传的图像
            source_img = self.uploaded_file.read()
            file_bytes = np.asarray(bytearray(source_img), dtype=np.uint8)
            image_ini = cv2.imdecode(file_bytes, 1)
            framecopy = image_ini.copy()
            
            # 处理图像
            image, detInfo, select_info = self.frame_process(image_ini, self.uploaded_file.name)
            
            # 保存结果
            self.logTable.save_to_csv()

            # 显示结果
            new_width = 1080
            new_height = int(new_width * (9 / 16))
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_frame = cv2.resize(framecopy, (new_width, new_height))
            
            if self.display_mode == "智能叠加显示":
                self.image_placeholder.image(resized_image, channels="BGR", caption="🔬 显微镜图像AI分析结果")
            else:
                self.image_placeholder.image(resized_frame, channels="BGR", caption="🔬 原始显微镜图像")
                self.image_placeholder_res.image(resized_image, channels="BGR", caption="🤖 AI分割标注结果")

            self.logTable.add_frames(image, detInfo, cv2.resize(image_ini, (640, 640)))
            self.progress_bar.progress(100)
            
            # 更新分析评估
            self.update_analysis_assessment()
            
            cloud_logger.info("图像处理完成")

    def frame_process(self, image, file_name, video_time=None):
        """处理并预测单个图像帧"""
        try:
            pre_img = self.model.preprocess(image)
            
            # 更新模型参数
            params = {'conf': self.conf_threshold, 'iou': self.iou_threshold}
            self.model.set_param(params)

            t1 = time.time()
            pred = self.model.predict(pre_img)
            t2 = time.time()
            use_time = t2 - t1

            det = pred[0]
            detInfo = []
            select_info = ["全部目标"]

            if det is not None and len(det):
                det_info = self.model.postprocess(pred)
                if len(det_info):
                    disp_res = ResultLogger()
                    res = None
                    cnt = 0

                    for info in det_info:
                        name, bbox, conf, cls_id, mask = info['class_name'], info['bbox'], info['score'], info['class_id'], info['mask']

                        # 绘制检测结果
                        image, aim_frame_area = draw_detections(image, info, alpha=0.5)

                        # 生成生物学描述
                        biological_description = self.get_biological_description(name, int(aim_frame_area))
                        
                        res = disp_res.concat_results(name, bbox, biological_description,
                                                      video_time if video_time is not None else str(round(use_time, 2)))

                        # 添加日志条目
                        self.logTable.add_log_entry(file_name, name, bbox, biological_description, 
                                                   video_time if video_time is not None else str(round(use_time, 2)))
                        
                        detInfo.append([name, bbox, biological_description, 
                                      video_time if video_time is not None else str(round(use_time, 2)), cls_id])
                        select_info.append(name + "-" + str(cnt))
                        cnt += 1

                    # 显示结果
                    self.table_placeholder.table(res)
                    self.update_analysis_assessment()

            return image, detInfo, select_info
            
        except Exception as e:
            cloud_logger.error(f"图像处理错误: {e}")
            st.error(f"图像处理失败: {e}")
            return image, [], ["全部目标"]

    def get_biological_description(self, class_name, area):
        """根据检测类别和细胞面积生成专业的生物学描述"""
        descriptions = {
            "细胞核": {
                "small": f"检测到小细胞核 (面积: {area}px²) - 细胞分裂期或幼稚细胞",
                "medium": f"检测到正常细胞核 (面积: {area}px²) - 成熟细胞核形态",
                "large": f"检测到大细胞核 (面积: {area}px²) - 可能为活跃增殖细胞"
            },
            "细胞质": {
                "small": f"检测到少量细胞质 (面积: {area}px²) - 高核质比细胞",
                "medium": f"检测到适量细胞质 (面积: {area}px²) - 正常核质比例",
                "large": f"检测到丰富细胞质 (面积: {area}px²) - 分泌活跃或成熟细胞"
            },
            "组织结构": {
                "small": f"检测到局部组织结构 (面积: {area}px²) - 组织局部特征",
                "medium": f"检测到典型组织结构 (面积: {area}px²) - 正常组织形态",
                "large": f"检测到完整组织结构 (面积: {area}px²) - 组织结构完整"
            },
            "血管": {
                "small": f"检测到毛细血管 (面积: {area}px²) - 微血管结构",
                "medium": f"检测到小血管 (面积: {area}px²) - 组织供血血管",
                "large": f"检测到主要血管 (面积: {area}px²) - 大血管或动脉"
            }
        }
        
        # 根据面积大小分类
        if area < 2000:
            size_category = "small"
        elif area < 8000:
            size_category = "medium"
        else:
            size_category = "large"
            
        # 获取对应的生物学描述
        if class_name in descriptions:
            return descriptions[class_name][size_category]
        else:
            return f"检测到 {class_name} (面积: {area}px²) - 需要进一步分析"

    def update_analysis_assessment(self):
        """更新分析评估显示"""
        if not hasattr(self, 'analysis_assessment_placeholder'):
            return
            
        if not hasattr(self, 'logTable') or len(self.logTable.saved_results) == 0:
            self.analysis_assessment_placeholder.info("📊 暂无分析数据进行评估")
            return
        
        # 统计分析结果
        analysis_stats = {
            "细胞核": 0,
            "细胞质": 0,
            "组织结构": 0,
            "血管": 0,
            "其他结构": 0,
            "总分析数": len(self.logTable.saved_results)
        }
        
        for result in self.logTable.saved_results:
            if len(result) >= 1:
                class_name = result[0] if len(result) > 0 else "未知"
                if "细胞核" in str(class_name):
                    analysis_stats["细胞核"] += 1
                elif "细胞质" in str(class_name):
                    analysis_stats["细胞质"] += 1
                elif "组织结构" in str(class_name) or "组织" in str(class_name):
                    analysis_stats["组织结构"] += 1
                elif "血管" in str(class_name):
                    analysis_stats["血管"] += 1
                else:
                    analysis_stats["其他结构"] += 1
        
        # 计算分析质量等级
        total_structures = analysis_stats["细胞核"] + analysis_stats["细胞质"] + analysis_stats["组织结构"] + analysis_stats["血管"]
        
        if total_structures >= 10:
            quality_level = "🟢 高质量"
            quality_color = "#2ed573"
        elif total_structures >= 5:
            quality_level = "🟡 中等质量"
            quality_color = "#ffa726"
        else:
            quality_level = "🔴 需要改进"
            quality_color = "#ff4757"
        
        # 显示评估结果
        with self.analysis_assessment_placeholder.container():
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border-left: 4px solid {quality_color}; padding: 15px; border-radius: 5px;">
                    <h5 style="color: {quality_color}; margin-top: 0;">分析质量：{quality_level}</h5>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap;">
                        <span><strong>细胞核：</strong>{analysis_stats['细胞核']}个</span>
                        <span><strong>细胞质：</strong>{analysis_stats['细胞质']}个</span>
                        <span><strong>组织：</strong>{analysis_stats['组织结构']}个</span>
                        <span><strong>血管：</strong>{analysis_stats['血管']}个</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    def setupMainWindow(self):
        """运行细胞组织分割系统主界面"""
        # 分隔线
        st.markdown(
            """
            <div style="text-align: center; color: #666; margin: 20px 0;">
                <hr style="border: 1px solid #e0e0e0;">
                <p style="margin: 10px 0; font-size: 0.9em;">
                    🧬 AI-Powered Cell & Tissue Segmentation Platform | 基于人工智能的细胞组织分割分析平台 (云端版)
                </p>
                <hr style="border: 1px solid #e0e0e0;">
            </div>
            """,
            unsafe_allow_html=True
        )

        # 创建列布局
        col1, col2, col3 = st.columns([4, 1, 2])

        # 显示模式选择
        with col1:
            st.markdown("### 🔬 图像显示模式")
            self.display_mode = st.radio(
                "选择显示方式", 
                ["智能叠加显示", "对比分析显示"],
                help="叠加显示：在原图上直接标注分割结果；对比显示：原图与分割结果分别显示"
            )
            
            # 根据显示模式创建显示容器
            if self.display_mode == "智能叠加显示":
                self.image_placeholder = st.empty()
                if not hasattr(self.logTable, 'saved_images_ini') or not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="🔬 等待显微镜图像输入...")
            else:
                st.markdown("**原始图像 vs AI分割结果**")
                self.image_placeholder = st.empty()
                self.image_placeholder_res = st.empty()
                if not hasattr(self.logTable, 'saved_images_ini') or not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="🔬 原始显微镜图像")
                    self.image_placeholder_res.image(load_default_image(), caption="🤖 AI分割结果")
            
            # 进度条
            st.markdown("**🔄 分析进度**")
            self.progress_bar = st.progress(0)

        # 结果显示
        with col3:
            st.markdown("### 🔬 AI分析报告")
            self.table_placeholder = st.empty()
            res = concat_results("等待分析", "待分割区域", "0.00", "0.00s")
            self.table_placeholder.table(res)

            # 分析质量评估
            st.markdown("---")
            st.markdown("**📊 分析质量评估**")
            self.analysis_assessment_placeholder = st.empty()
            self.update_analysis_assessment()

        # 控制面板
        with col2:
            st.markdown("### 🎮 控制面板")
            
            # 主要控制按钮
            st.markdown("**主控制**")
            if st.button("🔬 开始AI分析", help="启动AI细胞组织分割分析", type="primary"):
                if self.uploaded_file is not None:
                    self.process_uploaded_file()
                else:
                    st.warning("⚠️ 请先上传图像文件")
            
            # 系统状态
            st.markdown("---")
            st.markdown("**📈 系统状态**")
            if hasattr(self, 'model') and self.model is not None:
                st.success("🟢 AI模型就绪")
            else:
                st.error("🔴 AI模型未加载")
                
            # 统计信息
            if hasattr(self, 'logTable') and hasattr(self.logTable, 'saved_results') and len(self.logTable.saved_results) > 0:
                total_analyses = len(self.logTable.saved_results)
                st.metric("总分析数", total_analyses)
            else:
                st.metric("总分析数", 0)

        # 版权信息
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;">
                <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
                    © 2025 <strong style="color: #2c3e50;">合溪生物科技</strong> | 
                    Powered by Hexi Biotechnology Co., Ltd.
                </p>
                <p style="margin: 5px 0 0 0; color: #adb5bd; font-size: 0.8em;">
                    专业医学AI影像分析解决方案提供商 | 云端智能分析平台
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
