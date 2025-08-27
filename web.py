import random
import tempfile
import time
import os
import cv2
import numpy as np
import streamlit as st
try:
    from QtFusion.path import abs_path
except ImportError:
    # 云端环境替代
    from cloud_utils import abs_path
# from QtFusion.utils import drawRectBox  # 注释掉有问题的导入

from log import ResultLogger, LogTable
from model import Web_Detector
from chinese_name_list import Label_list, Chinese_to_English
from ui_style import def_css_hitml
from utils import save_uploaded_file, concat_results, get_camera_names
import tempfile
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime

import numpy as np
import cv2
from hashlib import md5

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
        print(f"Failed to load cell_ini_image.jpg: {e}")
    
    try:
        # 备选方案：使用其他现有图片
        ini_image_path = abs_path("icon/ini-image.png", path_type="current")
        if os.path.exists(ini_image_path):
            return Image.open(ini_image_path)
    except Exception as e:
        print(f"Failed to load ini-image.png: {e}")
    
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
        print(f"Error creating default image: {e}")
        # 最后的备选方案：创建一个简单的纯色图片
        simple_img = np.ones((400, 600, 3), dtype=np.uint8) * 128
        return Image.fromarray(simple_img)

def drawRectBox(image, bbox, alpha=0.2, addText='', color=(0, 255, 0), thickness=2):
    """
    自定义的 drawRectBox 函数，用于替代 QtFusion.utils 中的版本
    
    Args:
        image: 输入图像
        bbox: 边界框坐标 [x1, y1, x2, y2]
        alpha: 透明度
        addText: 要添加的文本
        color: 颜色 (B, G, R)
        thickness: 线条粗细
    
    Returns:
        处理后的图像
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
        print(f"Error in drawRectBox: {e}")
        return image

def calculate_polygon_area(points):
    # 计算多边形面积的函数
    return cv2.contourArea(points.astype(np.float32))

def draw_with_chinese(img, text, position, font_size):
    # 假设这是一个自定义函数，用于在图像上绘制中文文本
    # 具体实现需要根据你的需求进行调整
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    thickness = 2
    cv2.putText(img, text, position, font, font_size, color, thickness, cv2.LINE_AA)
    return img

def generate_color_based_on_name(name):
    # 使用哈希函数生成稳定的颜色
    hash_object = md5(name.encode())
    hex_color = hash_object.hexdigest()[:6]  # 取前6位16进制数
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV 使用BGR格式

def draw_with_chinese(image, text, position, font_size=20, color=(255, 0, 0)):
    """
    在OpenCV图像上绘制中文文字
    """
    # 将图像从 OpenCV 格式（BGR）转换为 PIL 格式（RGB）
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # 尝试使用不同的字体，优先使用支持中文的字体
    font = None
    font_paths = [
        # 项目中的Arial字体（如果存在）
        "./Arial.ttf",
        # macOS 系统中文字体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/PingFang.ttc", 
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        # 常见的中文字体路径
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        # Helvetica 作为备选
        "/System/Library/Fonts/Helvetica.ttc"
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            # 测试字体是否能正确显示中文
            test_bbox = draw.textbbox((0, 0), "测试", font=font)
            if test_bbox[2] > test_bbox[0]:  # 如果能正确计算文本宽度，说明字体支持中文
                break
        except (OSError, IOError, AttributeError):
            continue
    
    # 如果所有字体都失败了，使用默认字体
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            # 最后的备选方案：使用 OpenCV 直接绘制
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            cv2.putText(image_cv, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size/30, color, 2, cv2.LINE_AA)
            return image_cv
    
    # 转换颜色格式（PIL使用RGB，传入的color是BGR）
    if len(color) == 3:
        pil_color = (color[2], color[1], color[0])  # BGR to RGB
    else:
        pil_color = color
    
    draw.text(position, text, font=font, fill=pil_color)
    # 将图像从 PIL 格式（RGB）转换回 OpenCV 格式（BGR）
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def adjust_parameter(image_size, base_size=1000):
    # 计算自适应参数，基于图片的最大尺寸
    max_size = max(image_size)
    return max_size / base_size

def adjust_parameter(image_size, base_size=1000):
    max_size = max(image_size)
    return max_size / base_size


def draw_detections(image, info, alpha=0.2):
    name, bbox, conf, cls_id, mask = info['class_name'], info['bbox'], info['score'], info['class_id'], info['mask']
    
    # 将中文标签转换为英文标签
    english_name = Chinese_to_English.get(name, name)
    
    adjust_param = adjust_parameter(image.shape[:2])
    spacing = int(20 * adjust_param)

    if mask is None:
        x1, y1, x2, y2 = bbox
        aim_frame_area = (x2 - x1) * (y2 - y1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=int(5 * adjust_param))
        
        # 使用英文标签和改进的文字绘制
        label_text = f"{english_name} {conf:.2f}"
        
        # 绘制黑色背景
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * adjust_param, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
        
        # 绘制白色文字（减少粗细）
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6 * adjust_param, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset = int(50 * adjust_param)  # 类别名称上方绘制，其下方留出空间
    else:
        mask_points = np.concatenate(mask)
        aim_frame_area = calculate_polygon_area(mask_points)
        mask_color = generate_color_based_on_name(name)
        try:
            overlay = image.copy()
            cv2.fillPoly(overlay, [mask_points.astype(np.int32)], mask_color)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.drawContours(image, [mask_points.astype(np.int32)], -1, (0, 0, 255), thickness=int(8 * adjust_param))

            # 计算面积、周长、圆度
            area = cv2.contourArea(mask_points.astype(np.int32))
            perimeter = cv2.arcLength(mask_points.astype(np.int32), True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # 计算色彩
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [mask_points.astype(np.int32)], -1, 255, -1)
            color_points = cv2.findNonZero(mask)
            selected_points = color_points[np.random.choice(color_points.shape[0], 5, replace=False)]
            colors = np.mean([image[y, x] for x, y in selected_points[:, 0]], axis=0)
            color_str = f"({colors[0]:.1f}, {colors[1]:.1f}, {colors[2]:.1f})"

            # 绘制类别名称（英文，改进显示）
            x, y = np.min(mask_points, axis=0).astype(int)
            english_name = Chinese_to_English.get(name, name)
            label_text = f"{english_name} {conf:.2f}"
            
            # 绘制黑色背景
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * adjust_param, 1)
            cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
            
            # 绘制白色文字（减少粗细）
            cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6 * adjust_param, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset = int(50 * adjust_param)  # 类别名称上方绘制，其下方留出空间

            # 绘制面积、周长、圆度和色彩值
            # metrics = [("Area", area), ("Perimeter", perimeter), ("Circularity", circularity), ("Color", color_str)]
            # for idx, (metric_name, metric_value) in enumerate(metrics):
            #     text = f"{metric_name}: {metric_value}"
            #     image = draw_with_chinese(image, text, (x, y - y_offset - spacing * (idx + 1)),
            #                               font_size=int(35 * adjust_param))

        except Exception as e:
            print(f"An error occurred: {e}")

    return image, aim_frame_area

def calculate_polygon_area(points):
    """
    计算多边形的面积，输入应为一个 Nx2 的numpy数组，表示多边形的顶点坐标
    """
    if len(points) < 3:  # 多边形至少需要3个顶点
        return 0
    return cv2.contourArea(points)

def format_time(seconds):
    # 计算小时、分钟和秒
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    # 格式化为字符串
    return "{:02}:{:02}:{:02}".format(int(hrs), int(mins), int(secs))



def save_chinese_image(file_path, image_array):
    """
    保存带有中文路径的图片文件

    参数：
    file_path (str): 图片的保存路径，应包含中文字符, 例如 '示例路径/含有中文的文件名.png'
    image_array (numpy.ndarray): 要保存的 OpenCV 图像（即 numpy 数组）
    """
    try:
        # 将 OpenCV 图片转换为 Pillow Image 对象
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

        # 使用 Pillow 保存图片文件
        image.save(file_path)

        print(f"成功保存图像到: {file_path}")
    except Exception as e:
        print(f"保存图像失败: {str(e)}")

class Detection_UI:
    """
    检测系统类。

    Attributes:
        model_type (str): 模型类型。
        conf_threshold (float): 置信度阈值。
        iou_threshold (float): IOU阈值。
        selected_camera (str): 选定的摄像头。
        file_type (str): 文件类型。
        uploaded_file (FileUploader): 上传的文件。
        detection_result (str): 检测结果。
        detection_location (str): 检测位置。
        detection_confidence (str): 检测置信度。
        detection_time (str): 检测用时。
    """

    def __init__(self):
        """
        初始化行人跌倒检测系统的参数。
        """
        # 初始化类别标签列表和为每个类别随机分配颜色
        self.cls_name = Label_list
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.cls_name))]

        # 设置页面标题
        self.title = "AI血细胞分析系统 - Blood Cell Analysis System"
        self.setup_page()  # 初始化页面布局
        def_css_hitml()  # 应用 CSS 样式

        # 初始化检测相关的配置参数
        self.model_type = None
        self.conf_threshold = 0.15  # 默认置信度阈值
        self.iou_threshold = 0.5  # 默认IOU阈值

        # 初始化相机和文件相关的变量
        self.selected_camera = None
        self.file_type = None
        self.uploaded_file = None
        self.uploaded_video = None
        self.custom_model_file = None  # 自定义的模型文件

        # 初始化检测结果相关的变量
        self.detection_result = None
        self.detection_location = None
        self.detection_confidence = None
        self.detection_time = None

        # 初始化UI显示相关的变量
        self.display_mode = None  # 设置显示模式
        self.close_flag = None  # 控制图像显示结束的标志
        self.close_placeholder = None  # 关闭按钮区域
        self.image_placeholder = None  # 用于显示图像的区域
        self.image_placeholder_res = None  # 图像显示区域
        self.table_placeholder = None  # 表格显示区域
        self.selectbox_placeholder = None  # 下拉框显示区域
        self.selectbox_target = None  # 下拉框选中项
        self.progress_bar = None  # 用于显示的进度条

        # 初始化FPS和视频时间指针
        self.FPS = 30
        self.timenow = 0

        # 初始化日志数据保存路径
        self.saved_log_data = abs_path("tempDir/log_table_data.csv", path_type="current")

        # 如果在 session state 中不存在logTable，创建一个新的LogTable实例
        if 'logTable' not in st.session_state:
            st.session_state['logTable'] = LogTable(self.saved_log_data)

        # 获取或更新可用摄像头列表
        if 'available_cameras' not in st.session_state:
            st.session_state['available_cameras'] = get_camera_names()
        self.available_cameras = st.session_state['available_cameras']

        # 初始化或获取识别结果的表格
        self.logTable = st.session_state['logTable']

        # 加载或创建模型实例
        if 'model' not in st.session_state:
            st.session_state['model'] = Web_Detector()  # 创建Detector模型实例

        self.model = st.session_state['model']
        
        # 加载训练的模型权重（默认使用 tempDir/best.pt）
        default_model_path = abs_path("tempDir/best.pt", path_type="current")
        
        if os.path.exists(default_model_path):
            try:
                self.model.load_model(model_path=default_model_path)
            except Exception as e:
                # 静默尝试备用模型
                self._load_backup_model()
        else:
            # 静默尝试备用模型
            self._load_backup_model()
        
        # 为模型中的类别重新分配颜色
        if hasattr(self.model, 'names') and self.model.names:
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                           range(len(self.model.names))]
        else:
            # 使用默认的类别数量
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                           range(6)]  # 默认6个类别
        
        self.setup_sidebar()  # 初始化侧边栏布局
    
    def _load_backup_model(self):
        """静默加载备用模型"""
        backup_paths = [
            abs_path("weights/yolov8s-seg.pt", path_type="current"),
            abs_path("weights/yolov8s.pt", path_type="current"),
            abs_path("yolo11s-seg.pt", path_type="current"),
            abs_path("yolo11s.pt", path_type="current")
        ]
        
        for backup_path in backup_paths:
            if os.path.exists(backup_path):
                try:
                    self.model.load_model(model_path=backup_path)
                    return
                except Exception as e:
                    continue
        
        # 只有在所有模型都加载失败时才显示错误
        st.error("⚠️ 找不到任何可用的模型文件！")

    def setup_page(self):
        # 设置页面布局
        # st.set_page_config(
        #     page_title=self.title,
        #     page_icon="🫁",
        #     initial_sidebar_state="expanded"
        # )

        # 专业化的标题和介绍
        st.markdown(
            f"""
            <div style="text-align: center; background: linear-gradient(90deg, #2d5016 0%, #3e7b27 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <div style="color: #e8f5e8; margin-bottom: 10px; font-size: 0.9em; font-weight: bold;">
                    合溪生物科技 | Hexi Biotechnology Co., Ltd.
                </div>
                <h1 style="color: white; margin: 0; font-size: 2.5em;">🔬 {self.title}</h1>
                <p style="color: #e8f5e8; margin: 10px 0 0 0; font-size: 1.1em;">
                    基于深度学习的细胞组织智能分割与分析系统
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 科研免责声明和分析说明
        col_disclaimer, col_instructions = st.columns([1, 1])
        
        with col_disclaimer:
            st.markdown(
                """
                <div style="background-color: #fff8e1; border: 1px solid #ffcc02; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
                    <h4 style="color: #ff6f00; margin-top: 0;">⚠️ 系统使用声明</h4>
                    <ul style="margin-bottom: 0; color: #ff6f00;">
                        <li>本系统仅供生物医学研究和教学使用</li>
                        <li>仅为辅助用于临床诊断或医疗决策</li>
                        <li>分析结果需要专业研究人员验证</li>
                        <li>细胞组织分割结果仅供科研参考</li>
                        <li>使用前请确保数据合规性</li>
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
                        <li><strong>分析指标：</strong>细胞名称、细胞面积、细胞周长、细胞圆度、细胞色彩值</li>
                        <li><strong>置信度：</strong>建议设置0.3-0.7之间</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    def setup_sidebar(self):
        """
        设置 Streamlit 侧边栏。

        在侧边栏中配置AI模型参数、分析模式以及显微镜图像输入等选项。
        """
        # 添加侧边栏标题
        st.sidebar.markdown("### 🔬 AI 分析参数配置")
        
        # 置信度阈值的滑动条
        st.sidebar.markdown("**分析敏感度设置**")
        self.conf_threshold = float(st.sidebar.slider(
            "置信度阈值 (Confidence Threshold)", 
            min_value=0.0, max_value=1.0, value=0.3,
            help="较低的值会分割更多细胞区域，较高的值只分割明确的细胞结构"
        ))
        
        # IOU阈值的滑动条
        self.iou_threshold = float(st.sidebar.slider(
            "重叠度阈值 (IoU Threshold)", 
            min_value=0.0, max_value=1.0, value=0.25,
            help="用于消除重复分割区域的阈值"
        ))
        
        # 设置侧边栏的模型设置部分
        st.sidebar.header("🤖 AI 模型配置")
        # 选择模型类型的下拉菜单
        self.model_type = st.sidebar.selectbox(
            "分析模式", 
            ["检测任务 (Detection)", "分割任务 (Segmentation)"],
            help="检测模式：标记细胞位置；分割模式：精确描绘细胞边界"
        )


        # 选择模型文件类型，可以是默认的或者自定义的
        model_file_option = st.sidebar.radio("模型配置", ["使用预训练模型", "自定义模型权重"])
        if model_file_option == "自定义模型权重":
            # 如果选择自定义模型文件，则提供文件上传器
            model_file = st.sidebar.file_uploader("上传训练好的.pt模型文件", type="pt")

            # 如果上传了模型文件，则保存并加载该模型
            if model_file is not None:
                self.custom_model_file = save_uploaded_file(model_file)
                self.model.load_model(model_path=self.custom_model_file)
                self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                               range(len(self.model.names))]
        elif model_file_option == "使用预训练模型":
            # 始终使用 tempDir/best.pt 作为默认模型
            default_model_path = abs_path("tempDir/best.pt", path_type="current")
            if os.path.exists(default_model_path):
                self.model.load_model(model_path=default_model_path)
                st.sidebar.success("✅ 已加载自定义训练模型：tempDir/best.pt")
            else:
                # 备用方案
                if self.model_type == "检测任务 (Detection)":
                    backup_path = abs_path("./yolo11s.pt", path_type="current")
                elif self.model_type == "分割任务 (Segmentation)":
                    backup_path = abs_path("./yolo11s-seg.pt", path_type="current")
                
                if os.path.exists(backup_path):
                    self.model.load_model(model_path=backup_path)
                    st.sidebar.warning("⚠️ best.pt未找到，使用备用模型")
                else:
                    st.sidebar.error("❌ 找不到任何可用的模型文件")
            
            # 为模型中的类别重新分配颜色
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                           range(len(self.model.names))]

        st.sidebar.markdown("---")

        # 设置侧边栏的摄像头配置部分
        st.sidebar.header("📹 实时分析设置")
        # 选择摄像头的下拉菜单
        self.selected_camera = st.sidebar.selectbox("显微镜设备选择", self.available_cameras)

        # 设置侧边栏的识别项目设置部分
        st.sidebar.header("🔬 显微镜图像输入")
        # 选择文件类型的下拉菜单
        self.file_type = st.sidebar.selectbox("图像类型", ["细胞切片图像", "组织学视频"])
        # 根据所选的文件类型，提供对应的文件上传器
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

        # 提供相关提示信息，根据所选摄像头和文件类型的不同情况
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 操作指南")
        if self.selected_camera == "摄像头检测关闭":
            if self.file_type == "细胞切片图像":
                st.sidebar.info("🔬 请上传显微镜图像，然后点击'开始分析'按钮进行AI细胞分割")
            if self.file_type == "组织学视频":
                st.sidebar.info("🎥 请上传显微镜视频，然后点击'开始分析'按钮进行批量分析")
        else:
            st.sidebar.info("📷 请点击'开始实时分析'按钮，启动显微镜实时分析模式")

    def load_model_file(self):
        if self.custom_model_file:
            self.model.load_model(self.custom_model_file)
        else:
            pass  # 载入

    def process_camera_or_file(self):
        """
        处理摄像头或文件输入。

        根据用户选择的输入源（摄像头、图片文件或视频文件），处理并显示检测结果。
        """
        # 如果选择了摄像头输入
        if self.selected_camera != "摄像头检测关闭":
            self.logTable.clear_frames()  # 清除之前的帧记录
            # 创建一个结束按钮
            self.close_flag = self.close_placeholder.button(label="停止")

            # 使用 OpenCV 捕获摄像头画面
            if str(self.selected_camera) == '0':
                camera_id = 0
            else:
                camera_id = self.selected_camera

            cap = cv2.VideoCapture(camera_id)

            self.uploaded_video = None

            fps = cap.get(cv2.CAP_PROP_FPS)

            self.FPS = fps

            # 设置总帧数为1000
            total_frames = 1000
            current_frame = 0
            self.progress_bar.progress(0)  # 初始化进度条

            try:
                if len(self.selected_camera) < 8:
                    camera_id = int(self.selected_camera)
                else:
                    camera_id = self.selected_camera

                cap = cv2.VideoCapture(camera_id)

                # 获取和帧率
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.FPS = fps

                # 创建进度条
                self.progress_bar.progress(0)

                # 创建保存文件的信息
                camera_savepath = './tempDir/camera'
                if not os.path.exists(camera_savepath):
                    os.makedirs(camera_savepath)
                # ret, frame = cap.read()
                # height, width, layers = frame.shape
                # size = (width, height)
                #
                # file_name = abs_path('tempDir/camera.avi', path_type="current")
                # out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

                while cap.isOpened() and not self.close_flag:
                    ret, frame = cap.read()
                    if ret:
                        # 调节摄像头的分辨率
                        # 设置新的尺寸
                        new_width = 1080
                        new_height = int(new_width * (9 / 16))
                        # 调整图像尺寸
                        frame = cv2.resize(frame, (new_width, new_height))


                        framecopy = frame.copy()
                        image, detInfo, _ = self.frame_process(frame, 'camera')

                        # 保存目标结果图片
                        if detInfo:
                            file_name = abs_path(camera_savepath + '/' + str(current_frame + 1) + '.jpg', path_type="current")
                            save_chinese_image(file_name, image)
                        #
                        # # 保存目标结果视频
                        # out.write(image)

                        # 设置新的尺寸
                        new_width = 1080
                        new_height = int(new_width * (9 / 16))
                        # 调整图像尺寸
                        resized_image = cv2.resize(image, (new_width, new_height))
                        resized_frame = cv2.resize(framecopy, (new_width, new_height))
                        if self.display_mode == "智能叠加显示":
                            self.image_placeholder.image(resized_image, channels="BGR", caption="🎥 实时显微镜图像分析")
                        else:
                            self.image_placeholder.image(resized_frame, channels="BGR", caption="📷 原始显微镜图像")
                            self.image_placeholder_res.image(resized_image, channels="BGR", caption="🤖 AI分割结果")

                        self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                        # 更新进度条
                        progress_percentage = int((current_frame / total_frames) * 100)
                        self.progress_bar.progress(progress_percentage)
                        current_frame = (current_frame + 1) % total_frames  # 重置进度条
                    else:
                        break
                if self.close_flag:
                    self.logTable.save_to_csv()
                    # 摄像头分析停止后更新分析评估
                    self.update_analysis_assessment()
                    cap.release()
                    # out.release()

                self.logTable.save_to_csv()
                # 摄像头分析完成后更新分析评估
                self.update_analysis_assessment()
                cap.release()
                # out.release()


            finally:

                if self.uploaded_video is None:
                    name_in = None
                else:
                    name_in = self.uploaded_video.name

                res = self.logTable.save_frames_file(fps=self.FPS, video_name=name_in)
                st.write("识别结果文件已经保存：" + self.saved_log_data)
                if res:
                    st.write(f"结果的目标文件已经保存：{res}")


        else:
            # 如果上传了图片文件
            if self.uploaded_file is not None:
                self.logTable.clear_frames()
                self.progress_bar.progress(0)
                # 显示上传的图片
                source_img = self.uploaded_file.read()
                file_bytes = np.asarray(bytearray(source_img), dtype=np.uint8)
                image_ini = cv2.imdecode(file_bytes, 1)
                framecopy = image_ini.copy()
                image, detInfo, select_info = self.frame_process(image_ini, self.uploaded_file.name)
                save_chinese_image('./tempDir/' + self.uploaded_file.name, image)
                # self.selectbox_placeholder = st.empty()
                # self.selectbox_target = self.selectbox_placeholder.selectbox("目标过滤", select_info, key="22113")

                self.logTable.save_to_csv()

                # 设置新的尺寸
                new_width = 1080
                new_height = int(new_width * (9 / 16))
                # 调整图像尺寸
                resized_image = cv2.resize(image, (new_width, new_height))
                resized_frame = cv2.resize(framecopy, (new_width, new_height))
                if self.display_mode == "智能叠加显示":
                    self.image_placeholder.image(resized_image, channels="BGR", caption="🔬 显微镜图像AI分析结果")
                else:
                    self.image_placeholder.image(resized_frame, channels="BGR", caption="🔬 原始显微镜图像")
                    self.image_placeholder_res.image(resized_image, channels="BGR", caption="🤖 AI分割标注结果")

                self.logTable.add_frames(image, detInfo, cv2.resize(image_ini, (640, 640)))
                self.progress_bar.progress(100)
                
                # 处理完图片后更新分析评估
                self.update_analysis_assessment()

            # 如果上传了视频文件
            elif self.uploaded_video is not None:
                # 处理上传的视频
                self.logTable.clear_frames()
                self.close_flag = self.close_placeholder.button(label="停止")

                video_file = self.uploaded_video
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                try:
                    tfile.write(video_file.read())
                    tfile.flush()

                    tfile.seek(0)  # 确保文件指针回到文件开头

                    cap = cv2.VideoCapture(tfile.name)

                    # 获取视频总帧数和帧率
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    self.FPS = fps
                    # 计算视频总长度（秒）
                    total_length = total_frames / fps if fps > 0 else 0
                    print('视频时长：' + str(total_length)[:4] + 's')
                    # 创建进度条
                    self.progress_bar.progress(0)

                    current_frame = 0

                    # 创建保存文件的信息
                    video_savepath = './tempDir/' + self.uploaded_video.name
                    if not os.path.exists(video_savepath):
                        os.makedirs(video_savepath)
                    # ret, frame = cap.read()
                    # height, width, layers = frame.shape
                    # size = (width, height)
                    # file_name = abs_path('tempDir/' + self.uploaded_video.name + '.avi', path_type="current")
                    # out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

                    while cap.isOpened() and not self.close_flag:
                        ret, frame = cap.read()
                        if ret:
                            framecopy = frame.copy()
                            # 计算当前帧对应的时间（秒）
                            current_time = current_frame / fps
                            if current_time < total_length:
                                current_frame += 1
                                current_time_str = format_time(current_time)
                                image, detInfo, _ = self.frame_process(frame, self.uploaded_video.name,video_time=current_time_str)
                                # 保存目标结果图片
                                if detInfo:
                                    # 将字符串转换为 datetime 对象
                                    time_obj = datetime.strptime(current_time_str, "%H:%M:%S")

                                    # 将 datetime 对象格式化为所需的字符串格式
                                    formatted_time = time_obj.strftime("%H_%M_%S")
                                    file_name = abs_path(video_savepath + '/' + formatted_time  + '_' + str(current_frame) + '.jpg',
                                                         path_type="current")
                                    save_chinese_image(file_name, image)

                                # # 保存目标结果视频
                                # out.write(image)

                                # 设置新的尺寸
                                new_width = 1080
                                new_height = int(new_width * (9 / 16))
                                # 调整图像尺寸
                                resized_image = cv2.resize(image, (new_width, new_height))
                                resized_frame = cv2.resize(framecopy, (new_width, new_height))
                                if self.display_mode == "智能叠加显示":
                                    self.image_placeholder.image(resized_image, channels="BGR", caption="🎥 显微镜序列图像分析")
                                else:
                                    self.image_placeholder.image(resized_frame, channels="BGR", caption="📷 原始显微镜序列图像")
                                    self.image_placeholder_res.image(resized_image, channels="BGR", caption="🤖 AI分割标注序列")

                                self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                                # 更新进度条
                                if total_length > 0:
                                    progress_percentage = int(((current_frame + 1) / total_frames) * 100)
                                    try:
                                        self.progress_bar.progress(progress_percentage)
                                    except:
                                        pass

                                current_frame += 1
                        else:
                            break
                    if self.close_flag:
                        self.logTable.save_to_csv()
                        cap.release()
                        # out.release()

                    self.logTable.save_to_csv()
                    # 视频处理完成后更新分析评估
                    self.update_analysis_assessment()
                    cap.release()
                    # out.release()

                finally:

                    if self.uploaded_video is None:
                        name_in = None
                    else:
                        name_in = self.uploaded_video.name

                    res = self.logTable.save_frames_file(fps=self.FPS, video_name=name_in)
                    st.write("识别结果文件已经保存：" + self.saved_log_data)
                    if res:
                        st.write(f"结果的目标文件已经保存：{res}")

                    tfile.close()
                    # 如果不需要再保留临时文件，可以在处理完后删除
                    print(tfile.name + ' 临时文件可以删除')
                    # os.remove(tfile.name)

            else:
                st.warning("请选择摄像头或上传文件。")

    def toggle_comboBox(self, frame_id):
        """
        处理并显示指定帧的检测结果。

        Args:
            frame_id (int): 指定要显示检测结果的帧ID。

        根据用户选择的帧ID，显示该帧的检测结果和图像。
        """
        # 确保已经保存了检测结果
        if len(self.logTable.saved_results) > 0:
            frame = self.logTable.saved_images_ini[-1]  # 获取最近一帧的图像
            image = frame  # 将其设为当前图像

            # 遍历所有保存的检测结果
            for i, detInfo in enumerate(self.logTable.saved_results):
                if frame_id != -1:
                    # 如果指定了帧ID，只处理该帧的结果
                    if frame_id != i:
                        continue

                if len(detInfo) > 0:
                    name, bbox, conf, use_time, cls_id = detInfo  # 获取检测信息
                    label = '%s %.0f%%' % (name, conf * 100)  # 构造标签文本

                    disp_res = ResultLogger()  # 创建结果记录器
                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(use_time))  # 合并结果
                    self.table_placeholder.table(res)  # 在表格中显示结果

                    # 如果有保存的初始图像
                    if len(self.logTable.saved_images_ini) > 0:
                        if len(self.colors) < cls_id:
                            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                                           range(cls_id+1)]
                        image = drawRectBox(image, bbox, alpha=0.2, addText=label,
                                            color=self.colors[cls_id])  # 绘制检测框和标签

            # 设置新的尺寸并调整图像尺寸
            new_width = 1080
            new_height = int(new_width * (9 / 16))
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # 根据显示模式显示处理后的图像或原始图像
            if self.display_mode == "智能叠加显示":
                self.image_placeholder.image(resized_image, channels="BGR", caption="🔍 AI细胞分析结果")
            else:
                self.image_placeholder.image(resized_frame, channels="BGR", caption="🔬 原始显微镜图像")
                self.image_placeholder_res.image(resized_image, channels="BGR", caption="🤖 AI细胞分析结果")

    def frame_process(self, image, file_name,video_time = None):
        """
        处理并预测单个图像帧的内容。

        Args:
            image (numpy.ndarray): 输入的图像。
            file_name (str): 处理的文件名。

        Returns:
            tuple: 处理后的图像，检测信息，选择信息列表。

        对输入图像进行预处理，使用模型进行预测，并处理预测结果。
        """
        # image = cv2.resize(image, (640, 640))  # 调整图像大小以适应模型
        pre_img = self.model.preprocess(image)  # 对图像进行预处理

        # 更新模型参数
        params = {'conf': self.conf_threshold, 'iou': self.iou_threshold}
        self.model.set_param(params)

        t1 = time.time()
        pred = self.model.predict(pre_img)  # 使用模型进行预测

        t2 = time.time()
        use_time = t2 - t1  # 计算单张图片推理时间

        aim_area = 0 #计算目标面积

        det = pred[0]  # 获取预测结果

        # 初始化检测信息和选择信息列表
        detInfo = []
        select_info = ["全部目标"]

        # 如果有有效的检测结果
        if det is not None and len(det):
            det_info = self.model.postprocess(pred)  # 后处理预测结果
            if len(det_info):
                disp_res = ResultLogger()
                res = None
                cnt = 0

                # 遍历检测到的对象
                for info in det_info:
                    name, bbox, conf, cls_id, mask = info['class_name'], info['bbox'], info['score'], info['class_id'], info['mask']

                    # 绘制检测框、标签和面积信息
                    image,aim_frame_area = draw_detections(image, info, alpha=0.5)
                    # image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=self.colors[cls_id])

                    # 根据不同类型提供专业化的生物学术语描述
                    biological_description = self.get_biological_description(name, int(aim_frame_area))
                    
                    res = disp_res.concat_results(name, bbox, biological_description,
                                                  video_time if video_time is not None else str(round(use_time, 2)))

                    # 添加日志条目
                    self.logTable.add_log_entry(file_name, name, bbox, biological_description, video_time if video_time is not None else str(round(use_time, 2)))
                    # 记录检测信息
                    detInfo.append([name, bbox, biological_description, video_time if video_time is not None else str(round(use_time, 2)), cls_id])
                    # 添加到选择信息列表
                    select_info.append(name + "-" + str(cnt))
                    cnt += 1

                # 在表格中显示检测结果
                self.table_placeholder.table(res)
                
                # 实时更新分析评估
                self.update_analysis_assessment()

        return image, detInfo, select_info

    def get_biological_description(self, class_name, area):
        """
        根据检测类别和细胞面积生成专业的生物学描述
        """
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

    def generate_analysis_assessment_content(self):
        """
        生成血细胞分析评估汇总内容
        """
        if not hasattr(self, 'logTable') or len(self.logTable.saved_results) == 0:
            return "📊 暂无血细胞分析数据", "", [], {}
        
        # 统计血细胞各类型的检测结果
        analysis_stats = {
            "嗜碱性粒细胞": 0,
            "嗜酸性粒细胞": 0,
            "幼红细胞": 0,
            "侵入物": 0,
            "淋巴细胞": 0,
            "单核细胞": 0,
            "髓细胞": 0,
            "中性粒细胞": 0,
            "血小板": 0,
            "红细胞": 0,
            "总检测数": len(self.logTable.saved_results)
        }
        
        for result in self.logTable.saved_results:
            if len(result) >= 1:
                # 结果结构：[name, bbox, biological_description, time, cls_id]
                class_name = result[0] if len(result) > 0 else "未知"
                # 使用中文名称进行统计
                if str(class_name) in analysis_stats:
                    analysis_stats[str(class_name)] += 1
        
        # 计算血细胞分布质量等级
        total_cells = sum([analysis_stats[key] for key in analysis_stats if key != "总检测数"])
        cell_types_found = len([key for key in analysis_stats if analysis_stats[key] > 0 and key != "总检测数"])
        
        if total_cells >= 20 and cell_types_found >= 5:
            quality_level = "🟢 优秀血涂片"
            quality_color = "#2ed573"
            recommendations = [
                f"检测到{cell_types_found}种血细胞类型，样本多样性好",
                f"总计{total_cells}个细胞，数量充足适合分析",
                "血细胞形态清晰，适合血液学研究",
                "建议保存为高质量血液样本"
            ]
        elif total_cells >= 10 and cell_types_found >= 3:
            quality_level = "🟡 良好血涂片"
            quality_color = "#ffa726"
            recommendations = [
                f"检测到{cell_types_found}种血细胞类型",
                f"总计{total_cells}个细胞，基本满足分析要求",
                "可进行基本血液学观察",
                "适合教学演示使用"
            ]
        else:
            quality_level = "🔴 需要改进"
            quality_color = "#ff4757"
            recommendations = [
                f"仅检测到{cell_types_found}种血细胞类型，样本单一",
                f"总计{total_cells}个细胞，数量偏少",
                "建议优化血涂片制备技术",
                "考虑重新采样或调整成像参数"
            ]
        
        return quality_level, quality_color, recommendations, analysis_stats

    def update_analysis_assessment(self):
        """
        更新分析评估显示
        """
        if not hasattr(self, 'analysis_assessment_placeholder'):
            return
            
        quality_level, quality_color, recommendations, analysis_stats = self.generate_analysis_assessment_content()
        
        # 如果没有数据，显示提示信息
        if isinstance(quality_level, str) and "暂无分析数据" in quality_level:
            self.analysis_assessment_placeholder.info(quality_level)
            return
        
        # 使用占位符更新内容
        with self.analysis_assessment_placeholder.container():
            # 显示血细胞分析评估结果
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border-left: 4px solid {quality_color}; padding: 15px; border-radius: 5px;">
                    <h5 style="color: {quality_color}; margin-top: 0;">血液分析质量：{quality_level}</h5>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 10px;">
                        <span><strong>中性粒细胞：</strong>{analysis_stats.get('中性粒细胞', 0)}</span>
                        <span><strong>淋巴细胞：</strong>{analysis_stats.get('淋巴细胞', 0)}</span>
                        <span><strong>单核细胞：</strong>{analysis_stats.get('单核细胞', 0)}</span>
                        <span><strong>嗜酸性粒细胞：</strong>{analysis_stats.get('嗜酸性粒细胞', 0)}</span>
                        <span><strong>嗜碱性粒细胞：</strong>{analysis_stats.get('嗜碱性粒细胞', 0)}</span>
                        <span><strong>红细胞：</strong>{analysis_stats.get('红细胞', 0)}</span>
                        <span><strong>血小板：</strong>{analysis_stats.get('血小板', 0)}</span>
                        <span><strong>幼红细胞：</strong>{analysis_stats.get('幼红细胞', 0)}</span>
                        <span><strong>髓细胞：</strong>{analysis_stats.get('髓细胞', 0)}</span>
                        <span><strong>侵入物：</strong>{analysis_stats.get('侵入物', 0)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 显示分析建议
            st.markdown("**🩸 血液学分析建议：**")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
                
            # 显示主要血细胞类型的统计图表
            if analysis_stats.get("总检测数", 0) > 0:
                col1, col2, col3, col4, col5 = st.columns(5)
                total = analysis_stats.get("总检测数", 1)  # 避免除零
                with col1:
                    neutrophil_count = analysis_stats.get("中性粒细胞", 0)
                    st.metric("中性粒细胞", neutrophil_count, 
                             delta=f"{neutrophil_count/total*100:.1f}%")
                with col2:
                    lymphocyte_count = analysis_stats.get("淋巴细胞", 0)
                    st.metric("淋巴细胞", lymphocyte_count,
                             delta=f"{lymphocyte_count/total*100:.1f}%")
                with col3:
                    monocyte_count = analysis_stats.get("单核细胞", 0)
                    st.metric("单核细胞", monocyte_count,
                             delta=f"{monocyte_count/total*100:.1f}%")
                with col4:
                    rbc_count = analysis_stats.get("红细胞", 0)
                    st.metric("红细胞", rbc_count,
                             delta=f"{rbc_count/total*100:.1f}%")
                with col5:
                    platelet_count = analysis_stats.get("血小板", 0)
                    st.metric("血小板", platelet_count,
                             delta=f"{platelet_count/total*100:.1f}%")

    def frame_table_process(self, frame, caption):
        # 显示画面并更新结果
        self.image_placeholder.image(frame, channels="BGR", caption=caption)

        # 更新检测结果
        detection_result = "None"
        detection_location = "[0, 0, 0, 0]"
        detection_confidence = str(random.random())
        detection_time = "0.00s"

        # 使用 display_detection_results 函数显示结果
        res = concat_results(detection_result, detection_location, detection_confidence, detection_time)
        self.table_placeholder.table(res)
        # 添加适当的延迟
        cv2.waitKey(1)

    def setupMainWindow(self):
        """ 运行细胞组织分割系统。 """
        # 专业化的分隔线和系统信息
        st.markdown(
            """
            <div style="text-align: center; color: #666; margin: 20px 0;">
                <hr style="border: 1px solid #e0e0e0;">
                <p style="margin: 10px 0; font-size: 0.9em;">
                    🩸 AI-Powered Blood Cell Analysis Platform | 基于人工智能的血细胞分析平台
                </p>
                <hr style="border: 1px solid #e0e0e0;">
            </div>
            """,
            unsafe_allow_html=True
        )

        # 创建列布局，优化细胞分析界面
        col1, col2, col3 = st.columns([4, 1, 2])

        # 在第一列设置显示模式的选择
        with col1:
            st.markdown("### 🔬 图像显示模式")
            self.display_mode = st.radio(
                "选择显示方式", 
                ["智能叠加显示", "对比分析显示"],
                help="叠加显示：在原图上直接标注分割结果；对比显示：原图与分割结果分别显示"
            )
            
            # 根据显示模式创建用于显示视频画面的空容器
            if self.display_mode == "智能叠加显示":
                self.image_placeholder = st.empty()
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="🔬 等待显微镜图像输入...")
            else:
                # "对比分析显示"
                st.markdown("**原始图像 vs AI分割结果**")
                self.image_placeholder = st.empty()
                self.image_placeholder_res = st.empty()
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="🔬 原始显微镜图像")
                    self.image_placeholder_res.image(load_default_image(), caption="🤖 AI分割结果")
            
            # 显示用的进度条
            st.markdown("**🔄 分析进度**")
            self.progress_bar = st.progress(0)

        # 创建一个空的结果表格
        res = concat_results("等待分析", "待分割区域", "0.00", "0.00s")

        # 在最右侧列设置分析结果显示
        with col3:
            st.markdown("### 🔬 AI分析报告")
            self.table_placeholder = st.empty()  # 调整到最右侧显示
            self.table_placeholder.table(res)

            # 分析质量评估汇总
            st.markdown("---")
            st.markdown("**📊 分析质量评估**")
            self.analysis_assessment_placeholder = st.empty()
            self.update_analysis_assessment()

        # 在中间列设置控制按钮
        with col2:
            st.markdown("### 🎮 控制面板")
            self.close_placeholder = st.empty()
            
            # 主要控制按钮
            st.markdown("**主控制**")
            if st.button("🔬 开始AI分析", help="启动AI细胞组织分割分析", type="primary"):
                self.process_camera_or_file()  # 运行摄像头或文件处理
            
            # 紧急停止按钮
            st.markdown("**紧急控制**")
            if st.button("⏹️ 停止分析", help="立即停止当前分析进程"):
                st.warning("⚠️ 分析进程已停止")
            
            # 系统状态显示
            st.markdown("---")
            st.markdown("**📈 系统状态**")
            status_placeholder = st.empty()
            
            if not hasattr(self, 'model') or self.model is None:
                status_placeholder.error("🔴 AI模型未加载")
            else:
                status_placeholder.success("🟢 AI模型就绪")
                
            # 实时统计信息
            if hasattr(self, 'logTable') and len(self.logTable.saved_results) > 0:
                total_analyses = len(self.logTable.saved_results)
                st.metric("总分析数", total_analyses)
            else:
                st.metric("总分析数", 0)
                
            # 如果没有保存的图像，则显示默认图像（仅在非运行状态）
            if not st.session_state.get('analysis_running', False):
                if not self.logTable.saved_images_ini:
                    if self.display_mode == "智能叠加显示":
                        self.image_placeholder.image(load_default_image(), caption="🔬 等待显微镜图像输入...")
                    else:  # 对比分析显示
                        self.image_placeholder.image(load_default_image(), caption="🔬 原始显微镜图像")
                        self.image_placeholder_res.image(load_default_image(), caption="🤖 AI分割结果")

        # 添加公司版权信息到页面底部
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;">
                <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
                    © 2025 <strong style="color: #2c3e50;">合溪生物科技</strong> | 
                    Powered by Hexi Biotechnology Co., Ltd.
                </p>
                <p style="margin: 5px 0 0 0; color: #adb5bd; font-size: 0.8em;">
                    专业医学AI影像分析解决方案提供商
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# 实例化并运行应用
if __name__ == "__main__":
    app = Detection_UI()
    app.setupMainWindow()
