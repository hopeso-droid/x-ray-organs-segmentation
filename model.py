# -*- coding: utf-8 -*-
import cv2  # 导入OpenCV库，用于处理图像和视频
import torch
# 云端环境替代导入
try:
    from QtFusion.models import Detector, HeatmapGenerator
except ImportError:
    # 创建简单的基类替代
    class Detector:
        """简单的检测器基类"""
        def __init__(self, params=None):
            """初始化检测器
            
            Args:
                params: 检测器参数，可以为None
            """
            self.params = params or {}
            self.imgsz = 640  # 默认图像尺寸
    
    class HeatmapGenerator:
        """简单的热图生成器基类"""
        def __init__(self, params=None):
            """初始化热图生成器
            
            Args:
                params: 生成器参数，可以为None
            """
            self.params = params or {}
from chinese_name_list import Chinese_name  # 从datasets库中导入Chinese_name字典，用于获取类别的中文名称
from ultralytics import YOLO  # 从ultralytics库中导入YOLO类，用于加载YOLO模型
from ultralytics.utils.torch_utils import select_device  # 从ultralytics库中导入select_device函数，用于选择设备
import os

# 导入云端日志器
try:
    from cloud_utils import cloud_logger
except ImportError:
    # 如果导入失败，创建一个简单的日志替代
    class SimpleLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    cloud_logger = SimpleLogger()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

ini_params = {
    'device': device,  # 设备类型，这里设置为CPU
    'conf': 0.05,  # 物体置信度阈值
    'iou': 0.2,  # 用于非极大值抑制的IOU阈值
    'classes': None,  # 类别过滤器，这里设置为None表示不过滤任何类别
    'verbose': False
}


def count_classes(det_info, class_names):
    """
    Count the number of each class in the detection info.

    :param det_info: List of detection info, each item is a list like [class_name, bbox, conf, class_id]
    :param class_names: List of all possible class names
    :return: A list with counts of each class
    """
    count_dict = {name: 0 for name in class_names}  # 创建一个字典，用于存储每个类别的数量
    for info in det_info:  # 遍历检测信息
        class_name = info['class_name']  # 获取类别名称
        if class_name in count_dict:  # 如果类别名称在字典中
            count_dict[class_name] += 1  # 将该类别的数量加1

    # Convert the dictionary to a list in the same order as class_names
    count_list = [count_dict[name] for name in class_names]  # 将字典转换为列表，列表的顺序与class_names相同
    return count_list  # 返回列表


class Web_Detector(Detector):  # 定义YOLOv8Detector类，继承自Detector类
    def __init__(self, params=None):  # 定义构造函数
        super().__init__(params)  # 调用父类的构造函数
        self.model = None
        self.img = None  # 初始化图像为None
        self.names = list(Chinese_name.values())  # 获取所有类别的中文名称
        self.params = params if params else ini_params  # 如果提供了参数则使用提供的参数，否则使用默认参数
        self.imgsz = 640  # 设置默认图像尺寸，用于模型预热

    def load_model(self, model_path):  # 定义加载模型的方法
        try:
            cloud_logger.info(f"开始加载模型: {model_path}")
            self.device = select_device(self.params['device'])  # 选择设备
            cloud_logger.info(f"选择设备: {self.device}")
            
            # 确定任务类型
            if os.path.basename(model_path)[:3] == 'seg' or 'seg' in model_path.lower():
                task = 'segment'
            else:
                task = 'segment'  # 默认使用分割任务
            
            cloud_logger.info(f"加载YOLO模型，任务类型: {task}")
            self.model = YOLO(model_path, task=task)
            
            cloud_logger.info("获取类别名称")
            names_dict = self.model.names  # 获取类别名称字典
            self.names = [Chinese_name[v] if v in Chinese_name else v for v in names_dict.values()]  # 将类别名称转换为中文
            cloud_logger.info(f"模型类别数量: {len(self.names)}")
            
            # 安全的模型预热
            try:
                cloud_logger.info("开始模型预热")
                dummy_input = torch.zeros(1, 3, self.imgsz, self.imgsz)
                if self.device.type != 'cpu':
                    dummy_input = dummy_input.to(self.device)
                    dummy_input = dummy_input.type_as(next(self.model.model.parameters()))
                
                with torch.no_grad():
                    _ = self.model(dummy_input, verbose=False)
                cloud_logger.info("模型预热完成")
            except Exception as e:
                cloud_logger.warning(f"模型预热失败，但可以继续使用: {e}")
            
            cloud_logger.info("✅ 模型加载成功")
            
        except Exception as e:
            cloud_logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
    def preprocess(self, img):  # 定义预处理方法
        self.img = img  # 保存原始图像
        return img  # 返回处理后的图像

    def predict(self, img):  # 定义预测方法
        results = self.model(img, **ini_params)
        return results

    def postprocess(self, pred):
        # 定义后处理方法
        results = []  # 初始化结果列表
        aim_id = 0  # 初始化 aim_id

        for res in pred[0].boxes:
            for box in res:
                # 提前计算并转换数据类型
                class_id = int(box.cls.cpu())
                bbox = box.xyxy.cpu().squeeze().tolist()
                bbox = [int(coord) for coord in bbox]  # 转换边界框坐标为整数

                result = {
                    "class_name": self.names[class_id],  # 类别名称
                    "bbox": bbox,  # 边界框
                    "score": box.conf.cpu().squeeze().item(),  # 置信度
                    "class_id": class_id,  # 类别ID
                    "mask": pred[0].masks[aim_id].xy if pred[0].masks is not None else None  # 掩码
                }
                results.append(result)  # 将结果添加到列表

                aim_id += 1  # 增加 aim_id，确保每个框都有对应的 mask

        return results  # 返回结果列表

    def set_param(self, params):
        self.params.update(params)
