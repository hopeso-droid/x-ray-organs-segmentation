# X-ray Organs Segmentation

🩻 **AI X光胸片器官分割系统 - X-ray Chest Organs Segmentation System**

基于深度学习的X光胸片器官智能识别与分割平台，采用YOLO技术实现高精度医学影像分析。

## 🚀 系统特点

- **智能分割**: 深度学习模型，实现胸部器官的精确识别与分割
- **多器官支持**: 支持心脏、肺部、横膜、肋骨、胸椎等多种胸部器官的分析
- **实时分析**: 提供实时图像分析和批量处理功能
- **专业界面**: 现代化的Web界面，适合医学研究和教学使用
- **多格式支持**: 支持JPEG、PNG、TIFF、DICOM等多种医学影像格式

## 🔧 技术架构

- **前端**: Streamlit Web应用框架
- **后端**: Python + OpenCV + PyTorch
- **AI模型**: 基于后训练独立优化的胸部X光器官分割系统
- **图像处理**: OpenCV + PIL
- **部署**: 支持本地部署和云端部署

## 📋 支持的器官类型

- 🫀 心脏 (Heart)
- 🫁 肺部 (Lungs)
- 🦴 肋骨 (Ribs)
- 🔗 胸椎 (Thoracic Spine)
- 📏 横膜 (Diaphragm)
- 🦴 胸骨 (Sternum)
- 🦴 锁骨 (Clavicle)
- 🫁 胸腔 (Thoracic Cavity)

## 🛠️ 安装与运行

### 环境要求
- Python 3.8+
- CUDA支持的GPU (推荐)

### 快速开始

```bash
# 克隆项目
git clone https://github.com/hopeso-droid/x-ray-organs-segmentation.git
cd x-ray-organs-segmentation

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

## 📊 使用方法

1. **上传X光胸片**: 支持单张图片或批量上传
2. **参数调节**: 调整置信度阈值和IoU阈值
3. **选择模式**: 检测模式或分割模式
4. **开始分析**: 点击分析按钮开始AI处理
5. **查看结果**: 获得器官分割结果和详细报告

## 🔬 分析功能

- **器官识别**: 自动识别胸部主要器官
- **精确分割**: 像素级别的器官边界描绘
- **形态分析**: 器官面积、周长、形状分析
- **质量评估**: 分析结果的可信度评估
- **报告生成**: 专业的分析报告输出

## ⚠️ 使用声明

- 本系统仅供医学影像科研和教学使用
- 不可用于临床诊断或医疗决策
- 分析结果需要专业医师验证
- 使用前请确保数据合规性

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交问题报告和功能建议！

## 📧 联系方式

- **开发团队**: 合溪生物科技
- **技术支持**: 请通过GitHub Issues联系

---

**注意**: 此系统为科研工具，不能替代专业医学诊断。