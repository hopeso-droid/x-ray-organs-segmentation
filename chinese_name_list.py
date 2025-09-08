# -*- coding: utf-8 -*-

# 根据X光胸片器官分割模型的类别定义中英文映射
Chinese_name = {
    'Heart': "心脏",
    'Left_Lung': "左肺",
    'Right_Lung': "右肺", 
    'Spine': "脊椎",
    'Weasand': "气管"
}

# 英文标签，用于图像上的文字显示
English_labels = {
    'Heart': "Heart",
    'Left_Lung': "Left_Lung",
    'Right_Lung': "Right_Lung",
    'Spine': "Spine",
    'Weasand': "Weasand"
}

# 中文到英文的映射
Chinese_to_English = {
    "心脏": "Heart",
    "左肺": "Left_Lung",
    "右肺": "Right_Lung",
    "脊椎": "Spine",
    "气管": "Weasand"
}

# 英文到中文的映射
English_to_Chinese = {
    "Heart": "心脏",
    "Left_Lung": "左肺",
    "Right_Lung": "右肺",
    "Spine": "脊椎",
    "Weasand": "气管"
}

Label_list = list(Chinese_name.values())