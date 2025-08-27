# -*- coding: utf-8 -*-

# 中文到英文的映射，用于图像显示（解决字体显示问题）
Chinese_name = {
    'nucleus': "细胞核",
    'cytoplasm': "细胞质",
    'tissue': "组织结构",
    'vessel': "血管",
    'cell': "细胞",
    'membrane': "细胞膜"
}

# 英文标签，用于图像上的文字显示
English_labels = {
    'nucleus': "Nucleus",
    'cytoplasm': "Cytoplasm",
    'tissue': "Tissue", 
    'vessel': "Vessel",
    'cell': "Cell",
    'membrane': "Membrane"
}

# 中文到英文的映射
Chinese_to_English = {
    "细胞核": "Nucleus",
    "细胞质": "Cytoplasm",
    "组织结构": "Tissue",
    "血管": "Vessel",
    "细胞": "Cell",
    "细胞膜": "Membrane"
}

Label_list = list(Chinese_name.values())
