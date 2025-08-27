# -*- coding: utf-8 -*-

# 根据data.yaml中的血细胞类别定义中英文映射
Chinese_name = {
    'Basophil': "嗜碱性粒细胞",
    'Eosinophil': "嗜酸性粒细胞", 
    'Erythroblast': "幼红细胞",
    'Intrusion': "侵入物",
    'Lymphocyte': "淋巴细胞",
    'Monocyte': "单核细胞",
    'Myelocyte': "髓细胞",
    'Neutrophil': "中性粒细胞",
    'Platelet': "血小板",
    'RBC': "红细胞"
}

# 英文标签，用于图像上的文字显示
English_labels = {
    'Basophil': "Basophil",
    'Eosinophil': "Eosinophil",
    'Erythroblast': "Erythroblast", 
    'Intrusion': "Intrusion",
    'Lymphocyte': "Lymphocyte",
    'Monocyte': "Monocyte",
    'Myelocyte': "Myelocyte",
    'Neutrophil': "Neutrophil",
    'Platelet': "Platelet",
    'RBC': "RBC"
}

# 中文到英文的映射
Chinese_to_English = {
    "嗜碱性粒细胞": "Basophil",
    "嗜酸性粒细胞": "Eosinophil",
    "幼红细胞": "Erythroblast",
    "侵入物": "Intrusion",
    "淋巴细胞": "Lymphocyte",
    "单核细胞": "Monocyte",
    "髓细胞": "Myelocyte",
    "中性粒细胞": "Neutrophil",
    "血小板": "Platelet",
    "红细胞": "RBC"
}

Label_list = list(Chinese_name.values())
