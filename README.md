# 表计读数
表盘检测
指针分割
求出角度
计算读数

## 运行demo
meter_demo.py文件进行表盘检测、指针分割及输出读数值。可以直接运行bash run.sh

## 参数说明
--detect_model 表盘检测模型路径, --segment_model 指针分割模型路径, --image_path 待读数的表计图片路径, --yaml_path 与该表计相匹配的配置文件路径

## config配置文件参数说明

fromValue: 表计起始值  
valueRange:  表计值范围
fromAngle : 表计起始角度
angles: 表计角度范围

本项目使用设备CPU可以使用，无需GPU。 

tag = v1.0 使用最小外接矩形计算角度
tag = v2.0 使用拟合直线和两向量夹角计算角度，同时更新了分割模型 
