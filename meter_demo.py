import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import timeit
import yaml
import matplotlib.pyplot as plt
import argparse
from data.yolov5_detect import YOLOv5_Detect


trans =transforms.Compose([transforms.ToTensor(),])

def init_model(destinate_pt_path):
    model = torch.jit.load(destinate_pt_path).to("cpu")  
    model.eval()
    return model

def calValue(angle, yaml_config):
    fromValue = yaml_config['fromValue'] 
    valueRange = yaml_config['valueRange'] 
    fromAngle = yaml_config['fromAngle']
    angles = yaml_config['angles']
    return fromValue + valueRange * (angle - fromAngle) / angles

def letterbox(img, new_shape=(512,512), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


def vector_calc_angle(output):
    
    x_axis = np.array([1,0])   # x轴正方向向量,
    line = np.array([output[0],output[1]])  #拟合的线 向量
    dot_product = np.dot(x_axis, line)
    if dot_product != 0:
        angle = np.arccos(dot_product)
        angle2x = angle * 180 / np.pi
    else:
        angle2x = 90
    return float(angle2x)


def infer_forward(image, coordinate, model, yaml_path): 
           
    sita = None
    areas = []
    pointer2 = []
    file = open(yaml_path, 'r')
    yaml_data = yaml.load(file.read(), Loader=yaml.FullLoader)
    file.close
    image = cv2.imread(image)
    image_crop = image[coordinate[1]:coordinate[3],coordinate[0]:coordinate[2]]
    image_pad, _, _ = letterbox(image_crop)
    lastx = image_pad.shape[0] / 2
    lasty = image_pad.shape[1] / 2
    img = cv2.cvtColor(image_pad, cv2.COLOR_BGR2RGB) 
    img_tensor = trans(img)
    img_tensor = img_tensor.unsqueeze(0).to("cpu")
    output = model(img_tensor).squeeze(0).permute(1,2,0).argmax(dim=2).detach().cpu().numpy()
    mask = np.where(output >= 2, 1, 0)
    for k in np.transpose(np.nonzero(mask)).tolist():
            pointer2.append([k[1],k[0]])
    output = cv2.fitLine(np.array(pointer2), cv2.DIST_L2, 0, 0.01, 0.01) 
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # type: ignore

    for x in range(len(contours)): 
        areas.append(cv2.contourArea(contours[x]))
    max_id = areas.index(max(areas))
    rect_value = cv2.minAreaRect(contours[max_id])
    
    if yaml_data['angles'] <= 180 and float(output[1]) > 0:   #半圆表盘                         
        sita = 90 + vector_calc_angle(output) 
    elif yaml_data['angles'] <= 180 and float(output[1]) < 0:
        sita = 270 - vector_calc_angle(output) 
    elif yaml_data['angles'] <= 180 and float(output[1]) == 0:
        if float(rect_value[0][0]) <= float(lastx):
            sita = 90
        else:
            sita = 270
    
    else:                                                      #圆表盘
        if float(rect_value[0][1]) >= float(lasty) and float(output[1]) < 0:             
            sita = 90 - vector_calc_angle(output)    
        
        elif float(rect_value[0][1]) >= float(lasty) and float(output[1]) > 0:
            sita = 270 + vector_calc_angle(output) 
            
        elif float(rect_value[0][1]) < float(lasty) and float(output[1]) < 0:
            sita = 270 - vector_calc_angle(output)
            
        elif float(rect_value[0][1]) < float(lasty) and float(output[1]) > 0:
            sita = 90 + vector_calc_angle(output)
            
        elif float(output[1]) == 0:
            if float(rect_value[0][0]) <= float(lastx):
                sita = 90
            else:
                sita = 270
        else:
            print("----None----")

    if sita != None:
        clock_value = calValue(sita, yaml_data)
    else:
        clock_value = sita
    return clock_value


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default='./data/model/detect_meter.torchscript', help='meter detect model')
    parser.add_argument('--segment_model', type=str, default='./data/model/segment_meter_letterbox.pt', help='meter segment model')
    parser.add_argument('--image_path', type=str, default='./data/pic/C240.jpg', help='image path')
    parser.add_argument('--yaml_path', type=str, default='./data/config/config_C240.yaml', help='config parameters')
    opt = parser.parse_args()

    detect_model, segment_model = opt.detect_model, opt.segment_model
    image_path, yaml_path = opt.image_path, opt.yaml_path
    
    detect_model = init_model(detect_model)
    infer_model  = init_model(segment_model)
    table_detect = YOLOv5_Detect(detect_model)
    coordinate = table_detect(image_path)
    clock_value = infer_forward(image_path, coordinate, infer_model, yaml_path)
    print("clock value=",f'{clock_value:.2f}')
