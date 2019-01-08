from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import argparse

from sort import Sort  #跟踪代码头文件


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img,result):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    result.append([c1[0],c1[1],c2[0],c2[1]])

    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    # cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # cv2.rectangle(img, c1, c2,color, -1)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

# 全局变量
colours = np.random.rand(32,3)*255
def track_fun(result,mot_tracker,orig_im):
    #参数介绍
    # result = []  # 坐标框容器  result.append([c1[0],c1[1],c2[0],c2[1]])
    # mot_tracker = Sort(args.sort_max_age, args.sort_min_hit) # 跟踪器对象
    # orig_im 最终显示图像
    result = np.array(result)  # 提出result，给det
    det = []
    if result != []:  # 非常重要，非空判断
        det = result[:, 0:4]

    trackers = mot_tracker.update(det)  # 用mot_tracker的update接口去更新det，进行多目标的跟踪
    for track in trackers:
        # 左上角坐标(x,y)
        lrx = int(track[0])
        lry = int(track[1])
        # 右下角坐标(x,y)
        rtx = int(track[2])
        rty = int(track[3])
        # track_id
        trackID = int(track[4])

        cv2.putText(orig_im, str(trackID), (lrx, lry), cv2.FONT_ITALIC, 0.6, (
            int(colours[trackID % 32, 0]), int(colours[trackID % 32, 1]), int(colours[trackID % 32, 2])), 2)
        cv2.rectangle(orig_im, (lrx, lry), (rtx, rty), (
            int(colours[trackID % 32, 0]), int(colours[trackID % 32, 1]), int(colours[trackID % 32, 2])), 2)

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest = 'video', help = "Video to run detection upon", default = "MOT06.mp4", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help ="Config file",default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "cfg/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)    # 必须是32的整数倍

    # track_sort的参数
    parser.add_argument('--sort_max_age',default=5,type=int)
    parser.add_argument('--sort_min_hit',default=3,type=int)

    return parser.parse_args()



if __name__ == '__main__':
    args = arg_parse()

    ## track_sort初始化
    mot_tracker = Sort(args.sort_max_age, args.sort_min_hit)


    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    num_classes = 80

    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()
    
    videofile = args.video
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img, volatile = True), CUDA)

            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))
            
            im_dim = im_dim.repeat(output.size(0), 1)/inp_dim
            output[:,1:5] *= im_dim
            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

####
            result = []  # 坐标框容器
            list(map(lambda x: write(x, orig_im,result), output))

            track_fun(result, mot_tracker, orig_im)  # 跟踪代码
####

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break
    

    
    

