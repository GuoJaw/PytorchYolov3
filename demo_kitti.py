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


import math  #s数学公式头文件
import pandas as pd
import csv


###

def createCSVfile():
    global csvfile_
    global writer_
    global  file_name_

    file_name_ = 'CSV.csv'

    csvfile_ = open(file_name_, 'w')
    writer_ = csv.writer(csvfile_, delimiter=',')
    writer_.writerow(['frame_id', 'track_id', 'distance', 'center_x', 'center_y'])
    csvfile_.flush()


####


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
    #cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    #cv2.rectangle(img, c1, c2,color, -1)
    #cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#求距离
def distance_Car_Person( uf, vf, height, width):   # uf,vf分别对应矩形中心点center的坐标
	# 相机内参
	a = 105*3.1415926/180   # sin30°就得写成 Math.sin（30*Math.PI/180）
	h = 150   #镜头中心到地面的高度
	ax = 3507.703240769799  #相机内参 f/dx
	ay = 3506.889066124179  #相机内参 f/dy


	uo = int(width/2)  #uo，vo  成像平面坐标系的原点在像素坐标系中的像素坐标
	vo = int(height/2)


	e = (vo - vf) / ay
	b = math.atan(e)
	d = h / (math.tan(a + b))
	up = (h*h + d*d)*(uf - uo)*(uf - uo)
	down = (vf - vo)*(vf - vo)*(ax / ay)*(ax / ay) + ax*ax
	left = up / down
	l = int(math.sqrt(left + d*d))

	return l

# 从csv中获取id_num上次出现的值last_dist，用distance与last_dist对比，判断是否变近
def func(orig_im,distance,bottom_center,id_num,safe_distance,lt, rb,cT,height):
    ## 读取csv文件中的内容
    df = pd.read_csv(file_name_)

    # 获取track_id = id_num的最后5行
    last_row_bottom_values = df[df['track_id'].isin([id_num])].tail(5).values
    if (len(last_row_bottom_values) >= 5):
        last_bottom_center_1 = (int(last_row_bottom_values[0][3]), int(last_row_bottom_values[0][4]))
        last_bottom_center_2 = (int(last_row_bottom_values[1][3]), int(last_row_bottom_values[1][4]))
        last_bottom_center_3 = (int(last_row_bottom_values[2][3]), int(last_row_bottom_values[2][4]))
        last_bottom_center_4 = (int(last_row_bottom_values[3][3]), int(last_row_bottom_values[3][4]))
        last_bottom_center_5 = (int(last_row_bottom_values[4][3]), int(last_row_bottom_values[4][4]))

        cv2.circle(orig_im, last_bottom_center_1, 2, (0, 255, 255), 3, -1)  # 画点
        cv2.circle(orig_im, last_bottom_center_2, 2, (0, 255, 255), 3, -1)
        cv2.circle(orig_im, last_bottom_center_3, 2, (0, 255, 255), 3, -1)
        cv2.circle(orig_im, last_bottom_center_4, 2, (0, 255, 255), 3, -1)
        cv2.circle(orig_im, last_bottom_center_5, 3, (0, 255, 255), 3, -1)

    ####                df[df['track_id'].isin([id_num])]      tail(1)最后一行      values查看数据值
    row_bottom_values = df[df['track_id'].isin([id_num])].tail(1).values
    if safe_distance > distance:  # 如果在危险距离内部，进行决策
        if (len(row_bottom_values)):  # 如果 row_bottom_values 不是 []
            last_dist = int(row_bottom_values[0][2])
            # print(last_dist)
            if (last_dist > distance):  # 判断距离变近,则预警
                cv2.rectangle(orig_im, lt, rb, (0, 0, 255), 2)
                cv2.putText(orig_im, "!", cT, 0, 1e-3 * height * 2, (0, 0, 255), 2)

        #else:
            #print("NAN")

    #####
    cv2.putText(orig_im, str(distance), bottom_center, 0, 1e-3 * height, (0, 255, 0), 2)  # 画上距离


# 全局变量
colours = np.random.rand(32,3)*255
def track_fun(result,mot_tracker,orig_im,frame_id):
    #参数介绍
    # result = []  # 坐标框容器  result.append([c1[0],c1[1],c2[0],c2[1]])
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
        cv2.putText(orig_im, str(trackID), (lrx,lry), cv2.FONT_ITALIC, 0.6, (int(colours[trackID%32,0]),int(colours[trackID%32,1]),int(colours[trackID%32,2])),2)
        cv2.rectangle(orig_im,(lrx,lry),(rtx,rty),(int(colours[trackID%32,0]),int(colours[trackID%32,1]),int(colours[trackID%32,2])),2)
        id_num = str(trackID)  # track.track_id是跟踪的目标标号id

        lt = (lrx,lry)  # 左上点
        rt = (rtx,lry)  # 右上点
        lb = (lrx,rty)  # 左下点
        rb = (rtx,rty)  # 右下点

        cT = (lrx+int((rtx-lrx)/2),lry)  # 用于画上感叹号的位置 ！


        cv2.line(orig_im, lt, rb, (0, 255, 255),  1)  # 连接对角线
        cv2.line(orig_im, lb, rt, (0, 255, 255),  1)


        bottom_center = ((int)((rtx-lrx)/2)+lrx, rty)  # 中心点center
        cv2.circle(orig_im, bottom_center, 2, (0, 0, 255), 2, -1)  # 画上中心点


        height,width,_ = orig_im.shape

        bottom_center_X = bottom_center[0]
        bottom_center_Y = bottom_center[1]

#####
        # 下面计算距离，然后保存到csv中
        safe_distance = 50  # 安全距离暂时设置为50
        distance = distance_Car_Person(bottom_center[0], bottom_center[1], height, width)  # 计算距离distance
#####
        func(orig_im,distance,bottom_center,id_num,safe_distance,lt, rb,cT,height) #
#####
        writer_.writerow([frame_id, trackID, distance,bottom_center_X,bottom_center_Y])  # 将距离distance写入csv文件中
        csvfile_.flush()
######




def arg_parse():
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
    parser.add_argument("--video", dest = 'video', help = "Video to run detection upon", default = "kitti3.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.7)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help ="Config file",default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "cfg/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                default = "416", type = str)    # 必须是32的整数倍
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
    parser.add_argument("--video", dest='video', help="Video to run detection upon", default="kitti1.avi", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained", default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.7)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="kitti/yolov3_kitti.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="kitti/yolov3_kitti_final.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)  # 必须是32的整数倍

    # track_sort的参数
    parser.add_argument('--sort_max_age',default=5,type=int)
    parser.add_argument('--sort_min_hit',default=3,type=int)

    return parser.parse_args()


if __name__ == '__main__':

    createCSVfile()

    args = arg_parse()

    ## track_sort初始化
    mot_tracker = Sort(args.sort_max_age, args.sort_min_hit)


    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    
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
    frame_id = 0
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
            
            classes = load_classes('kitti/kitti.names') ##
            #classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

####
            result = []  # 坐标框容器
            list(map(lambda x: write(x, orig_im,result), output))

            track_fun(result, mot_tracker, orig_im, frame_id)  # 跟踪代码
####
            frame_id += 1


            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1.6
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break
    

    


