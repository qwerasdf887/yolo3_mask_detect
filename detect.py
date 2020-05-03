# coding=UTF-8
# This Python file uses the following encoding: utf-8

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import cv2
from model.models import Darknet_body
from model.utils import predict_box, draw_box, get_classes, get_anchors

if __name__ == '__main__':

    #creat model
    input_shape = (608,608)
    classes_path = './data/mask_classes.txt'
    class_name = get_classes(classes_path)
    num_classes = len(class_name)
    anchors_path = './data/yolo_anchors.txt'
    anchors = get_anchors(anchors_path) / input_shape[::-1]
    num_anchors = len(anchors)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    model = Darknet_body(input_shape, num_anchors, num_classes)

    model.load_weights('./best_loss.h5')
    #model.summary()
    
    #影片位置(可改為網路攝影機等設備)
    video = cv2.VideoCapture('./mask.mp4')

    #影像參數
    width = int(video.get(3))
    height = int(video.get(4))
    fps = int(video.get(5))

    print('width:{}\nheight:{}\nFPS:{}'. format(width, height, fps))
    
    #等比例縮放參數
    scale = min( input_shape[1] / width, input_shape[0] / height)
    new_w = int(scale * width)
    new_h = int(scale * height)

    dx = (input_shape[1] - new_w) // 2
    dy = (input_shape[0] - new_h) // 2

    color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    
    #寫入成影片
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, input_shape[::-1])

    while True:
        ret, frame = video.read()
        if ret == False:
            break
        #建立一個與model input shape相同大小的背景板
        img = np.ones((input_shape[0], input_shape[1], 3), dtype=np.uint8) * 127
        img[dy: dy + new_h, dx: dx + new_w, :] = cv2.resize(frame, (new_w, new_h))
        det_result = model.predict(np.expand_dims(img, axis=0) / 255)

        boxes, cls, score = predict_box(det_result, input_shape, anchors, anchor_mask)
        draw_box(img, boxes, cls, class_name, score, color)

        #寫入影像
        out.write(img)

        cv2.imshow('video', img)
        cv2.waitKey(1)

    video.release()
    out.release()
    cv2.destroyAllWindows()
    
    