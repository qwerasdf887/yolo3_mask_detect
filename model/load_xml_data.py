#讀取使用LabelImg生成的標記資料(VOC格式)
#並且圖像以及座標轉換至輸出size默認
#input:
#xml data path
#output shape(Opt):default(608,608)

#return:
#resize img : default(608, 608, 3)
#box: (number of box,4)
import xml.etree.cElementTree as ET
import os
import cv2
import numpy as np

def img_aug(img):
    h, w, _ = img.shape
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    #隨機一個數值如果大於0.5則將影像色度調整
    if np.random.rand() >= 0.5:
        #調整色度
        hug = np.random.randint(60)
        img[:,:,0] = (img[:,:,0] + hug)%180
    
    #隨機一個數值如果大於0.5則將影像飽和度調整
    if np.random.rand() >= 0.5:
        #調整色度
        s = np.random.randint(-30,30)
        img[:,:,1] = img[:,:,1] + s
    
    #隨機一個數值如果大於0.5則將影像亮度調整
    if np.random.rand() >= 0.5:
        #調整色度
        v = np.random.randint(-30,30)
        img[:,:,2] = img[:,:,2] + v

    img[img>255] = 255
    img[img<0] = 0
    
    img = img.astype(np.uint8)
    img  = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    return img

def load_data(name, ann_path, img_path, classes, output_shape=(416,416), jitter=.3, max_boxes=40, random=True):
    #load xml data
    tree = ET.parse(ann_path + name)
    
    root = tree.getroot()
    
    #load img
    if os.path.isfile(img_path + name[:-3] + 'jpg'):
        img = cv2.imread(img_path + name[:-3] + 'jpg')
    if os.path.isfile(img_path + name[:-3] + 'jpeg'):
        img = cv2.imread(img_path + name[:-3] + 'jpeg')
    if os.path.isfile(img_path + name[:-3] + 'png'):
        img = cv2.imread(img_path + name[:-3] + 'png')
    h, w, _ = img.shape
    
    #計算縮放至output shape的倍率
    scale = min( output_shape[0] / h, output_shape[1] / w)
    
    #隨機縮放0.7~1
    if np.random.rand() >= 0.5:
        scale *= np.random.uniform((1-jitter), 1)
    
    h = int(h * scale)
    w = int(w * scale)
    
    #resize至計算後的大小
    img = cv2.resize(img, (w, h))
    
    #是否使用image augmentation
    if random:
        img = img_aug(img)
    
    #算與output size寬高差距，並且隨機一個差距點將圖貼上，空圖部分用127取代
    dx = int(np.random.uniform(0, output_shape[1] - w))
    dy = int(np.random.uniform(0, output_shape[0] - h))
    
    new_img = np.ones((output_shape[0], output_shape[1], 3), np.uint8) * 127
    #將圖片貼上
    new_img[dy:dy+h, dx:dx+w] = img
    
    loc_list = np.zeros((max_boxes, 5))
    #load obj loc
    box_count = 0
    for obj in root.iter('object'):
        #難易度
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        
        #標記內容不再指定類別中 or 困難度=1則跳過該box
        if cls not in classes or int(difficult) == 1:
            continue
    
        #名稱對應的label index
        cls_id = classes.index(cls)
    
        #找到bounding box的兩個座標
        loc = obj.find('bndbox')
    
        x_min = int(loc.find('xmin').text) * scale + dx
        y_min = int(loc.find('ymin').text) * scale + dy
        x_max = int(loc.find('xmax').text) * scale + dx
        y_max = int(loc.find('ymax').text) * scale + dy
        
        loc_list[box_count, :] = np.array([x_min, y_min, x_max, y_max, cls_id])
    
    
    loc_list = np.array(loc_list, dtype='float32')
    
    return new_img, loc_list

def preprocess_true_boxes(true_boxes, input_shape, anchors, anchor_mask, num_classes, tiny=False):
    
    #output layer數，tiny:2，ori:3
    if tiny:
        num_layers = 2
    else:
        num_layers = 3
    
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    
    #x,y中心點
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    #w,h值
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    #除以寬高得到歸一化值
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]
    
    #batch size 數目
    m = true_boxes.shape[0]
    #3個預測輸出map的size
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    
    # 擴展維度已進行 broadcasting機制.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    
    #假如標記錯誤 or 寬高為0 則忽略該框
    valid_mask_w = boxes_wh[..., 0] > 0
    valid_mask_h = boxes_wh[..., 1] > 0
    valid_mask = valid_mask_w * valid_mask_h
    
    for b in range(m):
        # 消除標記錯誤的box
        wh = boxes_wh[b, valid_mask[b]]
        
        if len(wh)==0: continue
        # 擴展維度已進行 broadcasting機制.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        
        #利用IOU挑選最適合的anchors
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1
    return y_true