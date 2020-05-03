# coding=UTF-8
# This Python file uses the following encoding: utf-8

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import cv2
from model.models import Darknet_body, DarknetTiny_body
from model.loss import YoLoLoss
from model.load_xml_data import load_data, preprocess_true_boxes
from model.utils import get_classes, get_anchors
import os
from tensorflow.keras.utils import Sequence
import math
from tensorflow.keras.mixed_precision import experimental as mixed_precision
tf.keras.backend.set_learning_phase(1)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def main():
    #input shape要為32的倍數，因為5次downsampling
    input_shape = (608, 608)
    annotation_path = 'medical-masks-dataset/labels/'
    image_path = 'medical-masks-dataset/images/'
    log_dir = './'
    classes_path = './data/mask_classes.txt'
    anchors_path = './data/yolo_anchors.txt'
    class_name = get_classes(classes_path)
    #class數量
    num_classes = len(class_name)
    anchors = get_anchors(anchors_path) / input_shape[::-1]
    num_anchors = len(anchors)
    is_tiny_version = False
    data_name = os.listdir(annotation_path)
    print('anchors:', anchors)

    #creat model
    if is_tiny_version:
        model = DarknetTiny_body(input_shape, num_anchors, num_classes)
        anchor_mask = [[3,4,5], [0,1,2]]
        
    else:
        model = Darknet_body(input_shape, num_anchors, num_classes)
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        model.load_weights('yolo.h5')
        
    loss = [YoLoLoss(input_shape, anchors[mask], classes=num_classes) for mask in anchor_mask]
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath= log_dir + 'best_loss.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    verbose=1)]
    batch_size = 4
    num_step = math.ceil(len(data_name) / batch_size)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer=opt, loss=loss)
    model.summary()
    model.fit(imgSequence(data_name, annotation_path, image_path, batch_size, input_shape, anchors,
                          anchor_mask, num_classes, class_name, is_tiny_version),
              steps_per_epoch = num_step, callbacks=callbacks, epochs = 200)

class imgSequence(Sequence):
    def __init__(self, xml_name, ann_path, img_path, batch_size, input_shape, anchors, anchor_mask,
                 num_classes, class_name, is_tiny_version):
        self.xml_name = xml_name
        self.ann_path = ann_path
        self.img_path = img_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.class_name = class_name
        self.is_tiny = is_tiny_version
    
    def __len__(self):
        return math.ceil(len(self.xml_name) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_name = self.xml_name[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        image_data, label = self.creat_data(batch_name)
        
        return image_data, label, [None, None, None]
        
    def on_epoch_end(self):
        np.random.shuffle(self.xml_name)
        
    def creat_data(self, batch_data):
        image_data = []
        box_data = []
        
        for i in range(len(batch_data)):
            image, box = load_data(batch_data[i], self.ann_path, self.img_path, self.class_name, output_shape=self.input_shape)
            image_data.append(image)
            box_data.append(box)
            
        image_data = (np.array(image_data) / 255).astype(np.float32)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.anchor_mask,
                                       self.num_classes, tiny=self.is_tiny)
        
        return image_data, y_true
# In[6]:

if __name__ == '__main__':
    main()



