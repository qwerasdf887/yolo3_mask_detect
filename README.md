# yolo3_mask_detect
根據kaggle提供的dataset訓練的口罩偵測程式

## Train data 來源

[Kaggle Medical Masks Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset)

## 環境:

1. Tensorflow 2.1
2. Python 3.5~3.7
3. OpenCV 3~4

## h5 weight

[h5 weight](https://drive.google.com/file/d/1yUIntmmEdBWiGHoagWiR2WAVpGErlQk0/view?usp=sharing)

## demo video

[video](https://drive.google.com/file/d/1_DMYV3FriaU3FCcmsSsRIWOeihkAtpAc/view?usp=sharing)

## 更改detect.py中影像路徑即可觀看運行結果

影片位置(可改為網路攝影機等設備)
```bashrc
    video = cv2.VideoCapture('./mask.mp4')
```
## Training
>至Kaggle下載檔案並且解壓縮至目錄下
>修改train_on_xml.py相對應的一些資訊即可自行訓練
>最好使用pre-trained好的weight
>可使用converty.py加上yolov3本身修改好的cfg得到對應的.h5權重
```bashrc
python convert.py yolov3.cfg yolov3.weights yolo.h5
```

## 目前缺點
>因為training data沒有畫面佔比大的train data，所以佔比大的偵測不到。