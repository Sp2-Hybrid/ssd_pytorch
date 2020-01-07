# 遮挡问题——基于SSD

## 1. 项目文件结构

 - data（存储数据集，使用VOC格式）

   - VOCdevkit
     - VOC
       - Annotations
       - ImageSets
         - Main
           - trainval.txt
           - train.txt
           - val.txt
           - test.txt
       - JPEGImages

   - __init__.py
   - voc0712.py
   - coco.py
   - config.py
   - coco_labels.txt

- eval（模型训练后进行测试）

  - Anno_test
  - Image_test
  - results
  - calculate_map_test.py
  - test.py
  - core_coreless_test.txt

- layers

- utils

- weights

- ssd.py

- test.py

- train.py（模型训练）

## 2.数据集整理

	- 按照voc数据集格式进行整理，数据集的根目录为"data/VOCdevkit"
	
	- 与voc格式稍有不同，使用txt格式的标注文件，标注文件的读取见"data/voc0712.py"中VOCDetection类
	
	- 数据集中的trainval.txt对应训练集，test.txt对应测试集

## 3. 网络结构

### 	3.1base-net  VGG网络

​			![base-net](https://github.com/Sp2-Hybrid/ssd_pytorch/blob/master/base-net.jpg)

### 	3.2 辅助特征层

​			![extra-layers](https://github.com/Sp2-Hybrid/ssd_pytorch/blob/master/extra-layers.jpg)

### 	3.3 类别预测层和边界框预测层

## 4. 训练

```python
python train.py
```

在voc0712.py中修改VOC_ROOT地址，根据需要修改相关的参数

## 5.测试

将测试所用的文件Anno_test和Image_test以及core_coreless_test.txt文件复制到eval文件夹下，运行指令：

```python
python test.py --voc_root 'eval所在的父目录'
```

在"eval/results"目录下将得到测试结果文件，接着运行：

```python
python calculate_map_test.py
```

将得到最终的计算结果。
