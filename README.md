# 基于改进U-Net的水下鱼类图像语义分割应用研究

## 文件结构
在根目录下：

**./src/：** 源码

**./trainval.py：** 训练脚本

**./test.py：** 在测试集上测试的脚本

**./test_one_image.py：** 在单张输入图像上测试的脚本

在src目录下：

**./src/datasets：** 数据集处理有关代码

**./src/models：** 模型定义有关代码

**./src/training：** 训练、测试有关代码

## 数据集
目前只能处理DeepFish分割数据集，但留有方便实现其它数据集的接口（**./src/datasets**）

自行下载DeepFish数据集中的Seg部分之后，解压在某个目录下，在后续执行命令时指定数据集位置即可

在下面的训练和测试指令中所指定的数据集目录下必须直接包含images、masks等解压后的目录和文件

## 模型
已实现的模型与下面命令中应指定的模型名称之间的对应列在下方，冒号前是模型名称：


参考的经典模型：

unet：朴素的U-Net实现，参考论文[1]的结构

fcn8：DeepFish数据集论文[2]中使用的FCN8（Backbone为ResNet）结构


自主实现的模型：

unet_resnet：DeepFish U-Net (Res)

unet_cgm：DeepFish U-Net (Res+CGM)

attention_unet：DeepFish U-Net (Res+Att)

attention_unet_cgm：DeepFish U-Net (Res+Att+CGM)


## 训练
**示例训练命令：**
```
python trainval.py -d "./Deepfish_Segmentation" -m "./Experiments/UNet" -n "unet" -e 100 -b 8 -re True
```

**该示例命令的行为：**

以特定参数进行训练，训练的checkpoint（内含模型参数和其它训练信息，比如损失函数值随Epoch变化的记录）会以.tar文件的形式保存在-m参数指定的位置


**常用命令行参数：**

-d：数据集位置，**必须在命令中指定**

-m：模型和实验输出位置，**必须在命令中指定**

-n：模型名称，**默认值为"unet"**

-e：训练Epoch数，**默认值为1**

-b：训练Batch Size，**默认值为1**

-re：是否重新训练，**默认值为False**，如果False就从-m参数指定的位置读取checkpoint.tar继续训练


**其它命令行参数：**

-lr：学习率，**默认值为0.001**

-rs：是否将输入图像resize为512*512后再输入网络，**默认值为True**

-sv：是否保存训练过程中的Loss图像，保存位置为-m参数指定的位置，**默认值为True**

-si：在保存最后训练结果的基础上，每隔多少Epoch额外保存一个checkpoint，**默认值为100**

## 在测试集上测试
**示例测试命令：**
```
python test.py -d "./Deepfish_Segmentation" -m "./Experiments/UNet" -n "unet"
```

**该示例命令的行为：**

以特定参数在整个测试集上进行测试，测试结果（mIoU等量化指标）输出在终端

**常用命令行参数：**

-d：数据集位置，**必须在命令中指定**

-m：模型位置，**必须在命令中指定**

-n：模型名称，**默认值为"unet"**

**其它命令行参数：**

-b：测试使用的Batch Size，**默认值为1**

-rs：是否将输入图像resize为512*512后再输入网络，**默认值为True**

-sp：测试集分区名，**默认值为"test"**，如果想要在训练或验证集上测试可以选择"train"或"valid"

-p：是否纯粹使用模型参数进行测试，**默认值为False**，如果True则会从-m指定位置读取model_state_dict.pth作为测试的模型参数，否则会读取checkpoint.tar（既包含模型参数，也包含其它训练信息）

## 在单张图像上测试
**示例测试命令：**
```
python test_one_image.py -d "./Deepfish_Segmentation" -m "./Experiments/UNet" -n "unet" -i 0 -r "./test_result"
```

**该示例命令的行为：**

以特定参数在**测试集中的单张图像**上进行测试，暂不支持自己使用图像进行测试，测试结果（mIoU等量化指标）输出在终端，模型输出的分割掩码图像存储在-r参数指定的位置

**常用命令行参数：**

-d：数据集位置，**必须在命令中指定**

-m：模型位置，**必须在命令中指定**

-n：模型名称，**默认值为"unet"**

-i：测试图像在测试集中的位置（index），**默认值为0**，从0开始计算

-r：图像输出位置

**其它命令行参数：**

-th：决定输出掩码中一个像素属不属于鱼类的Threshold，**默认值为0.5**，认为一个像素属于鱼类当且仅当模型输出的概率大于Threshold

-rs：是否将输入图像resize为512*512后再输入网络，**默认值为True**

-p：是否纯粹使用模型参数进行测试，**默认值为False**，如果True则会从-m指定位置读取model_state_dict.pth作为测试的模型参数，否则会读取checkpoint.tar（既包含模型参数，也包含其它训练信息）

## 参考文献

[1] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015: 234-241.

[2] Saleh A, Laradji I H, Konovalov D A, et al. A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis[J]. Scientific Reports, 2020, 10(1): 14671.
