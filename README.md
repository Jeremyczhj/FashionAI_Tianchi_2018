![](https://github.com/Jeremyczhj/FashionAI_Tianchi_2018/blob/master/datasets/3.jpg)
---
* 天池大数据竞赛——FashionAI全球挑战赛—服饰属性标签识别
* 采用多任务学习的策略，比赛最终成绩为决赛21名，在此记录一下在比赛过程中踩过的坑
* [FashionAI全球挑战赛官方链接，数据集可在此下载](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11409391.333.4.7cb749fenAbYGF&raceId=231649 "悬停显示")
* [我的CSDN链接](https://blog.csdn.net/jeremyczh/article/details/80571294 "悬停显示")
* `transfer learning`  `multitasks learning`  `keras`  `FashionAI`  `Tianchi big data`
---
### 环境
* ubuntu16.04/windows10
* python 3.6.2
* keras 2.1.6
* tensroflow 1.8.0
* opencv-python 3.4
* imgaug 0.2.5

### 使用
* 因为懒，代码写完就跑，跑完就算，没怎么优化与封装
* config.py-------------配置了样本目录等信息
* cal_std_mean.py------计算数据集的std与mean
* Multitask_train.py----训练脚本
* Multitask_predict-----预测脚本
* dataset.py------------数据预处理
* inceptionv4.py--------Inceptionv4模型API
* 来[这里](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.686b6561aZJ1xi&raceId=231649 "悬停显示")下载数据集，解压到datasets文件夹下，就可以执行`python Multitask_train.py`进行训练了


### 思路分享
* 迁移学习+多任务学习+模型融合
* 分别设计了两个多任务模型结构如下：
    * 长度类别多输出模型：
![](https://github.com/Jeremyczhj/FashionAI_Tianchi_2018/blob/master/datasets/1.png)
    * 领子类别多输出模型：  
![](https://github.com/Jeremyczhj/FashionAI_Tianchi_2018/blob/master/datasets/2.png)
* 融合Inceptionv4与Inceptionresnetv2，分别进行预测再对结果做平均
* 对测试集样本进行增广

### 提高分数的技巧
* shuffle很重要
* 多任务学习比单任务学习成绩提高`1-2%`
* 合适的图像增广，推荐使用imgaug，功能强大。dataset.py有详细代码，能提高`2-3%`
* 图像标准化，计算本数据集的std与mean，而不是直接用imagenet的std与mean，提高`0.5-1%`
* 增大图像输入尺寸可提高分类准确率，提高`1-2%`
* finetune，算力允许的前提下finetune整个模型，对比只训练最后一层提高`3-5%`
* 使用Adam先快速收敛，再用SGD慢慢调，效率会比较高
* 模型融合，提高`1-2%`
* 对测试集进行增广，本例选择了镜像，加上旋转5、10、15度进行预测，最后再取平均，提高`0.5-1%`

      
