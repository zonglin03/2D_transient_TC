# 2D_transient_TC

## 概述

本项目是一个使用PyTorch求解二维瞬态热传导问题。

## 问题描述

如图1所示，材料1和材料2组成三明治结构，其中右端处于973k的燃气中，下边界和左边界为绝热边界，上边界左段处于大气中，大气温度为283 k，求5s、15s、30s和60s时刻计算域内温度分布。
<img src="./torch/title_1.png" width = "500" height = "300" />  
其中
<img src="./torch/title_2.jpg" width="500" height="180" />


## 依赖关系

列出运行项目所需的所有依赖项。这可能包括Python库，如PyTorch、NumPy、Matplotlibos、tqdm、multiprocessing、imageio、functools等。

```bash
pip install -r reqirements.txt
```

## 使用方法

调整参数时，务必结合物理实际，否者结果可能会不可控

```bash
python main.py
```

## 算法概述
使用有限体积法和中心差分的方法，将传热的微分方程转换为差分形式的线性方程组进行传热计算。\
程序中使用的GPU和并行池的方法进行加速计算。\
同时，使用tqdm是代码运行更美观。

## 代码结构
- **Node\_info** 存储每个节点属性的类,\
包括但不限于加权因子f 、各种系数、历史时间与温度列表等，\
以及一些批量处理函数__init__(节点类定义的初始化), \_\_repr\_\_(输出样式),normalize(各类系数初始化函数),a_p (A系数计算) S_calculate(源项计算函数) 
- **generate_nodes()**  使用内节点法划分网格
- **def Coefficient_normalize()** 控制方程系数初始化
- **A_init()** 矩阵A的计算函数 
- **B_init()** 源项系数B的计算函数 
- **T_plot()** 温度云图绘制函数 
- **data_write_csv()** 将数据写成csv文件保存的函数


## 运行截图

- 显示格式报错截图\
<img src="./run/显式报错.png" width = "500" height = "90" />  

- 全隐格式运行\
<img src="./run/全隐格式.png" width = "500" height = "100" />  


## 运行结果

- 全隐格式，f=1\
<img src="./torch/全隐格式animation_f=1.gif" width = "500" height = "200" />  

- C-N格式，f=0.5\
<img src="./torch/C-N格式animation_f=0.5.gif" width = "500" height = "200" />  

- 全隐格式，f=1\
<img src="./torch/显示格式animation_f=0.gif" width = "500" height = "200" />  

## 参考文献

- 田瑞峰，刘平安. 传热与流体流动的数值计算[M/OL]. 哈尔滨工程大学出版社, 2015[2024-01-14].
- 李人宪. 有限体积法基础[M/OL]. 国防工业出版社, 2008[2024-01-14]. 
- 陶文铨. 传热学：第五版[M/OL]. 高等教育出版社, 2019[2024-01-14]. 
- 阿斯顿·张（ASTON ZHANG）, 李沐（MU LI）, [美]扎卡里·C. 立顿（ZACHARY C. LIPTON）, 等. 动手学深度学习[M/OL]. 何孝霆（XIAOTING HE）, 瑞潮儿·胡（RACHEL HU）, 译. PyTorch版. 人民邮电出版社, 2023[2023-07-11]. 
