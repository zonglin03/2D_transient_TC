"""
    作者：李宗霖
    功能：求解二维非稳态热传导问题
    特性：可以通过调整加权因子f取值实现使用不同的时间差分格式求解瞬态二维热传导问题
    
    注：1.本程序使用有限体积法进行求解
        2.网格划分使用内节点法
        3.空间离散化使用中心差分格式
        4.物性交界面的导热系数插值使用调和平均法
        5.如果出现报错TypeError: float() argument must be a string or a real number, not 'complex'时，请重新设置更小的时间步长重新计算
"""

# 必要的运行库
import os
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import multiprocessing
import imageio.v2 as imageio
import time
import functools

# 节点属性的类
class Node_info:

    # 加权因子
    f = 0
    # 0 为显式格式
    # 1 为全隐格式
    # 0.5 为C-N格式(Crank - Nicoson)

    # 源项
    # 源项线性化：S = S_C + S_P * T_P
    S_C = 0
    S_P = 0

    # 节点初始化
    def __init__(self, node_id, x, y,
                 east_node_id, west_node_id,
                 north_node_id, south_node_id):

        self.node_id = node_id
        self.x = x
        self.y = y
        self.east_node_id = east_node_id
        self.west_node_id = west_node_id
        self.north_node_id = north_node_id
        self.south_node_id = south_node_id
        # 初始温度283K
        self.temperature = [283.0]
        self.time_step = [0.00]

    # 输出节点信息
    def __repr__(self):
        return f"Node ID: {self.node_id}, X: {self.x:.5f}, Y: {self.y:.5f},\nEast Node ID: {self.east_node_id}, West Node ID: {self.west_node_id}, North Node ID: {self.north_node_id}, South Node ID: {self.south_node_id},\nTime step:\n\t{self.time_step}, \nTemperature:\n\t{self.temperature}\n"

    # 节点控制方程系数初始化计算
    def normalize(self,  division_x, division_y,
                  division_t, densities, Cp,
                  k_east, k_west, k_south, k_north):
        self.a_east = k_east * division_y / division_x
        self.a_west = k_west * division_y / division_x
        self.a_south = k_south * division_x / division_y
        self.a_north = k_north * division_x / division_y
        self.a_persent_0 = densities * Cp * division_x * division_y / division_t

    # 节点控制方程系数初始化计算
    def a_p(self, division_x, division_y):
        self.a_persent = self.a_persent_0 + self.f * \
            (self.a_east + self.a_west + self.a_south +
             self.a_north - self.S_P * division_x * division_y)
        
    def b_p(self, division_x, division_y):
        self.b = self.S_C * division_x * division_y

    # 节点源项计算
    def S_calculate(self, division_x, division_y, ):
        if self.north_node_id == 0 and self.east_node_id == 0:
            # 右上角点
            q = 3.4e-8 * (973**4 - self.temperature[-1] ** 4)
            self.S_C = q / division_x + q / division_y
        elif self.north_node_id == 0 and self.x < 0.03:
            # 左上边界，自然对流边界
            q = 3.421 * (self.temperature[-1] - 283) ** (5/4)
            self.S_C = - q / division_y
        elif self.north_node_id == 0:
            # 右上边界，热辐射边界
            q = 3.4e-8 * (973**4 - self.temperature[-1] ** 4)
            self.S_C = q / division_y
        elif self.east_node_id == 0:
            # 左边界，热辐射边界
            q = 3.4e-8 * (973**4 - self.temperature[-1] ** 4)
            self.S_C = q / division_x


def generate_nodes(x_length, y_length, division_x, division_y):
    '''
    内节点法划分网格
    '''

    nodes = []
    x_coords = np.arange(0, x_length, division_x)
    y_coords = np.arange(0, y_length, division_y)

    for i in range(len(y_coords)):
        for j in range(len(x_coords)):

            persent_node_id = round(i * x_length/division_x + j + 1)

            east_node_id = round(i * x_length/division_x + j + 2)
            if j == len(x_coords) - 1:
                east_node_id = 0

            west_node_id = round(i * x_length/division_x + j)
            if j == 0:
                west_node_id = 0

            north_node_id = round((i + 1) * x_length/division_x + j + 1)
            if i == len(y_coords) - 1:
                north_node_id = 0

            south_node_id = round((i - 1) * x_length/division_x + j + 1)
            if i == 0:
                south_node_id = 0

            node = Node_info(persent_node_id,
                             x_coords[j] + 0.5 *
                             division_x, y_coords[i] + 0.5 * division_y,
                             east_node_id, west_node_id, north_node_id, south_node_id)
            nodes.append(node)

    print(f"网格数量{x_length/division_x:.0f}x{y_length/division_y:.0f}={len(nodes)}")
    return nodes


def Coefficient_normalize(nodes, division_x, division_y, division_t, jiao1, jiao2, jiao3, jiao4):
    """
    控制方程系数初始化
    """
    densities1 = 7830
    densities2 = 1500

    Cp1 = 465
    Cp2 = 1465

    k1 = 53.6
    k2 = 0.2093
    k3 = 2 * k1*k2/(k1+k2)

    for node in nodes:
        if node.y == 0.004 - division_y*0.5:
            # 材料一的上界面
            node.normalize(division_x, division_y, division_t,
                           densities1, Cp1, k1, k1, k1, k3)

        elif node.y == 0.004 + division_y*0.5:
            # 材料二的下界面
            node.normalize(division_x, division_y, division_t,
                           densities2, Cp2, k2, k2, k3, k2)

        elif node.y == 0.006 - division_y*0.5:
            # 材料二的上界面
            node.normalize(division_x, division_y, division_t,
                           densities2, Cp2, k2, k2, k2, k3)

        elif node.y == 0.006 + division_y*0.5:
            # 材料一的下界面
            node.normalize(division_x, division_y, division_t,
                           densities1, Cp1, k1, k1, k3, k1)

        elif node.y > 0.004 and node.y < 0.006:
            # 材料二
            node.normalize(division_x, division_y, division_t,
                           densities2, Cp2, k2, k2, k2, k2)

        else:
            # 材料一
            node.normalize(division_x, division_y, division_t,
                           densities1, Cp1, k1, k1, k1, k1)

        # 优先处理角点
        if node.node_id == jiao1:
            # 左下角点
            node.a_west = 0
            node.a_south = 0
        elif node.node_id == jiao2:
            # 左上角点
            node.a_west = 0
            node.a_north = 0
        elif node.node_id == jiao3:
            # 右下角点
            node.a_east = 0
            node.a_north = 0
        elif node.node_id == jiao4:
            # 右上角点
            node.a_east = 0
            node.a_south = 0
        elif node.west_node_id == 0:
            # 左边界，绝热
            node.a_west = 0
        elif node.south_node_id == 0:
            # 下边界，绝热
            node.a_south = 0
        elif node.east_node_id == 0:
            # 右边界，辐射
            node.a_east = 0
        elif node.north_node_id == 0:
            node.a_north = 0
    return nodes


def A_init(nodes, division_x, division_y):
    A = np.zeros((len(nodes), len(nodes)+1))
    for node in nodes:
        node.a_p(division_x, division_y)

        A[node.node_id - 1, node.node_id] = node.a_persent
        A[node.node_id - 1, node.west_node_id] = - node.f * node.a_west
        A[node.node_id - 1, node.east_node_id] = - node.f * node.a_east
        A[node.node_id - 1, node.north_node_id] = - node.f * node.a_north
        A[node.node_id - 1, node.south_node_id] = - node.f * node.a_south
    AA = np.delete(A, 0, axis=1)
    return AA

def B_init(nodes, division_x, division_y):

    B = np.zeros((len(nodes), 1))
    for node in nodes:
        node.S_calculate(division_x, division_y)
        node.b_p(division_x, division_y)
        B[node.node_id - 1] = (1 - node.f) * (node.a_west * nodes[node.west_node_id - 1].temperature[-1] + node.a_east * nodes[node.east_node_id - 1].temperature[-1] + node.a_south * nodes[node.south_node_id - 1].temperature[-1] +
                                              node.a_north * nodes[node.north_node_id - 1].temperature[-1]) + node.temperature[-1] * (node.a_persent_0 - (1 - node.f)*(node.a_west + node.a_east + node.a_south + node.a_north - node.S_P * division_x*division_y))

    return B

def B_change(node, division_x, division_y, B):
    B_i = B
    for node in nodes:
        node.S_calculate(division_x, division_y)
        node.b_p(division_x, division_y)
        B_i[node.node_id - 1] = B[node.node_id - 1] + node.b

    return B_i
def T_plot(time=-1, nodes=None, T_max=None, T_min=None, store_path=None):
    # 配置画布
    plt.clf()
    plt.figure(figsize=(10, 5), dpi=300)

    x = np.array([node.x for node in nodes])  # X坐标
    y = np.array([node.y for node in nodes])  # Y坐标
    temperatures = np.array([node.temperature[time]
                            for node in nodes])  # 对应的温度数据

    # 散点图
    # im.scatter(x, y, c=temperatures, cmap='coolwarm', s=100, edgecolors='k', linewidths=1, alpha=0.8)
    # 在每个数据点上标注数值
    # for i, txt in enumerate(temperatures):
    #    im.text(x[i], y[i], "{:.2f}".format(txt), color='black', fontsize=8, ha='center', va='center')

    # 温度云图
    im = plt.tricontourf(x, y, temperatures, levels=30, cmap="rainbow")
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Temperature (°C)')
    plt.clim(vmin=T_min, vmax=T_max)

    # 添加标题和轴标签
    plt.title(f"{nodes[0].time_step[time]:.4f}s Temperature Cloud Map")
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')

    # 保存图片
    plt.savefig(store_path + '//' + f't={nodes[0].time_step[time]:.4f}s.png')
    plt.close()

    return im


if __name__ == '__main__':

    print("################################")
    print("二维非稳态热传导问题求解程序")
    # 指定工作文件夹
    work_path = os.getcwd()
    if not os.path.basename(work_path) == '2D_transient_TC':
        work_path = os.path.join(work_path, '2D_transient_TC')
    if not os.path.exists(work_path):
        os.mkdir(work_path)
    print("工作路径: " + work_path)

    # 时间计算域
    time_length = 45
    # 时间步长设置
    time_step = 0.001
    n = round(time_length/time_step) + 1
    print("时间域: 0 ", time_length)
    print("Time step: ", time_step)

    # 空间计算域
    x_length = 0.05
    y_length = 0.01
    # 划分网格尺寸
    division_x = 0.0002
    division_y = 0.0002

    print("X方向网格尺寸", division_x)
    print("Y方向网格尺寸", division_y)

    # 生成内节点网格
    nodes = generate_nodes(x_length, y_length,
                           division_x, division_y)

    # 算法稳定性判定
    if nodes[0].f == 0:
        store_path = os.path.join(work_path, '显示格式')
        if time_step < 7830*465/2/53.6 * min(division_x, division_y)**2:
            print("显式格式满足稳定性判定准则")
        else:
            raise ValueError("显式格式不满足稳定性判定准则，请重新调整时间步长")
    elif nodes[0].f == 1:
        store_path = os.path.join(work_path, '全隐格式')
    elif nodes[0].f == 0.5:
        store_path = os.path.join(work_path, 'C-N格式')
    else:
        store_path = os.path.join(work_path, f"其他时间格式f={nodes[0].f}")

    # 图片储存文件夹位置
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    print("图片路径: " + store_path)

    # 初始化系数
    Coefficient_normalize(nodes, division_x, division_y, time_step,
                          1, round((x_length/division_x) *
                                   (y_length/division_y-1)+1),
                          len(nodes), round(x_length/division_x))

    # 计时开始
    time_start = time.time()
    # 求解线性方程组
    with tqdm(total=n, desc='In iteration') as tbar:
        A = A_init(nodes, division_x, division_y)
        B_0 = B_init(nodes, division_x, division_y)
        for i in range(n):
            B = B_change(nodes, division_x, division_y, B_0)
            time_p = round(i*time_step, 3)
            tbar.set_postfix(time=f"{time_p:.3f}s")
            tbar.update()  # 默认参数n=1，每update一次，进度+n
            Temperature = solve(A, B)
            for node in nodes:
                node.temperature.append(
                    round(float(Temperature[node.node_id - 1][0]), 3))
                node.time_step.append(time_p)


    # 计时结束
    time_end = time.time()
    print("Iteration processing time:", time_end - time_start)

    # 设置图像colorbar范围
    T_min = min(min([node.temperature for node in nodes]))
    T_max = max(max([node.temperature for node in nodes]))
    print("温度范围:", T_min, T_max)

    # 获取cpu进程数
    core = multiprocessing.cpu_count()
    partitions = int(core/2)
    print("CPU Cores num: %d" % core)
    print("Threads num: %d" % partitions)

    # 计时开始
    time_start = time.time()
    # 多核并行进程池
    pool = multiprocessing.Pool(processes=8)

    # 使用 functools.partial 创建一个新函数plot_fnc，将默认参数传递给 T_plot
    plot_fnc = functools.partial(
        T_plot, nodes=nodes, T_max=T_max, T_min=T_min, store_path=store_path)
    # 绘制第一张
    plot_fnc(0)
    # 加入tqdm的pbar，用于显示进度条
    pbar = tqdm(total=round(n/50,0), desc="Ploting")
    update = lambda *args: pbar.update()
    # 使用map_async分发数据
    for i in list(range(1, n+1, 50)):
        data = pool.map_async(plot_fnc, (i,), callback=update)
    # 关闭进程池，不在添加新进程
    pool.close()
    # 等待所有子进程结束
    pool.join()
    pbar.close()
    # 计时结束
    time_end = time.time()
    print("Plot processing time:", time_end - time_start)

    # 计时开始
    time_start = time.time()

    # 绘制GIF动态图
    img_list = []
    file_path = [f't={n*1:.4f}s.png'
                 for n in range(time_length*1+1)]
    for img in tqdm(file_path, desc='Reading pngs'):
        img_list.append(imageio.imread(os.path.join(store_path, img)))

    img_list.extend([img_list[-1]] * 10)

    print("GIF动态图保存中...")
    imageio.mimsave(store_path+"animation_f="+str(nodes[0].f)+".gif",
                    img_list, 'GIF', fps=5, loop=0)

    # 计时结束
    time_end = time.time()
    print("Animation processing time2:", time_end - time_start)
