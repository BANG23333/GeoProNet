#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from osgeo import gdal
import geopandas as gpd
import fiona
from matplotlib import pyplot as plt
from shapely.geometry import LineString
import numpy as np

data = gpd.read_file(r'D:\NYC_data\NYC Street Centerline\geo_export_b407cb84-674b-436b-bf3c-0fab9b97aa74.shp')#读取磁盘上的矢量文件
#data = gpd.read_file('shapefile/china.gdb', layer='province')#读取gdb中的矢量数据
#print(data.crs)  # 查看数据对应的投影信息
#print(data.head())  # 查看前5行数据
data.plot()
plt.figure(figsize=(12,12))
plt.show()#简单展示


# In[ ]:


x_max = -73.66301
x_min = -74.27480125628799
y_max = 40.94787907007066
y_min = 40.453256660128865
x_len = 68
y_len = 55


# In[ ]:


def locator(lat, lon):
    global x_min, y_min, x_max, y_max, x_len, y_len
    Y, X = -1, -1
    for i in range(1, y_len+1):
        temp_lat = y_min + i*(y_max - y_min)/y_len
        if lat <= temp_lat:
            Y = i
            break

    for j in range(1, x_len+1):
        temp_lon = x_min + j*(x_max - x_min)/x_len
        if lon <= temp_lon:
            X = j
            break

    return Y-1, X-1


# In[ ]:


#------------------------ 生成邻接矩阵-----------------
x_len = 68
y_len = 55
Nodes = x_len * y_len
adj_matrix = np.zeros((Nodes,Nodes))

max_len = 0
for _, road in data.iterrows():
    road_geom = road['geometry']  # 获取道路几何对象
    road_length = road['shape_leng']  # 获取道路长度
    start_x = road_geom.xy[0][0]  #起点经纬度
    start_y = road_geom.xy[1][0]
    end_x = road_geom.xy[0][-1]  #终点经纬度
    end_y = road_geom.xy[1][-1]
    start_Y,start_X = locator(start_y, start_x)  #起点、终点 grid[x][y] 
    end_Y,end_X = locator(end_y, end_x)
    #68 * 55的地图离散为点，根据所在位置X,Y    编号为X*55+Y
    start_id = start_X*55 + start_Y
    end_id = end_X*55 + end_Y
    if start_id > end_id:
        adj_matrix[start_id][end_id] = road_length
    else:
        adj_matrix[end_id][start_id] = road_length
    if road_length >max_len:
        max_len = road_length 
    print(start_id, end_id,road_length)
for i in range(68*55):
    for j in range(i):
        adj_matrix[j][i]=adj_matrix[i][j] 
np.save(r'D:\NYC_crash_data\NYC_data\adj_matrix.npy',adj_matrix)
print(max_len)


# In[ ]:


#---------------------  根据mask记录有效点-------------------
mask = np.load(r'D:\NYC_crash_data\68_55_mask.npy')   #

# 记录mask上值大于等于1的点坐标
coordinates = np.argwhere(mask >= 1)

num_points = len(coordinates)
output_file_path = r'D:\NYC_crash_data\coordinates.txt'   #
print(num_points)

with open(output_file_path, 'w') as file:
    file.write(f"Number of points with value >= 1: {num_points}\n")
    file.write("Coordinates of points with value >= 1:\n")
    for coord in coordinates:
        file.write(f"({coord[0]}, {coord[1]})\n")

print(f"Results saved to: {output_file_path}")


# In[ ]:


#----------------  缩小邻接矩阵规模---------------------------

# 加载原始的(3740, 3740)邻接矩阵
adj = np.load(r'D:\NYC_crash_data\NYC_data\adj_matrix.npy')  #
nonzero_count = np.count_nonzero(adj)
print("Number of non-zero elements in the adjacency matrix:", nonzero_count)


# 加载有用点的坐标信息
with open(r'D:\NYC_crash_data\coordinates.txt', 'r') as file:
    lines = file.readlines()
    coordinates = [tuple(map(int, line.strip()[1:-1].split(','))) for line in lines[2:]]

# 提取相应点的邻接矩阵
selected_adjacency_matrix = np.zeros((1128, 1128))
for i, coord_i in enumerate(coordinates):
    for j, coord_j in enumerate(coordinates):
        selected_adjacency_matrix[i, j] = adj[coord_i[0]*55 + coord_i[1], coord_j[0]*55 + coord_j[1]]

# 验证提取后的邻接矩阵的形状
print("提取后的邻接矩阵形状：", selected_adjacency_matrix.shape)
np.save(r'D:\NYC_crash_data\NYC_data\adj_1128.npy',selected_adjacency_matrix)  #

