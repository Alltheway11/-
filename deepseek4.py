import sys
import numpy as np
import cv2
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,  
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QListWidget, QInputDialog, QColorDialog, 
                            QMessageBox, QMenu, QListWidgetItem, QAction)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor, QDrag, QBrush
from PyQt5.QtCore import Qt, QPoint, QRect, QMimeData, QByteArray, QDataStream, QIODevice

class PerspectiveLayer:


    """图层类，存储每个图层的信息"""
    def __init__(self, name, image=None):
        self.name = name
        self.original_image = image  # 原始图像
        self.warped_image = image    # 透视变换后的图像
        self.visible = True          # 是否可见
        self.points = []             # 透视变换的控制点（用于计算消失点）
        self.warp_points = []        # 粘贴内容的变换控制点
        self.z_order = 0             # 图层顺序
        self.opacity = 1.0           # 不透明度
        self.selection_points = []   # 选区四边形的四个点
        self.position = QPoint(0, 0) # 图层位置
        self.source_vp = []          # 关联的消失点（用于2点透视约束）
        self.original_width = 0      # 原始宽度（用于透视约束）
        self.original_height = 0     # 原始高度（用于透视约束）
        self.drag_offset = QPoint(0, 0) # 拖拽偏移量
        self.layer_drag_mode = False # 图层拖拽模式
        self.layer_scale = 1.0       # 图层缩放比例
        self.layer_position = QPoint(0, 0) # 图层拖拽位置
        self.drag_layer_image = None # 拖拽图层图像
        self.initial_center = QPoint(0, 0)  # 初始中心点
        self.initial_vp2_distance = 1.0  # 初始到VP2的距离

class PerspectiveGrid:
    """透视网格类，管理透视网格的绘制和消失点计算"""
    def __init__(self):
        self.vanishing_points = []  # 消失点
        self.grid_color = QColor(255, 0, 0, 100)  # 网格颜色
        self.grid_size = 50  # 网格密度
        self.line_count = 15  # 从消失点发出的线数量
        self.enabled = True  # 是否显示网格
        self.primary_vp = []  # 主要消失点（用于2点透视）
        
    def calculate_two_point_perspective(self, points):
        """计算2点透视的消失点"""
        if len(points) != 4:
            return []
            
        # 提取四个点，假设是一个矩形在2点透视下的投影
        p1, p2, p3, p4 = points
        
        # 计算两组平行线的交点作为消失点（2点透视）
        vp1 = self.line_intersection(p1, p2, p3, p4)  # 水平方向消失点
        vp2 = self.line_intersection(p2, p3, p4, p1)  # 垂直方向消失点
        
        # 验证消失点有效性
        valid_vp1 = not (abs(vp1.x()) < 10 and abs(vp1.y()) < 10)
        valid_vp2 = not (abs(vp2.x()) < 10 and abs(vp2.y()) < 10)
        
        self.primary_vp = [vp1, vp2] if valid_vp1 and valid_vp2 else []
        return self.primary_vp
    
    def line_intersection(self, a1, a2, b1, b2):
        """计算两条线的交点"""
        x1, y1 = a1.x(), a1.y()
        x2, y2 = a2.x(), a2.y()
        x3, y3 = b1.x(), b1.y()
        x4, y4 = b2.x(), b2.y()
        
        # 计算分母
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return QPoint(10000, 10000)  # 平行线，返回远处点
        
        # 计算交点
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = t_num / den
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return QPoint(int(x), int(y))
    
    def calculate_radial_lines(self, vp, width, height, control_points):
        """计算从消失点发出的辐射线 - 只连接到控制点，不生成额外延长线"""
        lines = []
        vp_x, vp_y = vp.x(), vp.y()
        
        # 只添加连接消失点和每个控制点的线
        for pt in control_points:
            lines.append((QPoint(vp_x, vp_y), pt))
            
        return lines

class Canvas(QLabel):
    """绘图区域类，负责显示和处理图像"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.pixmap = QPixmap()
        self.scale_factor = 1.0  # 缩放因子
        self.offset = QPoint(0, 0)  # 偏移量
        self.dragging_view = False  # 是否正在拖拽视图
        self.last_drag_pos = QPoint()
        self.dragging_point = -1  # 正在拖动的点索引
        self.dragging_edge = -1  # 正在拖动的边索引
        self.dragging_layer = -1  # 正在拖动的图层索引
        self.edge_drag_offset = QPoint()  # 拖动边时的偏移量
        
        # 设置右键菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
    def show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu()
        layer_drag_action = QAction("图层拖拽", self)
        
        layer_drag_action.triggered.connect(self.toggle_layer_drag_mode)
        
        menu.addAction(layer_drag_action)
        menu.exec_(self.mapToGlobal(position))
    
    def load_image(self, image_path):
        """加载图像"""
        self.pixmap.load(image_path)
        self.update()
        
    def paintEvent(self, event):
        """绘制事件"""
        super().paintEvent(event)
        if self.pixmap.isNull():
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 应用缩放和偏移
        painter.translate(self.offset)
        painter.scale(self.scale_factor, self.scale_factor)
        
        # 绘制背景图像
        if not self.pixmap.isNull():
            painter.drawPixmap(0, 0, self.pixmap)
        
        # 绘制图层（按z_order排序）
        self.parent.draw_layers(painter)
        
        # 绘制透视网格
        if self.parent.grid.enabled:
            self.draw_grid(painter)
        
        # 如果不是图层拖拽模式，绘制选区
        current_layer = self.parent.get_current_layer()
        if not current_layer or not current_layer.layer_drag_mode:
            self.draw_selection(painter)
        
        # 绘制点和消失点
        self.draw_points(painter)
    
    def draw_selection(self, painter):
        """绘制四边形选区"""
        current_layer = self.parent.get_current_layer()
        if not current_layer or len(current_layer.selection_points) < 4:
            return
            
        # 绘制选区边界
        pen = QPen(QColor(0, 255, 255, 200), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 255, 255, 30))  # 半透明青色
        
        # 绘制四边形
        points = current_layer.selection_points
        for i in range(4):
            painter.drawLine(points[i], points[(i+1)%4])
        
        # 填充选区
        painter.drawPolygon([p for p in points])
        
        # 绘制控制点（顶点）- 改为圆点，半径为原来的1/4
        pen = QPen(QColor(255, 0, 0), 1)
        brush = QBrush(QColor(255, 0, 0, 200))
        painter.setPen(pen)
        painter.setBrush(brush)
        for point in points:
            painter.drawEllipse(point, 1, 1)  # 选区控制点半径变为1.25
    
    def draw_points(self, painter):
        """绘制控制点和消失点"""
        # 绘制当前图层的控制点（绿色圆点）- 半径为原来的1/4
        current_layer = self.parent.get_current_layer()
        if current_layer and not current_layer.layer_drag_mode:
            pen = QPen(QColor(0, 255, 0), 1)  # 绿色边框
            brush = QBrush(QColor(0, 255, 0, 200))  # 半透明绿色填充
            painter.setPen(pen)
            painter.setBrush(brush)
            
            for point in current_layer.points:
                # 绘制圆点，半径为1像素
                painter.drawEllipse(point, 1, 1)
                
                # 绘制点的编号
                painter.drawText(point.x() + 3, point.y() - 3, 
                                str(current_layer.points.index(point) + 1))
        
        # 绘制粘贴图层的变换控制点（蓝色圆点）- 半径为原来的1/4
        for i, layer in enumerate(self.parent.layers):
            if layer.warp_points and len(layer.warp_points) == 4 and layer.visible and not layer.layer_drag_mode:
                pen = QPen(QColor(0, 0, 255), 1)  # 蓝色边框
                brush = QBrush(QColor(0, 0, 255, 200))  # 半透明蓝色填充
                painter.setPen(pen)
                painter.setBrush(brush)
                
                for point in layer.warp_points:
                    # 绘制圆点，半径为1像素
                    painter.drawEllipse(point, 1, 1)
        
        # 绘制消失点（2点透视专用颜色）- 改为圆点，半径为原来的1/4
        if self.parent.grid.primary_vp:
            # 第一个消失点（红色圆点）- 水平方向
            pen = QPen(QColor(255, 0, 0), 1)
            brush = QBrush(QColor(255, 0, 0, 200))
            painter.setPen(pen)
            painter.setBrush(brush)
            vp1 = self.parent.grid.primary_vp[0]
            painter.drawEllipse(vp1, 1.5, 1.5)  # 消失点半径变为1.5
            painter.drawText(vp1.x() + 3, vp1.y() - 3, "VP1 (水平)")
            
            # 第二个消失点（蓝色圆点）- 深度方向
            if len(self.parent.grid.primary_vp) > 1:
                pen = QPen(QColor(0, 0, 255), 1)
                brush = QBrush(QColor(0, 0, 255, 200))
                painter.setPen(pen)
                painter.setBrush(brush)
                vp2 = self.parent.grid.primary_vp[1]
                painter.drawEllipse(vp2, 1.5, 1.5)
                painter.drawText(vp2.x() + 3, vp2.y() - 3, "VP2 (深度)")
    
    def draw_grid(self, painter):
        """绘制2点透视网格 - 只绘制连接到控制点的线"""
        current_layer = self.parent.get_current_layer()
        if not current_layer or len(current_layer.points) < 4 or not self.parent.grid.primary_vp:
            return
            
        # 获取图像尺寸
        img_size = (self.pixmap.width(), self.pixmap.height())
        
        # 为两个消失点绘制网格线
        if len(self.parent.grid.primary_vp) >= 2:
            vp1, vp2 = self.parent.grid.primary_vp[:2]
            
            # 从VP1发出的线（红色）- 水平方向，只连接到控制点
            lines1 = self.parent.grid.calculate_radial_lines(vp1, img_size[0], img_size[1], current_layer.points)
            pen = QPen(QColor(255, 0, 0, 100), 1)
            painter.setPen(pen)
            for line in lines1:
                painter.drawLine(line[0], line[1])
                
            # 从VP2发出的线（蓝色）- 深度方向，只连接到控制点
            lines2 = self.parent.grid.calculate_radial_lines(vp2, img_size[0], img_size[1], current_layer.points)
            pen = QPen(QColor(0, 0, 255, 100), 1)
            painter.setPen(pen)
            for line in lines2:
                painter.drawLine(line[0], line[1])
    
    def distance_point_to_line(self, point, line_start, line_end):
        """计算点到直线的距离"""
        x, y = point.x(), point.y()
        x1, y1 = line_start.x(), line_start.y()
        x2, y2 = line_end.x(), line_end.y()
        
        # 直线方程：ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        # 计算距离
        distance = abs(a*x + b*y + c) / np.sqrt(a**2 + b**2) if (a**2 + b**2) > 0 else float('inf')
        return distance
    
    def find_closest_point(self, pos):
        """找到离pos最近的选区点"""
        current_layer = self.parent.get_current_layer()
        if not current_layer or len(current_layer.selection_points) < 4:
            return -1
            
        min_dist = 10  # 最小距离阈值（像素）
        closest_idx = -1
        
        for i, point in enumerate(current_layer.selection_points):
            dist = np.hypot(point.x() - pos.x(), point.y() - pos.y())
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx
    
    def find_closest_warp_point(self, pos):
        """找到离pos最近的变换控制点"""
        current_layer = self.parent.get_current_layer()
        if not current_layer or len(current_layer.warp_points) < 4:
            return -1
            
        min_dist = 10  # 最小距离阈值（像素）
        closest_idx = -1
        
        for i, point in enumerate(current_layer.warp_points):
            dist = np.hypot(point.x() - pos.x(), point.y() - pos.y())
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx
    
    def find_closest_control_point(self, pos):
        """找到离pos最近的透视控制点（绿色点）"""
        current_layer = self.parent.get_current_layer()
        if not current_layer or len(current_layer.points) < 1:
            return -1
            
        min_dist = 10  # 最小距离阈值（像素）
        closest_idx = -1
        
        for i, point in enumerate(current_layer.points):
            dist = np.hypot(point.x() - pos.x(), point.y() - pos.y())
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx
    
    def find_closest_edge(self, pos):
        """找到离pos最近的选区边"""
        current_layer = self.parent.get_current_layer()
        if not current_layer or len(current_layer.selection_points) < 4:
            return -1, QPoint()
            
        min_dist = 10  # 最小距离阈值（像素）
        closest_idx = -1
        closest_point = QPoint()
        
        for i in range(4):
            p1 = current_layer.selection_points[i]
            p2 = current_layer.selection_points[(i+1)%4]
            
            dist = self.distance_point_to_line(pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
                # 计算投影点
                x, y = pos.x(), pos.y()
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()
                
                t = ((x - x1)*(x2 - x1) + (y - y1)*(y2 - y1)) / \
                    ((x2 - x1)**2 + (y2 - y1)** 2 + 1e-8)
                t = max(0, min(1, t))
                
                proj_x = x1 + t*(x2 - x1)
                proj_y = y1 + t*(y2 - y1)
                closest_point = QPoint(int(proj_x), int(proj_y))
        
        return closest_idx, closest_point
    
    def find_layer_at_pos(self, pos):
        """找到pos位置的图层"""
        # 从顶层往下找
        for i in reversed(range(len(self.parent.layers))):
            layer = self.parent.layers[i]
            if layer.visible and layer.warped_image:
                # 检查点是否在图层范围内
                if (0 <= pos.x() - layer.position.x() < layer.warped_image.width() and
                    0 <= pos.y() - layer.position.y() < layer.warped_image.height()):
                    return i
        return -1
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        current_layer = self.parent.get_current_layer()
        if not current_layer:
            return
            
        # 转换坐标考虑缩放和偏移
        scene_pos = self.transform_pos(event.pos())
        
        if event.button() == Qt.LeftButton:
            # 检查是否在图层拖拽模式
            if current_layer.layer_drag_mode and current_layer.drag_layer_image:
                # 检查是否点击了图层
                if self.is_point_in_layer(scene_pos, current_layer):
                    # 准备拖动图层
                    self.dragging_layer = self.parent.current_layer_idx
                    self.dragging_view = True
                    current_layer.drag_offset = QPoint(
                        scene_pos.x() - current_layer.layer_position.x(),
                        scene_pos.y() - current_layer.layer_position.y()
                    )
                    return
                else:
                    # 点击图层外部，退出图层拖拽模式
                    current_layer.layer_drag_mode = False
                    self.update()
                    return
            
            # 检查是否在控制点模式
            elif self.parent.control_point_mode:
                # 检查是否点击了已有的控制点（用于拖动）
                control_idx = self.find_closest_control_point(scene_pos)
                if control_idx != -1:
                    self.dragging_point = control_idx
                    self.dragging_view = True
                    return
                
                # 检查是否点击了选区的点
                point_idx = self.find_closest_point(scene_pos)
                if point_idx != -1:
                    self.dragging_point = point_idx
                    self.dragging_view = True
                    return
                    
                # 检查是否点击了选区的边
                edge_idx, _ = self.find_closest_edge(scene_pos)
                if edge_idx != -1:
                    self.dragging_edge = edge_idx
                    self.dragging_view = True
                    # 计算鼠标到边的偏移
                    p1 = current_layer.selection_points[edge_idx]
                    p2 = current_layer.selection_points[(edge_idx+1)%4]
                    dist = self.distance_point_to_line(scene_pos, p1, p2)
                    
                    # 计算方向向量（垂直于边）
                    dx = p2.y() - p1.y()
                    dy = p1.x() - p2.x()
                    if dx != 0 or dy != 0:
                        length = np.sqrt(dx**2 + dy**2)
                        dx_normalized = dx / length
                        dy_normalized = dy / length
                        
                        # 计算偏移方向
                        cross = (scene_pos.x() - p1.x()) * dy - (scene_pos.y() - p1.y()) * dx
                        direction = 1 if cross > 0 else -1
                        
                        self.edge_drag_offset = QPoint(
                            int(dx_normalized * dist * direction),
                            int(dy_normalized * dist * direction)
                        )
                    return
                
                # 检查是否点击了粘贴图层的变换控制点
                warp_idx = self.find_closest_warp_point(scene_pos)
                if warp_idx != -1:
                    self.dragging_point = warp_idx
                    self.dragging_view = True
                    return
                
                # 添加新的控制点（最多4个）
                if len(current_layer.points) < 4:
                    current_layer.points.append(scene_pos)
                    
                    # 同时设置为选区点
                    if len(current_layer.selection_points) < 4:
                        current_layer.selection_points.append(scene_pos)
                    
                    self.update()
                    
                    # 如果已添加4个点，自动退出控制点模式
                    if len(current_layer.points) == 4:
                        self.parent.toggle_control_point_mode()
                return
            else:
                # 检查是否点击了图层
                layer_idx = self.find_layer_at_pos(scene_pos)
                if layer_idx != -1:
                    # 选中该图层
                    self.parent.current_layer_idx = layer_idx
                    self.parent.layer_list.setCurrentRow(layer_idx)
                    
                    # 准备拖动图层
                    self.dragging_layer = layer_idx
                    self.dragging_view = True
                    layer = self.parent.layers[layer_idx]
                    layer.drag_offset = QPoint(
                        scene_pos.x() - layer.position.x(),
                        scene_pos.y() - layer.position.y()
                    )
                    return
                
                # 拖动视图
                self.dragging_view = True
                self.last_drag_pos = event.pos()
    
    def is_point_in_layer(self, point, layer):
        """检查点是否在图层范围内"""
        if not layer.drag_layer_image:
            return False
            
        # 计算图层边界
        layer_width = layer.drag_layer_image.width() * layer.layer_scale
        layer_height = layer.drag_layer_image.height() * layer.layer_scale
        
        layer_left = layer.layer_position.x() - layer_width / 2
        layer_top = layer.layer_position.y() - layer_height / 2
        layer_right = layer_left + layer_width
        layer_bottom = layer_top + layer_height
        
        return (layer_left <= point.x() <= layer_right and 
                layer_top <= point.y() <= layer_bottom)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 关键改进：约束变换符合2点透视"""
        current_layer = self.parent.get_current_layer()
        if not current_layer:
            return
            
        scene_pos = self.transform_pos(event.pos())
        
        if self.dragging_view:
            if self.dragging_point != -1:
                # 拖动顶点（透视约束处理）
                if current_layer.warp_points and len(current_layer.warp_points) >= 4 and current_layer.source_vp:
                    # 2点透视约束下的点拖动逻辑
                    vp1, vp2 = current_layer.source_vp[:2]
                    drag_idx = self.dragging_point
                    
                    # 更新当前拖动的点
                    current_layer.warp_points[drag_idx] = scene_pos
                    
                    # 根据2点透视规则更新其他点
                    if drag_idx == 0:  # 左上角点
                        # 右上角点应保持与VP1共线
                        p1 = current_layer.warp_points[0]
                        p2 = self.get_point_along_line(vp1, p1, current_layer.original_width)
                        current_layer.warp_points[1] = p2
                        
                        # 左下角点应保持与VP2共线
                        p3 = self.get_point_along_line(vp2, p1, current_layer.original_height)
                        current_layer.warp_points[3] = p3
                        
                        # 右下角点由右上角和左下角点确定
                        current_layer.warp_points[2] = self.line_intersection(p2, vp2, p3, vp1)
                        
                    elif drag_idx == 1:  # 右上角点
                        # 左上角点应保持与VP1共线
                        p2 = current_layer.warp_points[1]
                        p1 = self.get_point_along_line(vp1, p2, -current_layer.original_width)
                        current_layer.warp_points[0] = p1
                        
                        # 右下角点应保持与VP2共线
                        p3 = self.get_point_along_line(vp2, p2, current_layer.original_height)
                        current_layer.warp_points[2] = p3
                        
                        # 左下角点由左上角和右下角点确定
                        current_layer.warp_points[3] = self.line_intersection(p1, vp2, p3, vp1)
                        
                    elif drag_idx == 2:  # 右下角点
                        # 左下角点应保持与VP1共线
                        p3 = current_layer.warp_points[2]
                        p4 = self.get_point_along_line(vp1, p3, -current_layer.original_width)
                        current_layer.warp_points[3] = p4
                        
                        # 右上角点应保持与VP2共线
                        p2 = self.get_point_along_line(vp2, p3, -current_layer.original_height)
                        current_layer.warp_points[1] = p2
                        
                        # 左上角点由右上角和左下角点确定
                        current_layer.warp_points[0] = self.line_intersection(p2, vp1, p4, vp2)
                        
                    elif drag_idx == 3:  # 左下角点
                        # 右下角点应保持与VP1共线
                        p4 = current_layer.warp_points[3]
                        p3 = self.get_point_along_line(vp1, p4, current_layer.original_width)
                        current_layer.warp_points[2] = p3
                        
                        # 左上角点应保持与VP2共线
                        p1 = self.get_point_along_line(vp2, p4, -current_layer.original_height)
                        current_layer.warp_points[0] = p1
                        
                        # 右上角点由左上角和右下角点确定
                        current_layer.warp_points[1] = self.line_intersection(p1, vp1, p3, vp2)
                        
                    # 应用透视变换
                    self.parent.apply_perspective_to_layer(current_layer)
                elif current_layer.selection_points and len(current_layer.selection_points) >= 4:
                    # 拖动选区点，同时更新对应的透视控制点
                    current_layer.selection_points[self.dragging_point] = scene_pos
                    if len(current_layer.points) > self.dragging_point:
                        current_layer.points[self.dragging_point] = scene_pos
                elif current_layer.points and len(current_layer.points) > 0:
                    # 拖动透视控制点，同时更新对应的选区点
                    current_layer.points[self.dragging_point] = scene_pos
                    if len(current_layer.selection_points) > self.dragging_point:
                        current_layer.selection_points[self.dragging_point] = scene_pos
                    
                    # 重新计算消失点
                    if len(current_layer.points) == 4:
                        self.parent.grid.calculate_two_point_perspective(current_layer.points)
                self.update()
            elif self.dragging_edge != -1:
                # 拖动边
                idx1 = self.dragging_edge
                idx2 = (self.dragging_edge + 1) % 4
                
                # 计算边应该移动到的目标位置
                target_pos = scene_pos - self.edge_drag_offset
                
                # 计算边的方向向量
                p1 = current_layer.selection_points[idx1]
                p2 = current_layer.selection_points[idx2]
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                
                # 计算垂直于边的方向向量
                perp_dx = dy
                perp_dy = -dx
                if perp_dx != 0 or perp_dy != 0:
                    length = np.sqrt(perp_dx**2 + perp_dy**2)
                    perp_dx /= length
                    perp_dy /= length
                
                # 计算从原始边到目标点的距离
                dist = self.distance_point_to_line(target_pos, p1, p2)
                
                # 移动两个顶点
                move_x = int(perp_dx * dist)
                move_y = int(perp_dy * dist)
                
                current_layer.selection_points[idx1] = QPoint(
                    p1.x() + move_x,
                    p1.y() + move_y
                )
                current_layer.selection_points[idx2] = QPoint(
                    p2.x() + move_x,
                    p2.y() + move_y
                )
                
                # 同时更新对应的透视控制点
                if len(current_layer.points) > idx1:
                    current_layer.points[idx1] = current_layer.selection_points[idx1]
                if len(current_layer.points) > idx2:
                    current_layer.points[idx2] = current_layer.selection_points[idx2]
                
                # 重新计算消失点
                if len(current_layer.points) == 4:
                    self.parent.grid.calculate_two_point_perspective(current_layer.points)
                
                self.update()
            elif self.dragging_layer != -1:
                # 拖动图层
                layer = self.parent.layers[self.dragging_layer]

                if layer.layer_drag_mode and layer.drag_layer_image:
                    # 图层拖拽模式：移动整个图层
                    layer.layer_position = QPoint(
                        scene_pos.x() - layer.drag_offset.x(),
                        scene_pos.y() - layer.drag_offset.y()
                    )

                    # 计算新的缩放比例
                    if self.parent.grid.primary_vp and len(self.parent.grid.primary_vp) >= 2:
                        vp1 = self.parent.grid.primary_vp[0]
                        vp2 = self.parent.grid.primary_vp[1]
                        current_center = layer.layer_position

                        # 计算当前中心到两个消失点的距离
                        dx1 = current_center.x() - vp1.x()
                        dy1 = current_center.y() - vp1.y()
                        current_vp1_distance = np.sqrt(dx1 * dx1 + dy1 * dy1)

                        dx2 = current_center.x() - vp2.x()
                        dy2 = current_center.y() - vp2.y()
                        current_vp2_distance = np.sqrt(dx2 * dx2 + dy2 * dy2)

                        # 计算向量A：从初始中心到当前中心
                        vector_A = current_center - layer.initial_center
                        # 计算向量B1：从初始中心到vp1
                        vector_B1 = vp1 - layer.initial_center
                        # 计算向量B2：从初始中心到vp2
                        vector_B2 = vp2 - layer.initial_center

                        # 计算向量A与向量B1的夹角（弧度）
                        dot1 = vector_A.x() * vector_B1.x() + vector_A.y() * vector_B1.y()
                        det1 = vector_A.x() * vector_B1.y() - vector_A.y() * vector_B1.x()
                        angle1 = math.atan2(det1, dot1)

                        # 计算向量A与向量B2的夹角（弧度）
                        dot2 = vector_A.x() * vector_B2.x() + vector_A.y() * vector_B2.y()
                        det2 = vector_A.x() * vector_B2.y() - vector_A.y() * vector_B2.x()
                        angle2 = math.atan2(det2, dot2)

                        # 取绝对值，得到0到π之间的角度
                        angle1_abs = abs(angle1)
                        angle2_abs = abs(angle2)

                        # 选择夹角较小的消失点
                        if angle1_abs < angle2_abs:
                            # 使用vp1
                            if layer.initial_vp1_distance != 0:
                                scale_factor = current_vp1_distance / layer.initial_vp1_distance
                            else:
                                scale_factor = 1.0
                        else:
                            # 使用vp2
                            if layer.initial_vp2_distance != 0:
                                scale_factor = current_vp2_distance / layer.initial_vp2_distance
                            else:
                                scale_factor = 1.0

                        # 限制缩放范围，例如在0.1到2.0之间
                        scale_factor = max(0.1, min(scale_factor, 2.0))
                        layer.layer_scale = scale_factor


                else:
                    # 普通图层拖动：移动图层位置
                    layer.position = QPoint(
                        scene_pos.x() - layer.drag_offset.x(),
                        scene_pos.y() - layer.drag_offset.y()
                    )
                self.update()
    
    def line_intersection(self, a1, a2, b1, b2):
        """计算两条线的交点（用于透视约束）"""
        x1, y1 = a1.x(), a1.y()
        x2, y2 = a2.x(), a2.y()
        x3, y3 = b1.x(), b1.y()
        x4, y4 = b2.x(), b2.y()
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return QPoint(x2, y2)  # 平行线，返回终点
        
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = t_num / den
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return QPoint(int(x), int(y))
    
    def get_point_along_line(self, start, end, length):
        """沿直线从起点到终点方向移动指定长度"""
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        
        if dx == 0 and dy == 0:
            return end
            
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0:
            return end
            
        ratio = length / dist
        x = end.x() + dx * ratio
        y = end.y() + dy * ratio
        
        return QPoint(int(x), int(y))
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.dragging_view = False
            self.dragging_point = -1
            self.dragging_edge = -1
            self.dragging_layer = -1
    
    def wheelEvent(self, event):
        """鼠标滚轮事件，用于缩放"""
        current_layer = self.parent.get_current_layer()
        
        # 图层拖拽模式下的缩放
        if current_layer and current_layer.layer_drag_mode and current_layer.drag_layer_image:
            factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            current_layer.layer_scale *= factor
            current_layer.layer_scale = max(0.1, min(current_layer.layer_scale, 5.0))  # 限制缩放范围
            self.update()
            return
            
        # 普通视图缩放
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.scale_factor *= factor
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))  # 限制缩放范围
        self.update()
    
    def transform_pos(self, pos):
        """将窗口坐标转换为场景坐标（考虑缩放和偏移）"""
        x = (pos.x() - self.offset.x()) / self.scale_factor
        y = (pos.y() - self.offset.y()) / self.scale_factor
        return QPoint(int(x), int(y))
    
    def toggle_layer_drag_mode(self):
        """切换图层拖拽模式 - 集成复制选区和粘贴透视功能"""
        current_layer = self.parent.get_current_layer()
        if not current_layer:
            QMessageBox.warning(self, "警告", "请先选择图层")
            return
            
        if len(current_layer.selection_points) != 4:
            QMessageBox.warning(self, "警告", "请先标记4个点创建完整的四边形选区")
            return
            
        # 切换图层拖拽模式
        current_layer.layer_drag_mode = not current_layer.layer_drag_mode

        if current_layer.layer_drag_mode:
            # 复制选区内容到拖拽图层
            self.copy_selection_to_drag_layer(current_layer)

            if current_layer.drag_layer_image:
                # 初始化图层位置和缩放
                if len(current_layer.selection_points) == 4:
                    # 计算选区中心作为初始位置
                    center_x = sum(p.x() for p in current_layer.selection_points) / 4
                    center_y = sum(p.y() for p in current_layer.selection_points) / 4
                    current_layer.layer_position = QPoint(int(center_x), int(center_y))
                    current_layer.initial_center = current_layer.layer_position

                    if self.parent.grid.primary_vp and len(self.parent.grid.primary_vp) >= 2:
                        vp1 = self.parent.grid.primary_vp[0]
                        vp2 = self.parent.grid.primary_vp[1]
                        # 计算到vp1的初始距离
                        dx1 = current_layer.initial_center.x() - vp1.x()
                        dy1 = current_layer.initial_center.y() - vp1.y()
                        current_layer.initial_vp1_distance = np.sqrt(dx1 * dx1 + dy1 * dy1)
                        # 计算到vp2的初始距离
                        dx2 = current_layer.initial_center.x() - vp2.x()
                        dy2 = current_layer.initial_center.y() - vp2.y()
                        current_layer.initial_vp2_distance = np.sqrt(dx2 * dx2 + dy2 * dy2)
                    else:
                        current_layer.initial_vp1_distance = 1.0
                        current_layer.initial_vp2_distance = 1.0
                else:
                    current_layer.layer_position = QPoint(0, 0)
                    current_layer.initial_center = QPoint(0, 0)
                    current_layer.initial_vp2_distance = 1.0

                current_layer.layer_scale = 1.0
                QMessageBox.information(self, "提示", "图层拖拽模式已启用 - 可以拖拽和缩放图层")
            else:
                current_layer.layer_drag_mode = False
                QMessageBox.warning(self, "警告", "无法复制选区内容")
    
    def copy_selection_to_drag_layer(self, layer):
        """复制四边形选区到拖拽图层 - 修复颜色问题"""
        if len(layer.selection_points) != 4 or not self.pixmap:
            layer.drag_layer_image = None
            return
        
        try:
            # 对点进行排序（左上、右上、右下、左下）
            sorted_points = self.sort_points(layer.selection_points)
            
            # 计算四边形边界框
            min_x = int(min(p.x() for p in sorted_points))
            max_x = int(max(p.x() for p in sorted_points))
            min_y = int(min(p.y() for p in sorted_points))
            max_y = int(max(p.y() for p in sorted_points))
            
            width = max_x - min_x
            height = max_y - min_y
            
            if width <= 0 or height <= 0:
                layer.drag_layer_image = None
                return
            
            # 创建一个与四边形边界框相同大小的RGBA图像
            drag_layer = np.zeros((height, width, 4), dtype=np.uint8)
            
            # 定义源四边形和目标四边形
            src_points = np.float32([[p.x(), p.y()] for p in sorted_points])
            
            # 计算四边形在图层中的位置（相对于边界框）
            layer_points = []
            for point in sorted_points:
                layer_x = point.x() - min_x
                layer_y = point.y() - min_y
                layer_points.append([layer_x, layer_y])
            
            dst_points = np.float32(layer_points)
            
            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 转换QPixmap为OpenCV图像
            qimg = self.pixmap.toImage()
            width_orig = qimg.width()
            height_orig = qimg.height()
            ptr = qimg.bits()
            ptr.setsize(height_orig * width_orig * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height_orig, width_orig, 4))
            
            # 直接使用RGBA格式，避免颜色转换问题
            cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            
            # 应用透视变换，保持原始颜色 - 关键修复
            warped_image = cv2.warpPerspective(
                cv_img, matrix, (width, height),
                flags=cv2.INTER_LANCZOS4,  # 使用高质量插值
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            
            # 转换回RGBA格式
            warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGRA2RGBA)
            
            # 创建一个掩码来标识四边形区域
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_points.astype(np.int32), 255)
            
            # 设置Alpha通道为掩码
            warped_image[:, :, 3] = mask
            
            # 转换回QImage
            height_drag, width_drag, channels_drag = warped_image.shape
            bytes_per_line = channels_drag * width_drag
            
            q_img = QImage(
                warped_image.data, 
                width_drag, 
                height_drag, 
                bytes_per_line, 
                QImage.Format_RGBA8888
            )
            
            # 创建深拷贝，避免数据被释放
            layer.drag_layer_image = q_img.copy()
            
        except Exception as e:
            print(f"复制四边形区域失败: {str(e)}")
            layer.drag_layer_image = None
    
    def sort_points(self, points):
        """对点进行排序（左上、右上、右下、左下）"""
        # 计算中心点
        center_x = sum(p.x() for p in points) / 4
        center_y = sum(p.y() for p in points) / 4
        
        # 计算每个点相对于中心的角度
        angles = []
        for point in points:
            angle = np.arctan2(point.y() - center_y, point.x() - center_x)
            angles.append(angle)
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        sorted_points = [points[i] for i in sorted_indices]
        
        return sorted_points

class VanishingPointEditor(QMainWindow):
    """消失点编辑器主窗口"""
    def __init__(self):
        super().__init__()
        # 先初始化图层列表和相关属性
        self.layers = []  # 图层列表
        self.current_layer_idx = -1  # 当前图层索引
        self.grid = PerspectiveGrid()  # 透视网格
        self.control_point_mode = False  # 控制点标记模式（已合并选区功能）
        # 再初始化界面
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("2点透视编辑器 (图层拖拽版)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # 创建左侧工具栏
        tool_panel = QWidget()
        tool_layout = QVBoxLayout(tool_panel)
        tool_panel.setMaximumWidth(200)
        
        # 添加按钮
        self.add_layer_btn = QPushButton("添加图层")
        self.add_layer_btn.clicked.connect(self.add_layer)
        
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.upload_image)
        
        # 透视点标记按钮（已合并选区功能）
        self.control_point_btn = QPushButton("标记透视点/选区")
        self.control_point_btn.clicked.connect(self.toggle_control_point_mode)
        self.control_point_btn.setCheckable(True)
        
        self.clear_points_btn = QPushButton("清除透视点")
        self.clear_points_btn.clicked.connect(self.clear_points)
        
        self.calc_vp_btn = QPushButton("计算2点透视")
        self.calc_vp_btn.clicked.connect(self.calculate_vanishing_points)
        
        self.show_grid_btn = QPushButton("显示/隐藏网格")
        self.show_grid_btn.clicked.connect(self.toggle_grid)
        
        self.scale_up_btn = QPushButton("放大")
        self.scale_up_btn.clicked.connect(lambda: self.scale_image(1.1))
        
        self.scale_down_btn = QPushButton("缩小")
        self.scale_down_btn.clicked.connect(lambda: self.scale_image(0.9))
        
        # 网格密度调整
        self.grid_density_label = QLabel("网格密度:")
        self.grid_density_btn = QPushButton("调整")
        self.grid_density_btn.clicked.connect(self.adjust_grid_density)
        
        # 图层拖拽按钮
        self.layer_drag_btn = QPushButton("图层拖拽")
        self.layer_drag_btn.clicked.connect(self.toggle_layer_drag_mode)
        
        # 图层上下移按钮
        self.layer_up_btn = QPushButton("图层上移")
        self.layer_up_btn.clicked.connect(self.move_layer_up)
        
        self.layer_down_btn = QPushButton("图层下移")
        self.layer_down_btn.clicked.connect(self.move_layer_down)
        
        # 图层列表
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.select_layer)
        
        # 添加控件到工具栏
        tool_layout.addWidget(self.add_layer_btn)
        tool_layout.addWidget(self.upload_btn)
        tool_layout.addWidget(QLabel("图层:"))
        tool_layout.addWidget(self.layer_list)
        tool_layout.addWidget(self.layer_up_btn)
        tool_layout.addWidget(self.layer_down_btn)
        tool_layout.addWidget(self.control_point_btn)
        tool_layout.addWidget(self.clear_points_btn)
        tool_layout.addWidget(self.calc_vp_btn)
        tool_layout.addWidget(self.show_grid_btn)
        tool_layout.addWidget(self.grid_density_label)
        tool_layout.addWidget(self.grid_density_btn)
        tool_layout.addWidget(self.layer_drag_btn)
        tool_layout.addWidget(self.scale_up_btn)
        tool_layout.addWidget(self.scale_down_btn)
        tool_layout.addStretch()
        
        # 创建画布
        self.canvas = Canvas(self)
        
        # 添加到主布局
        main_layout.addWidget(tool_panel)
        main_layout.addWidget(self.canvas)
        
        self.setCentralWidget(main_widget)
        
        # 添加默认图层
        self.add_layer("背景图层")
    
    def toggle_control_point_mode(self):
        """切换透视控制点标记模式（已合并选区功能）"""
        # 切换控制点模式
        self.control_point_mode = not self.control_point_mode
        self.control_point_btn.setChecked(self.control_point_mode)
        
        # 更改光标
        cursor = Qt.CrossCursor if self.control_point_mode else Qt.ArrowCursor
        self.canvas.setCursor(cursor)
        
        # 显示提示信息
        if self.control_point_mode:
            QMessageBox.information(self, "提示", "请在图像上标记一个矩形的4个角点（按顺时针或逆时针顺序），这些点将同时作为透视控制点和选区边界")
    
    def toggle_layer_drag_mode(self):
        """切换图层拖拽模式"""

        self.canvas.toggle_layer_drag_mode()
    
    def move_layer_up(self):
        """将当前图层上移"""
        if self.current_layer_idx >= 0 and self.current_layer_idx < len(self.layers) - 1:
            # 交换z_order
            self.layers[self.current_layer_idx].z_order += 1
            self.layers[self.current_layer_idx + 1].z_order -= 1
            
            # 交换列表中的位置
            self.layers[self.current_layer_idx], self.layers[self.current_layer_idx + 1] = \
            self.layers[self.current_layer_idx + 1], self.layers[self.current_layer_idx]
            
            # 更新列表
            self.update_layer_list()
            self.current_layer_idx += 1
            self.layer_list.setCurrentRow(self.current_layer_idx)
            self.canvas.update()
    
    def move_layer_down(self):
        """将当前图层下移"""
        if self.current_layer_idx > 0 and self.current_layer_idx < len(self.layers):
            # 交换z_order
            self.layers[self.current_layer_idx].z_order -= 1
            self.layers[self.current_layer_idx - 1].z_order += 1
            
            # 交换列表中的位置
            self.layers[self.current_layer_idx], self.layers[self.current_layer_idx - 1] = \
            self.layers[self.current_layer_idx - 1], self.layers[self.current_layer_idx]
            
            # 更新列表
            self.update_layer_list()
            self.current_layer_idx -= 1
            self.layer_list.setCurrentRow(self.current_layer_idx)
            self.canvas.update()
    
    def update_layer_list(self):
        """更新图层列表"""
        self.layer_list.clear()
        for layer in self.layers:
            self.layer_list.addItem(layer.name)
    
    def apply_perspective_to_layer(self, layer):
        """将透视变换应用到图层 - 修复黑色区域问题"""
        if not layer.original_image or not layer.warp_points or len(layer.warp_points) != 4:
            return
            
        # 获取变换控制点
        dst_points = np.float32([[p.x(), p.y()] for p in layer.warp_points])
        
        # 原始图像的四个角
        w = layer.original_image.width()
        h = layer.original_image.height()
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 转换QImage为OpenCV图像（带Alpha通道）
        qimg = layer.original_image
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        
        # 直接使用RGBA格式，避免颜色转换问题
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        
        # 计算输出图像的大小和偏移（修复黑色区域关键）
        all_points = np.float32([[p.x(), p.y()] for p in layer.warp_points])
        min_x, min_y = np.int32(all_points.min(axis=0))
        max_x, max_y = np.int32(all_points.max(axis=0))
        
        out_width = max(1, int(max_x - min_x))
        out_height = max(1, int(max_y - min_y))
        
        # 调整变换矩阵以消除偏移
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
        adjusted_matrix = translation @ matrix
        
        # 应用透视变换，使用边缘填充而非黑色填充
        transformed_cv = cv2.warpPerspective(
            cv_img, adjusted_matrix, (out_width, out_height),
            borderMode=cv2.BORDER_TRANSPARENT  # 关键修复：使用透明填充而非黑色
        )
        
        # 转换回RGBA格式
        transformed_cv = cv2.cvtColor(transformed_cv, cv2.COLOR_BGRA2RGBA)
        
        # 转换回QImage（保留Alpha通道）
        q_img = QImage(
            transformed_cv.data, 
            out_width, 
            out_height, 
            out_width * 4, 
            QImage.Format_RGBA8888
        )
        
        # 更新图层的变换后图像和位置（使用计算出的最小坐标）
        layer.warped_image = q_img
        layer.position = QPoint(min_x, min_y)
    
    def add_layer(self, name=None):
        """添加新图层"""
        if not name:
            name, ok = QInputDialog.getText(self, "图层名称", "输入图层名称:")
            if not ok or not name:
                name = f"图层{len(self.layers) + 1}"
        
        new_layer = PerspectiveLayer(name)
        new_layer.z_order = len(self.layers)
        self.layers.append(new_layer)
        self.update_layer_list()
        self.current_layer_idx = len(self.layers) - 1
        self.layer_list.setCurrentRow(self.current_layer_idx)
    
    def upload_image(self):
        """上传图片到当前图层"""
        current_layer = self.get_current_layer()
        if not current_layer:
            QMessageBox.warning(self, "警告", "请先创建图层")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # 加载图片到图层
            current_layer.original_image = QImage(file_path)
            current_layer.warped_image = QImage(file_path)
            
            # 如果是第一个有图片的图层，设置为画布背景
            if self.canvas.pixmap.isNull():
                self.canvas.load_image(file_path)
    
    def get_current_layer(self):
        """获取当前选中的图层"""
        if 0 <= self.current_layer_idx < len(self.layers):
            return self.layers[self.current_layer_idx]
        return None
    
    def select_layer(self, item):
        """选择图层"""
        self.current_layer_idx = self.layer_list.row(item)
    
    def clear_points(self):
        """清除当前图层的点"""
        current_layer = self.get_current_layer()
        if current_layer:
            current_layer.points = []
            current_layer.selection_points = []  # 同时清除选区
            self.grid.primary_vp = []
            self.canvas.update()
    
    def calculate_vanishing_points(self):
        """计算2点透视的消失点"""
        current_layer = self.get_current_layer()
        if not current_layer or len(current_layer.points) != 4:
            QMessageBox.warning(self, "警告", "请先在图层上标记4个点（矩形的四个角）")
            return
            
        # 计算2点透视消失点
        self.grid.calculate_two_point_perspective(current_layer.points)
        
        if not self.grid.primary_vp or len(self.grid.primary_vp) < 2:
            QMessageBox.warning(self, "警告", "无法计算有效的消失点，请确保标记的是一个矩形的四个角")
            return
            
        self.canvas.update()
        
        # 显示消失点信息
        msg = "2点透视消失点计算完成:\n"
        vp1, vp2 = self.grid.primary_vp[:2]
        msg += f"消失点 1 (水平方向): ({vp1.x()}, {vp1.y()})\n"
        msg += f"消失点 2 (深度方向): ({vp2.x()}, {vp2.y()})\n"
        msg += "现在可以使用图层拖拽功能复制和移动选区"
        
        QMessageBox.information(self, "2点透视信息", msg)
    
    def toggle_grid(self):
        """显示/隐藏网格"""
        self.grid.enabled = not self.grid.enabled
        self.canvas.update()
    
    def adjust_grid_density(self):
        """调整网格密度"""
        value, ok = QInputDialog.getInt(self, "网格密度", "输入线数量 (5-50):", 
                                       self.grid.line_count, 5, 50)
        if ok:
            self.grid.line_count = value
            self.canvas.update()
    
    def scale_image(self, factor):
        """缩放图像"""
        self.canvas.scale_factor *= factor
        self.canvas.scale_factor = max(0.1, min(self.canvas.scale_factor, 5.0))  # 限制缩放范围
        self.canvas.update()
    
    def draw_layers(self, painter):
        """绘制所有可见图层"""
        # 按z_order排序图层
        sorted_layers = sorted(self.layers, key=lambda x: x.z_order)
        
        for layer in sorted_layers:
            if layer.visible:
                # 保存当前变换状态
                painter.save()
                
                if layer.layer_drag_mode and layer.drag_layer_image:
                    # 图层拖拽模式：绘制拖拽图层
                    center_x = layer.layer_position.x()
                    center_y = layer.layer_position.y()
                    
                    # 计算缩放后的图像尺寸
                    scaled_width = layer.drag_layer_image.width() * layer.layer_scale
                    scaled_height = layer.drag_layer_image.height() * layer.layer_scale
                    
                    # 计算绘制位置（使图像中心对准图层位置）
                    draw_x = center_x - scaled_width / 2
                    draw_y = center_y - scaled_height / 2
                    
                    # 绘制缩放后的图像
                    scaled_rect = QRect(int(draw_x), int(draw_y), 
                                      int(scaled_width), int(scaled_height))
                    painter.drawImage(scaled_rect, layer.drag_layer_image)
                
                # 绘制普通图层（无论是否在拖拽模式）
                if layer.warped_image and not layer.layer_drag_mode:
                    painter.setOpacity(layer.opacity)
                    painter.drawImage(layer.position, layer.warped_image)
                    painter.setOpacity(1.0)  # 重置透明度
                
                # 恢复变换状态
                painter.restore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VanishingPointEditor()
    window.show()
    sys.exit(app.exec_())