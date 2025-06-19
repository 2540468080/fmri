import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.widgets import Slider, Button, RadioButtons

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class NiftiViewer:
    """NIfTI 图像查看器类"""

    def __init__(self, file_path):
        """初始化查看器"""
        self.file_path = file_path
        self.img = None
        self.data = None
        self.affine = None
        self.header = None
        self.current_view = 'axial'  # 默认视图为轴状面
        self.current_slice = 0
        self.fig = None
        self.ax = None
        self.slider = None
        self.buttons = None
        self.radio = None
        self.img_plot = None

    def load_data(self):
        """加载NIfTI数据"""
        try:
            self.img = nib.load(self.file_path)
            self.data = self.img.get_fdata()
            self.affine = self.img.affine
            self.header = self.img.header
            print(f"成功加载图像: {os.path.basename(self.file_path)}")
            print(f"图像形状: {self.data.shape}")
            print(f"数据类型: {self.data.dtype}")
            print(f"像素尺寸: {self.header.get_zooms()}")
        except Exception as e:
            print(f"加载图像失败: {e}")
            sys.exit(1)

    def update(self, val):
        """更新切片显示"""
        self.current_slice = int(val)
        self._update_plot()

    def change_view(self, label):
        """更改视图方向"""
        self.current_view = label
        self.current_slice = self._get_mid_slice()
        self.slider.valmax = self._get_max_slice()
        self.slider.val = self.current_slice
        self.slider.ax.set_xlim(0, self._get_max_slice())
        self._update_plot()

    def _update_plot(self):
        """更新绘图"""
        if self.current_view == 'axial':
            slice_data = self.data[:, :, self.current_slice]
        elif self.current_view == 'coronal':
            slice_data = self.data[:, self.current_slice, :]
        elif self.current_view == 'sagittal':
            slice_data = self.data[self.current_slice, :, :]

        if self.img_plot is None:
            self.img_plot = self.ax.imshow(slice_data, cmap='gray')
        else:
            self.img_plot.set_data(slice_data)

        self.ax.set_title(f"{self.current_view.capitalize()} 视图 - 切片: {self.current_slice}")
        self.fig.canvas.draw_idle()

    def _get_max_slice(self):
        """获取当前视图的最大切片数"""
        if self.current_view == 'axial':
            return self.data.shape[2] - 1
        elif self.current_view == 'coronal':
            return self.data.shape[1] - 1
        elif self.current_view == 'sagittal':
            return self.data.shape[0] - 1
        return 0

    def _get_mid_slice(self):
        """获取当前视图的中间切片"""
        return self._get_max_slice() // 2

    def display_info(self):
        """显示图像信息"""
        print("\n=== 图像详细信息 ===")
        print(f"文件名: {os.path.basename(self.file_path)}")
        print(f"完整路径: {os.path.abspath(self.file_path)}")
        print(f"形状: {self.data.shape}")
        print(f"维度: {len(self.data.shape)}")
        print(f"数据类型: {self.data.dtype}")
        print(f"像素间距: {self.header.get_zooms()}")
        print(f"数据范围: [{np.min(self.data)}, {np.max(self.data)}]")
        print(f"数据均值: {np.mean(self.data)}")
        print(f"数据标准差: {np.std(self.data)}")

        # 显示头文件信息
        print("\n=== 头文件信息 ===")
        print(self.header)

    def visualize(self):
        """可视化NIfTI图像"""
        if self.data is None:
            self.load_data()

        # 创建图形和子图
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0.1, 0.3, 0.8, 0.6])  # [left, bottom, width, height]

        # 初始化显示中间切片
        self.current_slice = self._get_mid_slice()
        self._update_plot()

        # 创建滑块
        slider_ax = self.fig.add_axes([0.1, 0.2, 0.8, 0.03])
        self.slider = Slider(slider_ax, '切片', 0, self._get_max_slice(),
                             valinit=self.current_slice, valstep=1)
        self.slider.on_changed(self.update)

        # 创建视图选择按钮
        radio_ax = self.fig.add_axes([0.1, 0.05, 0.15, 0.1])
        self.radio = RadioButtons(radio_ax, ('axial', 'coronal', 'sagittal'), active=0)
        self.radio.on_clicked(self.change_view)

        # 创建信息按钮
        info_ax = self.fig.add_axes([0.7, 0.05, 0.2, 0.04])
        info_button = Button(info_ax, '显示图像信息')
        info_button.on_clicked(lambda event: self.display_info())

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.show()


def main():
    """主函数"""
    # 设置默认文件路径，将此处修改为你需要的路径
    default_file_path = "D:\\fmri\group_results\ReHo_z.nii.gz"

    parser = argparse.ArgumentParser(description='NIfTI 图像查看器')
    parser.add_argument('--file', default=default_file_path, help='.nii或.nii.gz文件路径')
    parser.add_argument('--info', action='store_true', help='仅显示图像信息而不进行可视化')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"错误: 文件不存在 - {args.file}")
        sys.exit(1)

    viewer = NiftiViewer(args.file)

    try:
        viewer.load_data()

        if args.info:
            viewer.display_info()
        else:
            viewer.visualize()
    except KeyboardInterrupt:
        print("\n程序已被用户中断")
        sys.exit(0)


if __name__ == "__main__":
    main()