import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.widgets import Slider, Button, RadioButtons

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示

def visualize_nii_gz(file_path):
    """
    加载并可视化.nii.gz格式的医学图像

    参数:
    file_path (str): .nii.gz文件的路径
    """
    # 加载NIfTI文件
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(f"成功加载图像，形状: {data.shape}")
    except Exception as e:
        print(f"加载图像失败: {e}")
        return

    # 检查数据维度
    if data.ndim == 3:
        # 三维数据：直接显示
        display_data = data
        time_points = 1
        initial_time_point = 0
    elif data.ndim == 4:
        # 四维数据：默认显示第一个时间点
        display_data = data[:, :, :, 0]
        time_points = data.shape[3]
        initial_time_point = 0
    else:
        print(f"不支持的维度数: {data.ndim}")
        return

    # 创建可视化界面
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(left=0.05, bottom=0.30, right=0.95, top=0.95)

    # 显示三个方向的初始切片
    mid_sagittal = display_data.shape[0] // 2
    mid_coronal = display_data.shape[1] // 2
    mid_axial = display_data.shape[2] // 2

    # 轴向视图（z方向）
    ax_axial = axes[0].imshow(display_data[:, :, mid_axial], cmap='gray', origin='lower')
    axes[0].set_title('轴向视图 (Axial)')

    # 冠状面视图（y方向）
    ax_coronal = axes[1].imshow(display_data[:, mid_coronal, :], cmap='gray', origin='lower')
    axes[1].set_title('冠状面视图 (Coronal)')

    # 矢状面视图（x方向）
    ax_sagittal = axes[2].imshow(display_data[mid_sagittal, :, :], cmap='gray', origin='lower')
    axes[2].set_title('矢状面视图 (Sagittal)')

    # 添加滑块控件
    axcolor = 'lightgoldenrodyellow'
    ax_axial_slider = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_coronal_slider = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_sagittal_slider = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_time_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor) if time_points > 1 else None

    # 创建滑块
    axial_slider = Slider(ax_axial_slider, '轴向切片', 0, display_data.shape[2] - 1, valinit=mid_axial, valstep=1)
    coronal_slider = Slider(ax_coronal_slider, '冠状面切片', 0, display_data.shape[1] - 1, valinit=mid_coronal,
                            valstep=1)
    sagittal_slider = Slider(ax_sagittal_slider, '矢状面切片', 0, display_data.shape[0] - 1, valinit=mid_sagittal,
                             valstep=1)

    # 如果是四维数据，添加时间点滑块
    if time_points > 1:
        time_slider = Slider(ax_time_slider, '时间点', 0, time_points - 1, valinit=initial_time_point, valstep=1)
    else:
        time_slider = None

    # 更新函数
    def update_axial(val):
        slice_idx = int(val)
        ax_axial.set_data(display_data[:, :, slice_idx])
        fig.canvas.draw_idle()

    def update_coronal(val):
        slice_idx = int(val)
        ax_coronal.set_data(display_data[:, slice_idx, :])
        fig.canvas.draw_idle()

    def update_sagittal(val):
        slice_idx = int(val)
        ax_sagittal.set_data(display_data[slice_idx, :, :])
        fig.canvas.draw_idle()

    # 四维数据的时间点更新函数
    def update_time(val):
        nonlocal display_data
        time_idx = int(val)
        display_data = data[:, :, :, time_idx]

        # 更新所有视图
        ax_axial.set_data(display_data[:, :, int(axial_slider.val)])
        ax_coronal.set_data(display_data[:, int(coronal_slider.val), :])
        ax_sagittal.set_data(display_data[int(sagittal_slider.val), :, :])

        fig.canvas.draw_idle()

    # 连接滑块和更新函数
    axial_slider.on_changed(update_axial)
    coronal_slider.on_changed(update_coronal)
    sagittal_slider.on_changed(update_sagittal)

    if time_slider is not None:
        time_slider.on_changed(update_time)

    # 显示图像
    plt.show()


if __name__ == "__main__":
    # 请替换为你的.nii.gz文件路径
    file_path = "D:\\fmri\\fmri_start\sub-02\\func\sub-02_task-rest_bold.nii.gz"
    visualize_nii_gz(file_path)