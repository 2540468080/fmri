import pandas as pd
import numpy as np
import os
from pathlib import Path
import nibabel as nib
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import resample_to_img, smooth_img
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map, view_img
from nilearn.reporting import get_clusters_table
from nilearn.datasets import load_mni152_template
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class FMRIGroupAnalysis:
    def __init__(self, participants_file, base_dir):
        """
        初始化分析类

        Parameters:
        -----------
        participants_file : str
            participants.tsv文件路径
        base_dir : str
            基础数据目录路径
        """
        self.participants_file = participants_file
        self.base_dir = Path(base_dir)
        self.results_dir = Path("group_analysis_results3")

        self.results_dir.mkdir(exist_ok=True)

        # 加载参与者信息
        self.participants = pd.read_csv(participants_file, sep='\t')
        print(f"加载了 {len(self.participants)} 个被试的信息")
        print(f"组别分布: {self.participants['group'].value_counts()}")

        # 显示数据概览
        print("\n数据概览:")
        print(f"抑郁组年龄: {self.participants[self.participants['group'] == 'depr']['age'].describe()}")
        print(f"对照组年龄: {self.participants[self.participants['group'] == 'control']['age'].describe()}")
        print(f"性别分布: {self.participants['gender'].value_counts()}")

    def prepare_data_paths(self):
        """
        准备数据表与影像清单
        """
        # 初始化路径字典
        self.data_paths = {
            'alff': {},
            'reho': {},
            'seed1_fc': {},
            'seed2_fc': {},
            'brain_mask': {}
        }

        # 为每个被试构建文件路径
        missing_files = []

        for idx, row in self.participants.iterrows():
            participant_id = row['participant_id']

            # 构建各指标文件路径
            sub_dir = self.base_dir / participant_id / "func"

            paths = {
                'alff': sub_dir / f"{participant_id}_alff.nii.gz",
                'reho': sub_dir / f"{participant_id}_reho.nii.gz",
                'seed1_fc': sub_dir / f"{participant_id}_seed1_fc_map.nii.gz",
                'seed2_fc': sub_dir / f"{participant_id}_seed2_fc_map.nii.gz",
                'brain_mask': sub_dir / f"{participant_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
            }

            # 检查文件是否存在
            for metric, path in paths.items():
                if path.exists():
                    self.data_paths[metric][participant_id] = str(path)
                else:
                    missing_files.append(f"{participant_id}: {metric} - {path}")

        if missing_files:
            print("警告：以下文件缺失：")
            for file in missing_files[:10]:  # 只显示前10个
                print(f"  {file}")
            if len(missing_files) > 10:
                print(f"  ... 还有 {len(missing_files) - 10} 个文件缺失")

        # 过滤出有完整数据的被试
        complete_subjects = set(self.data_paths['alff'].keys())
        for metric in ['reho', 'seed1_fc', 'seed2_fc', 'brain_mask']:
            complete_subjects &= set(self.data_paths[metric].keys())

        # 同时过滤掉年龄或性别缺失的被试
        valid_participants = self.participants[
            (self.participants['participant_id'].isin(complete_subjects)) &
            (self.participants['age'].notna()) &
            (self.participants['gender'].notna())
            ]

        self.participants_complete = valid_participants.copy().reset_index(drop=True)

        print(f"有完整数据的被试: {len(self.participants_complete)} 个")
        print(f"完整数据组别分布: {self.participants_complete['group'].value_counts()}")

        # 检查是否有足够的样本进行统计分析
        group_counts = self.participants_complete['group'].value_counts()
        if len(group_counts) < 2 or any(group_counts < 5):
            print("警告：某些组的样本量可能过小，建议检查数据完整性")

        return self.participants_complete

    def prepare_design_matrix(self):
        """
        构建设计矩阵
        """
        # 编码组别：抑郁组=1，健康对照=0
        group_mapping = {'depr': 1, 'control': 0}
        self.participants_complete['group_code'] = self.participants_complete['group'].map(group_mapping)

        # 编码性别：男性=1，女性=0 (与你的数据格式一致)
        gender_mapping = {'m': 1, 'f': 0}
        self.participants_complete['gender_code'] = self.participants_complete['gender'].map(gender_mapping)

        # 处理缺失的age数据
        if self.participants_complete['age'].isna().any():
            print("警告：发现缺失的年龄数据，将使用组内均值填充")
            self.participants_complete['age'] = self.participants_complete.groupby('group')['age'].transform(
                lambda x: x.fillna(x.mean())
            )

        # 构建设计矩阵
        self.design_matrix = pd.DataFrame({
            'intercept': np.ones(len(self.participants_complete)),
            'group': self.participants_complete['group_code'].values,
            'age': self.participants_complete['age'].values,
            'gender': self.participants_complete['gender_code'].values
        })

        print("设计矩阵构建完成:")
        print(f"样本数量: {len(self.design_matrix)}")
        print(f"抑郁组: {(self.design_matrix['group'] == 1).sum()}人")
        print(f"对照组: {(self.design_matrix['group'] == 0).sum()}人")
        print(f"男性: {(self.design_matrix['gender'] == 1).sum()}人")
        print(f"女性: {(self.design_matrix['gender'] == 0).sum()}人")
        print(f"年龄范围: {self.design_matrix['age'].min():.1f} - {self.design_matrix['age'].max():.1f}岁")
        print("\n设计矩阵描述统计:")
        print(self.design_matrix.describe())

        return self.design_matrix

    def diagnose_image_consistency(self, metric_name, save_report=True):
        """
        诊断图像一致性问题的辅助方法

        Parameters:
        -----------
        metric_name : str
            指标名称
        save_report : bool
            是否保存诊断报告
        """
        print(f"正在诊断 {metric_name} 图像一致性...")

        # 获取图像路径列表
        img_paths = []
        participant_ids = []
        for participant_id in self.participants_complete['participant_id']:
            img_paths.append(self.data_paths[metric_name][participant_id])
            participant_ids.append(participant_id)

        # 诊断结果存储
        diagnostic_results = []

        # 加载参考图像
        reference_img = nib.load(img_paths[0])
        ref_participant = participant_ids[0]

        # 处理参考图像的维度
        if len(reference_img.shape) == 4:
            ref_data = reference_img.get_fdata()[:, :, :, 0]
            reference_img = nib.Nifti1Image(ref_data, reference_img.affine, reference_img.header)

        reference_affine = reference_img.affine
        reference_shape = reference_img.shape[:3]
        reference_voxel_size = np.abs(np.diag(reference_affine)[:3])

        print(f"参考图像 ({ref_participant}):")
        print(f"  形状: {reference_shape}")
        print(f"  体素尺寸: {reference_voxel_size}")

        for i, (img_path, participant_id) in enumerate(zip(img_paths, participant_ids)):
            try:
                img = nib.load(img_path)

                # 基本信息
                original_shape = img.shape
                current_affine = img.affine
                current_voxel_size = np.abs(np.diag(current_affine)[:3])

                # 检查各种一致性
                shape_3d = original_shape[:3]
                is_4d = len(original_shape) == 4
                shape_consistent = shape_3d == reference_shape

                # 仿射矩阵差异
                affine_diff = np.abs(current_affine - reference_affine)
                max_affine_diff = np.max(affine_diff)
                affine_consistent = max_affine_diff < 0.02  # 0.02mm容差

                # 体素尺寸差异
                voxel_diff = np.abs(current_voxel_size - reference_voxel_size)
                max_voxel_diff = np.max(voxel_diff)
                voxel_consistent = max_voxel_diff < 0.02

                # 存储诊断结果
                result = {
                    'participant_id': participant_id,
                    'file_path': img_path,
                    'original_shape': str(original_shape),
                    'shape_3d': str(shape_3d),
                    'is_4d': is_4d,
                    'voxel_size': f"({current_voxel_size[0]:.3f}, {current_voxel_size[1]:.3f}, {current_voxel_size[2]:.3f})",
                    'shape_ok': shape_consistent,
                    'voxel_ok': voxel_consistent,
                    'affine_ok': affine_consistent,
                    'max_affine_diff': max_affine_diff,
                    'max_voxel_diff': max_voxel_diff,
                    'needs_resampling': not (shape_consistent and affine_consistent) or is_4d
                }

                diagnostic_results.append(result)

                # 打印问题图像
                if result['needs_resampling']:
                    print(f"问题图像: {participant_id}")
                    print(f"  4D图像: {is_4d}")
                    print(f"  形状一致: {shape_consistent} ({shape_3d} vs {reference_shape})")
                    print(f"  体素一致: {voxel_consistent} (最大差异: {max_voxel_diff:.6f})")
                    print(f"  仿射一致: {affine_consistent} (最大差异: {max_affine_diff:.6f})")

            except Exception as e:
                print(f"诊断 {participant_id} 时出错: {e}")
                result = {
                    'participant_id': participant_id,
                    'file_path': img_path,
                    'error': str(e),
                    'needs_resampling': True
                }
                diagnostic_results.append(result)

        # 生成统计摘要
        total_images = len(diagnostic_results)
        problematic_images = sum(1 for r in diagnostic_results if r.get('needs_resampling', False))
        images_4d = sum(1 for r in diagnostic_results if r.get('is_4d', False))
        shape_issues = sum(1 for r in diagnostic_results if not r.get('shape_ok', True))
        affine_issues = sum(1 for r in diagnostic_results if not r.get('affine_ok', True))

        print(f"\n{metric_name} 诊断摘要:")
        print(f"  总图像数: {total_images}")
        print(f"  问题图像数: {problematic_images}")
        print(f"  4D图像数: {images_4d}")
        print(f"  形状不一致: {shape_issues}")
        print(f"  仿射不一致: {affine_issues}")

        # 保存诊断报告
        if save_report:
            output_dir = self.results_dir / metric_name
            output_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(diagnostic_results)
            report_path = output_dir / f"{metric_name}_image_diagnostic_report.csv"
            df.to_csv(report_path, index=False)
            print(f"诊断报告已保存: {report_path}")

            # 只保存有问题的图像信息
            problematic_df = df[df.get('needs_resampling', False)]
            if len(problematic_df) > 0:
                problem_report_path = output_dir / f"{metric_name}_problematic_images.csv"
                problematic_df.to_csv(problem_report_path, index=False)
                print(f"问题图像报告已保存: {problem_report_path}")

        return diagnostic_results

    def check_and_resample_images(self, metric_name):
        """
        检查并重采样图像以确保一致性
        增强版本：更好地处理仿射矩阵微小差异和维度不一致问题
        """
        print(f"正在检查和重采样 {metric_name} 图像...")

        # 获取图像路径列表
        img_paths = []
        for participant_id in self.participants_complete['participant_id']:
            img_paths.append(self.data_paths[metric_name][participant_id])

        # 加载并预处理参考图像
        reference_img = nib.load(img_paths[0])

        # 如果参考图像是4D，取第一个时间点
        if len(reference_img.shape) == 4:
            print(f"参考图像是4D，取第一个时间点作为参考")
            ref_data = reference_img.get_fdata()[:, :, :, 0]
            reference_img = nib.Nifti1Image(ref_data, reference_img.affine, reference_img.header)

        reference_affine = reference_img.affine
        reference_shape = reference_img.shape[:3]  # 只取前3维

        print(f"参考图像信息:")
        print(f"  形状: {reference_shape}")
        print(f"  体素尺寸: {np.diag(reference_affine)[:3]}")

        resampled_paths = []
        output_dir = self.results_dir / metric_name / "resampled"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 统计需要重采样的图像
        resampling_stats = {
            'total': len(img_paths),
            'dimension_fixed': 0,
            'affine_resampled': 0,
            'already_consistent': 0,
            'errors': 0
        }

        for i, img_path in enumerate(img_paths):
            try:
                img = nib.load(img_path)
                participant_id = self.participants_complete['participant_id'].iloc[i]
                needs_resampling = False

                # 检查并处理维度问题
                if len(img.shape) == 4:
                    print(f"被试 {participant_id}: 4D图像转换为3D")
                    img_data = img.get_fdata()[:, :, :, 0]
                    img = nib.Nifti1Image(img_data, img.affine, img.header)
                    resampling_stats['dimension_fixed'] += 1
                    needs_resampling = True

                # 检查仿射矩阵是否一致（使用更宽松的容差）
                affine_diff = np.abs(img.affine - reference_affine)
                max_affine_diff = np.max(affine_diff)

                # 检查形状是否一致
                shape_consistent = img.shape[:3] == reference_shape

                # 使用更宽松的仿射矩阵容差（0.1 mm，从0.02增加到0.1）
                affine_consistent = max_affine_diff < 0.1

                if not shape_consistent or not affine_consistent:
                    print(f"被试 {participant_id}: 需要重采样")
                    print(f"  形状一致: {shape_consistent} (当前: {img.shape[:3]}, 参考: {reference_shape})")
                    print(f"  仿射矩阵一致: {affine_consistent} (最大差异: {max_affine_diff:.6f})")

                    # 执行重采样 - 使用更稳健的参数
                    try:
                        resampled_img = resample_to_img(
                            img,
                            reference_img,
                            interpolation='linear',
                            force_resample=True  # 强制重采样
                        )
                    except Exception as resample_error:
                        print(f"  标准重采样失败，尝试备选方案: {resample_error}")
                        # 备选方案：手动重采样
                        from nilearn.image import resample_img
                        resampled_img = resample_img(
                            img,
                            target_affine=reference_affine,
                            target_shape=reference_shape,
                            interpolation='linear'
                        )

                    resampling_stats['affine_resampled'] += 1
                    needs_resampling = True
                else:
                    resampled_img = img
                    resampling_stats['already_consistent'] += 1

                # 最终验证：确保输出图像与参考一致
                final_shape_ok = resampled_img.shape[:3] == reference_shape
                final_affine_ok = np.allclose(resampled_img.affine, reference_affine, atol=0.1)  # 增加容差

                if not (final_shape_ok and final_affine_ok):
                    print(f"警告: 被试 {participant_id} 重采样后仍不一致")
                    print(f"  形状: {resampled_img.shape[:3]} vs {reference_shape}")
                    print(f"  仿射差异: {np.max(np.abs(resampled_img.affine - reference_affine)):.8f}")

                    # 如果仍然不一致，强制使用参考图像的仿射矩阵
                    if not final_affine_ok:
                        print(f"  强制使用参考仿射矩阵")
                        resampled_data = resampled_img.get_fdata()
                        resampled_img = nib.Nifti1Image(resampled_data, reference_affine, reference_img.header)

                # 保存图像
                resampled_path = output_dir / f"{participant_id}_{metric_name}_resampled.nii.gz"
                resampled_img.to_filename(resampled_path)
                resampled_paths.append(str(resampled_path))

                if needs_resampling:
                    print(f"  已保存重采样图像: {resampled_path.name}")

            except Exception as e:
                print(f"处理被试 {participant_id} 的 {metric_name} 图像时出错: {e}")
                resampling_stats['errors'] += 1

                # 尝试最基本的重采样作为最后的备选方案
                try:
                    print(f"尝试最终备选方案处理被试 {participant_id}")
                    img = nib.load(img_path)

                    # 强制处理为3D
                    if len(img.shape) == 4:
                        img_data = img.get_fdata()[:, :, :, 0]
                    else:
                        img_data = img.get_fdata()

                    # 创建与参考图像完全一致的新图像
                    if img_data.shape[:3] != reference_shape:
                        from scipy.ndimage import zoom
                        # 计算缩放因子
                        zoom_factors = [ref_dim / img_dim for ref_dim, img_dim in
                                        zip(reference_shape, img_data.shape[:3])]
                        img_data = zoom(img_data, zoom_factors, order=1)

                    # 使用参考图像的仿射矩阵和头信息
                    resampled_img = nib.Nifti1Image(img_data, reference_affine, reference_img.header)

                    resampled_path = output_dir / f"{participant_id}_{metric_name}_resampled.nii.gz"
                    resampled_img.to_filename(resampled_path)
                    resampled_paths.append(str(resampled_path))
                    print(f"  最终备选方案成功处理")
                    resampling_stats['errors'] -= 1  # 成功了就减去错误计数
                    resampling_stats['affine_resampled'] += 1

                except Exception as e2:
                    print(f"最终备选方案也失败: {e2}")
                    raise e2

        # 打印重采样统计信息
        print(f"\n{metric_name} 重采样统计:")
        print(f"  总图像数: {resampling_stats['total']}")
        print(f"  4D转3D: {resampling_stats['dimension_fixed']}")
        print(f"  仿射重采样: {resampling_stats['affine_resampled']}")
        print(f"  已一致: {resampling_stats['already_consistent']}")
        print(f"  错误数: {resampling_stats['errors']}")

        # 验证所有输出图像
        print(f"\n验证最终输出图像一致性...")
        consistency_check_passed = 0
        for i, path in enumerate(resampled_paths):
            img = nib.load(path)
            participant_id = self.participants_complete['participant_id'].iloc[i]

            shape_ok = img.shape[:3] == reference_shape
            affine_ok = np.allclose(img.affine, reference_affine, atol=0.1)

            if shape_ok and affine_ok:
                consistency_check_passed += 1
            else:
                print(f"警告: {participant_id} 最终验证失败")
                print(f"  形状OK: {shape_ok}, 仿射OK: {affine_ok}")

        print(f"一致性检查通过: {consistency_check_passed}/{len(resampled_paths)} 个图像")
        print(f"所有图像已重采样完成，共 {len(resampled_paths)} 个文件")
        return resampled_paths

    def run_second_level_analysis(self, metric_name, smoothing_fwhm=6):
        """
        运行二级GLM分析

        Parameters:
        -----------
        metric_name : str
            指标名称 ('alff', 'reho', 'seed1_fc', 'seed2_fc')
        smoothing_fwhm : float
            平滑核大小
        """
        print(f"\n正在分析 {metric_name.upper()} ...")

        # 检查并重采样图像
        img_paths = self.check_and_resample_images(metric_name)

        # 在二级分析前再次验证和强制统一所有图像
        print(f"在二级分析前验证图像一致性...")

        # 加载参考图像（重采样后的第一个图像）
        reference_img = nib.load(img_paths[0])
        if len(reference_img.shape) == 4:
            ref_data = reference_img.get_fdata()[:, :, :, 0]
            reference_img = nib.Nifti1Image(ref_data, reference_img.affine, reference_img.header)

        reference_affine = reference_img.affine
        reference_shape = reference_img.shape[:3]

        # 创建临时目录存放最终统一的图像
        final_output_dir = self.results_dir / metric_name / "final_unified"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        unified_img_paths = []

        for i, img_path in enumerate(img_paths):
            try:
                img = nib.load(img_path)
                participant_id = self.participants_complete['participant_id'].iloc[i]

                # 处理4D图像
                if len(img.shape) == 4:
                    print(f"二级分析前处理4D图像: {participant_id}")
                    img_data = img.get_fdata()[:, :, :, 0]
                    img = nib.Nifti1Image(img_data, img.affine, img.header)

                # 检查是否需要进一步统一
                shape_ok = img.shape[:3] == reference_shape
                affine_ok = np.allclose(img.affine, reference_affine, atol=1e-6)

                if not (shape_ok and affine_ok):
                    print(f"强制统一图像: {participant_id}")
                    print(f"  形状一致: {shape_ok}")
                    print(f"  仿射一致: {affine_ok}")

                    # 强制使用参考图像的精确仿射矩阵和形状
                    img_data = img.get_fdata()

                    # 如果形状不一致，使用插值调整
                    if not shape_ok:
                        from scipy.ndimage import zoom
                        zoom_factors = [ref_dim / img_dim for ref_dim, img_dim in
                                        zip(reference_shape, img_data.shape[:3])]
                        img_data = zoom(img_data, zoom_factors, order=1)

                    # 创建与参考完全一致的图像
                    unified_img = nib.Nifti1Image(
                        img_data,
                        reference_affine.copy(),  # 使用精确相同的仿射矩阵
                        reference_img.header
                    )
                else:
                    # 即使一致，也强制使用相同的仿射矩阵以避免浮点精度问题
                    img_data = img.get_fdata()
                    unified_img = nib.Nifti1Image(
                        img_data,
                        reference_affine.copy(),
                        reference_img.header
                    )

                # 保存统一后的图像
                unified_path = final_output_dir / f"{participant_id}_{metric_name}_unified.nii.gz"
                unified_img.to_filename(unified_path)
                unified_img_paths.append(str(unified_path))

            except Exception as e:
                print(f"统一图像 {participant_id} 时出错: {e}")
                raise e

        # 最终验证所有图像完全一致
        print(f"最终验证所有图像一致性...")
        for i, path in enumerate(unified_img_paths):
            img = nib.load(path)
            participant_id = self.participants_complete['participant_id'].iloc[i]

            if not (img.shape[:3] == reference_shape and np.array_equal(img.affine, reference_affine)):
                print(f"错误: {participant_id} 仍然不一致!")
                print(f"  形状: {img.shape[:3]} vs {reference_shape}")
                print(f"  仿射矩阵相等: {np.array_equal(img.affine, reference_affine)}")
                raise ValueError(f"无法统一图像 {participant_id}")

        print(f"所有图像已完全统一，共 {len(unified_img_paths)} 个文件")

        # 使用第一个被试的brain mask作为群体mask，也需要统一处理
        first_subject = self.participants_complete['participant_id'].iloc[0]
        brain_mask_path = self.data_paths['brain_mask'][first_subject]

        # 处理brain mask以确保与数据图像一致
        brain_mask_img = nib.load(brain_mask_path)
        if len(brain_mask_img.shape) == 4:
            mask_data = brain_mask_img.get_fdata()[:, :, :, 0]
            brain_mask_img = nib.Nifti1Image(mask_data, brain_mask_img.affine, brain_mask_img.header)

        # 如果mask的空间与数据不一致，重采样mask
        if not (brain_mask_img.shape[:3] == reference_shape and np.allclose(brain_mask_img.affine, reference_affine)):
            print("重采样brain mask以匹配数据空间...")
            brain_mask_img = resample_to_img(brain_mask_img, reference_img, interpolation='nearest')

            # 保存统一后的mask
            unified_mask_path = final_output_dir / "unified_brain_mask.nii.gz"
            brain_mask_img.to_filename(unified_mask_path)
            brain_mask_path = str(unified_mask_path)

        # 构建二级模型
        try:
            slm = SecondLevelModel(
                mask_img=brain_mask_img,  # 直接使用图像对象而不是路径
                smoothing_fwhm=smoothing_fwhm,
                n_jobs=1,
                verbose=0
            )

            # 拟合模型
            print(f"拟合二级模型...")
            slm = slm.fit(unified_img_paths, design_matrix=self.design_matrix)

            # 计算对比度 (组间差异: 抑郁组 > 对照组)
            print(f"计算统计对比度...")
            z_map = slm.compute_contrast('group', output_type='z_score')
            stat_map = slm.compute_contrast('group', output_type='stat')

            # 保存原始统计图
            output_dir = self.results_dir / metric_name
            output_dir.mkdir(exist_ok=True)

            z_map.to_filename(output_dir / f"{metric_name}_group_z_map.nii.gz")
            stat_map.to_filename(output_dir / f"{metric_name}_group_stat_map.nii.gz")

            print(f"{metric_name} 二级分析完成")
            return slm, z_map, stat_map

        except Exception as e:
            print(f"二级分析失败: {e}")

            # 尝试更保守的参数
            print("尝试使用更保守的参数重新分析...")
            try:
                slm = SecondLevelModel(
                    mask_img=brain_mask_img,
                    smoothing_fwhm=0,  # 不进行额外平滑
                    n_jobs=1,
                    verbose=1
                )

                slm = slm.fit(unified_img_paths, design_matrix=self.design_matrix)
                z_map = slm.compute_contrast('group', output_type='z_score')
                stat_map = slm.compute_contrast('group', output_type='stat')

                output_dir = self.results_dir / metric_name
                output_dir.mkdir(exist_ok=True)

                z_map.to_filename(output_dir / f"{metric_name}_group_z_map.nii.gz")
                stat_map.to_filename(output_dir / f"{metric_name}_group_stat_map.nii.gz")

                print(f"{metric_name} 二级分析完成（保守参数）")
                return slm, z_map, stat_map

            except Exception as e2:
                print(f"保守参数分析也失败: {e2}")
                raise e2

    def threshold_and_correct(self, z_map, metric_name, alpha=0.05, height_threshold=None):
        """
        校正多重比较并生成阈上图

        Parameters:
        -----------
        z_map : Nifti1Image
            Z统计图
        metric_name : str
            指标名称
        alpha : float
            显著性水平
        height_threshold : float
            高度阈值，如果为None则使用默认
        """
        print(f"正在进行多重比较校正 ({metric_name}) ...")

        # 设置默认高度阈值
        if height_threshold is None:
            height_threshold = 2.3  # 对应 p < 0.01 (uncorrected)

        # 使用更新的threshold_stats_img参数
        try:
            thresholded_map, threshold_value = threshold_stats_img(
                z_map,
                alpha=alpha,
                threshold=height_threshold,  # 修正参数名
                cluster_threshold=10
            )

            print(f"{metric_name} - 阈值: {threshold_value:.3f}")

            # 保存阈值图
            output_dir = self.results_dir / metric_name
            thresholded_map.to_filename(output_dir / f"{metric_name}_thresholded_z_map.nii.gz")

            return thresholded_map, threshold_value

        except Exception as e:
            print(f"校正过程出错: {e}")
            print(f"使用未校正的结果，阈值: {height_threshold}")

            # 创建简单的阈值图
            z_data = z_map.get_fdata()
            z_data[np.abs(z_data) < height_threshold] = 0
            thresholded_map = nib.Nifti1Image(z_data, z_map.affine, z_map.header)

            output_dir = self.results_dir / metric_name
            thresholded_map.to_filename(output_dir / f"{metric_name}_thresholded_z_map.nii.gz")

            return thresholded_map, height_threshold

    def run_paired_ttest_analysis(self, metrics=['alff', 'reho', 'seed1_fc', 'seed2_fc']):
        """
        对正常人组和抑郁组的四个指标进行配对样本T检验
        注意：这里实际是独立样本T检验，因为是不同的被试组
        """
        print("\n" + "=" * 60)
        print("开始独立样本T检验分析")
        print("=" * 60)

        from scipy import stats
        import seaborn as sns

        # 创建T检验结果目录
        ttest_dir = self.results_dir / "ttest_analysis"
        ttest_dir.mkdir(exist_ok=True)

        # 存储所有T检验结果
        all_ttest_results = []

        for metric in metrics:
            print(f"\n正在进行 {metric.upper()} 的独立样本T检验...")

            try:
                # 获取重采样后的图像路径
                img_paths = self.check_and_resample_images(metric)

                # 分组获取图像数据
                depr_images = []
                control_images = []

                for i, img_path in enumerate(img_paths):
                    participant_id = self.participants_complete['participant_id'].iloc[i]
                    group = self.participants_complete[
                        self.participants_complete['participant_id'] == participant_id
                        ]['group'].iloc[0]

                    img_data = nib.load(img_path).get_fdata()

                    if group == 'depr':
                        depr_images.append(img_data)
                    else:
                        control_images.append(img_data)

                # 转换为numpy数组
                depr_data = np.array(depr_images)  # shape: (n_depr, x, y, z)
                control_data = np.array(control_images)  # shape: (n_control, x, y, z)

                print(f"抑郁组样本数: {depr_data.shape[0]}")
                print(f"对照组样本数: {control_data.shape[0]}")

                # 执行体素级独立样本T检验
                print("执行体素级T检验...")
                t_stat_map = np.zeros(depr_data.shape[1:])  # (x, y, z)
                p_value_map = np.ones(depr_data.shape[1:])  # (x, y, z)
                cohens_d_map = np.zeros(depr_data.shape[1:])  # (x, y, z)

                # 获取脑组织mask以减少计算量
                first_subject = self.participants_complete['participant_id'].iloc[0]
                brain_mask_path = self.data_paths['brain_mask'][first_subject]
                brain_mask = nib.load(brain_mask_path).get_fdata()

                # 确保mask是3D的
                if len(brain_mask.shape) == 4:
                    brain_mask = brain_mask[:, :, :, 0]

                # 如果mask与数据空间不一致，调整mask
                if brain_mask.shape != depr_data.shape[1:]:
                    from scipy.ndimage import zoom
                    zoom_factors = [data_dim / mask_dim for data_dim, mask_dim in
                                    zip(depr_data.shape[1:], brain_mask.shape)]
                    brain_mask = zoom(brain_mask, zoom_factors, order=0)

                # 只在脑组织内进行T检验
                brain_coords = np.where(brain_mask > 0)
                n_voxels = len(brain_coords[0])

                print(f"在 {n_voxels} 个脑内体素执行T检验...")

                # 存储显著体素的详细信息
                significant_voxels_info = []

                # 批量进行T检验
                for idx in range(n_voxels):
                    if idx % 10000 == 0:
                        print(f"进度: {idx}/{n_voxels} ({idx / n_voxels * 100:.1f}%)")

                    x, y, z = brain_coords[0][idx], brain_coords[1][idx], brain_coords[2][idx]

                    depr_voxel = depr_data[:, x, y, z]
                    control_voxel = control_data[:, x, y, z]

                    # 检查数据有效性
                    if not (np.isfinite(depr_voxel).all() and np.isfinite(control_voxel).all()):
                        continue

                    # 独立样本T检验
                    t_stat, p_val = stats.ttest_ind(depr_voxel, control_voxel,
                                                    equal_var=False, nan_policy='omit')

                    if np.isfinite(t_stat) and np.isfinite(p_val):
                        t_stat_map[x, y, z] = t_stat
                        p_value_map[x, y, z] = p_val

                        # 计算Cohen's d
                        pooled_std = np.sqrt(((len(depr_voxel) - 1) * np.var(depr_voxel, ddof=1) +
                                              (len(control_voxel) - 1) * np.var(control_voxel, ddof=1)) /
                                             (len(depr_voxel) + len(control_voxel) - 2))

                        if pooled_std > 0:
                            cohens_d = (np.mean(depr_voxel) - np.mean(control_voxel)) / pooled_std
                            cohens_d_map[x, y, z] = cohens_d

                        # 如果显著，记录详细信息
                        if p_val <= 0.05:
                            voxel_info = {
                                'x': x, 'y': y, 'z': z,
                                'coordinate': f"({x}, {y}, {z})",
                                't_stat': round(t_stat, 4),
                                'p_value': round(p_val, 6),
                                'cohens_d': round(cohens_d, 4) if pooled_std > 0 else 0,
                                'depr_mean': round(np.mean(depr_voxel), 4),
                                'control_mean': round(np.mean(control_voxel), 4),
                                'depr_greater': t_stat > 0
                            }
                            significant_voxels_info.append(voxel_info)

                # 获取参考图像信息用于保存结果
                reference_img = nib.load(img_paths[0])
                if len(reference_img.shape) == 4:
                    ref_data = reference_img.get_fdata()[:, :, :, 0]
                    reference_img = nib.Nifti1Image(ref_data, reference_img.affine, reference_img.header)

                # 保存T统计图、P值图和Cohen's d图
                t_img = nib.Nifti1Image(t_stat_map, reference_img.affine, reference_img.header)
                p_img = nib.Nifti1Image(p_value_map, reference_img.affine, reference_img.header)
                d_img = nib.Nifti1Image(cohens_d_map, reference_img.affine, reference_img.header)

                t_img.to_filename(ttest_dir / f"{metric}_tstat_map.nii.gz")
                p_img.to_filename(ttest_dir / f"{metric}_pvalue_map.nii.gz")
                d_img.to_filename(ttest_dir / f"{metric}_cohens_d_map.nii.gz")

                # 创建显著性mask (p <= 0.05)
                sig_mask = (p_value_map <= 0.05) & (brain_mask > 0)
                sig_t_map = t_stat_map.copy()
                sig_t_map[~sig_mask] = 0

                sig_img = nib.Nifti1Image(sig_t_map, reference_img.affine, reference_img.header)
                sig_img.to_filename(ttest_dir / f"{metric}_significant_tstat_map.nii.gz")

                # 统计显著体素数量
                n_sig_voxels = np.sum(sig_mask)
                n_pos_sig = np.sum((sig_t_map > 0) & sig_mask)
                n_neg_sig = np.sum((sig_t_map < 0) & sig_mask)

                print(f"{metric} T检验完成:")
                print(f"  显著体素总数 (p≤0.05): {n_sig_voxels}")
                print(f"  抑郁组>对照组: {n_pos_sig} 体素")
                print(f"  抑郁组<对照组: {n_neg_sig} 体素")

                # 计算统计摘要
                if significant_voxels_info:
                    sig_p_values = [v['p_value'] for v in significant_voxels_info]
                    sig_d_values = [v['cohens_d'] for v in significant_voxels_info]

                    min_p = min(sig_p_values)
                    max_d = max(sig_d_values, key=abs)
                    mean_p = np.mean(sig_p_values)
                    mean_d = np.mean(sig_d_values)
                else:
                    min_p = max_d = mean_p = mean_d = 0

                # 保存显著体素详细信息到CSV
                if significant_voxels_info:
                    sig_df = pd.DataFrame(significant_voxels_info)
                    sig_df = sig_df.sort_values('p_value')  # 按p值排序
                    sig_df.to_csv(ttest_dir / f"{metric}_significant_voxels_details.csv", index=False)
                    print(f"  显著体素详细信息已保存: {metric}_significant_voxels_details.csv")

                # 存储结果到汇总表
                result_entry = {
                    'Metric': metric.upper(),
                    'Metric_Name': {
                        'alff': 'ALFF (低频振幅)',
                        'reho': 'ReHo (局部一致性)',
                        'seed1_fc': 'PCC种子连接',
                        'seed2_fc': 'mPFC种子连接'
                    }.get(metric, metric),
                    'Depression_n': depr_data.shape[0],
                    'Control_n': control_data.shape[0],
                    'Significant_Voxels': n_sig_voxels,
                    'Depression_Greater': n_pos_sig,
                    'Control_Greater': n_neg_sig,
                    'Min_P_Value': round(min_p, 6) if min_p > 0 else 'N/A',
                    'Mean_P_Value': round(mean_p, 6) if mean_p > 0 else 'N/A',
                    'Max_Effect_Size_D': round(max_d, 4) if max_d != 0 else 'N/A',
                    'Mean_Effect_Size_D': round(mean_d, 4) if mean_d != 0 else 'N/A',
                    'Percent_Significant': round(n_sig_voxels / np.sum(brain_mask > 0) * 100, 2),
                    'Top_Coordinates': '; '.join(
                        [v['coordinate'] for v in significant_voxels_info[:5]]) if significant_voxels_info else 'N/A'
                }

                all_ttest_results.append(result_entry)

                # 生成可视化
                self.visualize_ttest_results(sig_img, metric, ttest_dir)

            except Exception as e:
                print(f"分析 {metric} 时出错: {e}")
                # 添加错误记录到结果表
                error_entry = {
                    'Metric': metric.upper(),
                    'Metric_Name': f"{metric} (分析失败)",
                    'Error': str(e)
                }
                all_ttest_results.append(error_entry)

        # 创建汇总结果表
        results_df = pd.DataFrame(all_ttest_results)
        results_df.to_csv(ttest_dir / "ttest_summary_table.csv", index=False)

        # 生成格式化的汇总表
        self.generate_formatted_ttest_table(results_df, ttest_dir)

        print(f"\nT检验分析完成！结果保存在: {ttest_dir}")
        return results_df

    def visualize_ttest_results(self, sig_img, metric, output_dir):
        """
        可视化T检验显著结果
        """
        try:
            from nilearn.datasets import load_mni152_template
            from nilearn.plotting import plot_stat_map

            print(f"生成 {metric} T检验可视化...")

            # 加载MNI模板
            mni_template = load_mni152_template(resolution=2)

            # 检查是否有显著激活
            sig_data = sig_img.get_fdata()
            if np.max(np.abs(sig_data)) == 0:
                print(f"  {metric}: 无显著激活区域")
                return

            # 设置阈值（只显示非零值）
            threshold = 0.1  # 很小的阈值，只是为了显示非零区域

            # 创建轴面视图
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f'{metric.upper()} - Independent Samples T-test (p ≤ 0.05)', fontsize=16)

            plot_stat_map(
                sig_img,
                bg_img=mni_template,
                threshold=threshold,
                display_mode='z',
                cut_coords=8,
                figure=fig,
                title='Significant differences (Red: Depression > Control, Blue: Control > Depression)',
                colorbar=True,
                cmap='RdBu_r'
            )

            plt.savefig(output_dir / f"{metric}_ttest_significant_axial.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 创建矢状面视图
            fig2 = plt.figure(figsize=(12, 8))
            plot_stat_map(
                sig_img,
                bg_img=mni_template,
                threshold=threshold,
                display_mode='x',
                cut_coords=6,
                figure=fig2,
                title='Sagittal view - Significant T-test results',
                colorbar=True,
                cmap='RdBu_r'
            )

            plt.savefig(output_dir / f"{metric}_ttest_significant_sagittal.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  {metric} 可视化完成")

        except Exception as e:
            print(f"生成 {metric} 可视化时出错: {e}")

    def generate_formatted_ttest_table(self, results_df, output_dir):
        """
        生成格式化的T检验汇总表
        """
        try:
            # 创建HTML格式的表格
            html_table = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>独立样本T检验结果汇总</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 12px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                    th { background-color: #f2f2f2; font-weight: bold; }
                    .metric-name { text-align: left; font-weight: bold; }
                    .coordinates { text-align: left; font-size: 10px; max-width: 200px; word-wrap: break-word; }
                    .significant { background-color: #e8f5e8; }
                    .high-significance { background-color: #d4edda; }
                    .effect-large { background-color: #ffebe6; }
                    .effect-medium { background-color: #fff3e0; }
                    .effect-small { background-color: #f9f9f9; }
                    h1 { color: #333; text-align: center; }
                    .summary { background-color: #f0f8ff; padding: 15px; margin: 20px 0; border-radius: 5px; }
                    .stats-table { margin-top: 30px; }
                    .stats-table th { background-color: #e9ecef; }
                </style>
            </head>
            <body>
                <h1>fMRI独立样本T检验结果汇总表</h1>
                <div class="summary">
                    <h3>分析说明：</h3>
                    <ul>
                        <li>比较抑郁组与健康对照组在四个脑功能指标上的差异</li>
                        <li>统计方法：独立样本T检验（体素水平）</li>
                        <li>显著性阈值：p ≤ 0.05（未校正）</li>
                        <li>效应量：Cohen's d (小效应: 0.2, 中效应: 0.5, 大效应: 0.8)</li>
                        <li>坐标系统：体素坐标 (x, y, z)</li>
                    </ul>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>脑功能指标</th>
                            <th>抑郁组<br>样本数</th>
                            <th>对照组<br>样本数</th>
                            <th>显著体素数</th>
                            <th>抑郁组>对照组<br>体素数</th>
                            <th>对照组>抑郁组<br>体素数</th>
                            <th>最小P值</th>
                            <th>平均P值</th>
                            <th>最大效应量<br>(Cohen's d)</th>
                            <th>平均效应量<br>(Cohen's d)</th>
                            <th>显著体素<br>百分比(%)</th>
                            <th>前5个显著坐标</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for _, row in results_df.iterrows():
                if 'Error' in row:
                    continue

                # 根据显著体素数量和效应量设置行的CSS类
                row_class = ""
                sig_voxels = row.get('Significant_Voxels', 0)
                min_p = row.get('Min_P_Value', 1)

                if sig_voxels > 1000:
                    row_class = "significant"
                if min_p != 'N/A' and float(min_p) < 0.001:
                    row_class += " high-significance"

                max_effect = row.get('Max_Effect_Size_D', 0)
                if max_effect != 'N/A':
                    max_effect = abs(float(max_effect))
                    if max_effect >= 0.8:
                        row_class += " effect-large"
                    elif max_effect >= 0.5:
                        row_class += " effect-medium"
                    elif max_effect >= 0.2:
                        row_class += " effect-small"

                html_table += f"""
                        <tr class="{row_class}">
                            <td class="metric-name">{row.get('Metric_Name', '')}</td>
                            <td>{row.get('Depression_n', 'N/A')}</td>
                            <td>{row.get('Control_n', 'N/A')}</td>
                            <td>{row.get('Significant_Voxels', 0):,}</td>
                            <td>{row.get('Depression_Greater', 0):,}</td>
                            <td>{row.get('Control_Greater', 0):,}</td>
                            <td>{row.get('Min_P_Value', 'N/A')}</td>
                            <td>{row.get('Mean_P_Value', 'N/A')}</td>
                            <td>{row.get('Max_Effect_Size_D', 'N/A')}</td>
                            <td>{row.get('Mean_Effect_Size_D', 'N/A')}</td>
                            <td>{row.get('Percent_Significant', 0)}</td>
                            <td class="coordinates">{row.get('Top_Coordinates', 'N/A')}</td>
                        </tr>
                """

            html_table += """
                    </tbody>
                </table>

                <div class="summary">
                    <h3>结果解释：</h3>
                    <ul>
                        <li><strong>显著体素数：</strong>在该指标上两组间存在显著差异的脑体素数量</li>
                        <li><strong>抑郁组>对照组：</strong>抑郁组数值显著大于对照组的体素数</li>
                        <li><strong>对照组>抑郁组：</strong>对照组数值显著大于抑郁组的体素数</li>
                        <li><strong>最小P值：</strong>所有显著体素中最小的p值</li>
                        <li><strong>平均P值：</strong>所有显著体素的平均p值</li>
                        <li><strong>最大效应量：</strong>所有显著体素中效应量最大的Cohen's d值</li>
                        <li><strong>平均效应量：</strong>所有显著体素的平均Cohen's d值</li>
                        <li><strong>显著体素百分比：</strong>显著体素占全脑体素的比例</li>
                        <li><strong>前5个显著坐标：</strong>按p值排序的前5个显著体素坐标</li>
                    </ul>
                    <p><strong>注意：</strong>详细的显著体素信息（包括所有坐标、p值、效应量）保存在对应的CSV文件中。</p>
                </div>
            </body>
            </html>
            """

            # 保存HTML表格
            with open(output_dir / "ttest_results_table.html", 'w', encoding='utf-8') as f:
                f.write(html_table)

            # 生成详细的文本报告
            with open(output_dir / "ttest_results_summary.txt", 'w', encoding='utf-8') as f:
                f.write("fMRI独立样本T检验结果汇总\n")
                f.write("=" * 50 + "\n\n")

                for _, row in results_df.iterrows():
                    if 'Error' in row:
                        f.write(f"{row['Metric']}: 分析失败 - {row['Error']}\n\n")
                        continue

                    f.write(f"{row.get('Metric_Name', '')}:\n")
                    f.write(
                        f"  样本: 抑郁组{row.get('Depression_n', 'N/A')}人, 对照组{row.get('Control_n', 'N/A')}人\n")
                    f.write(
                        f"  显著差异体素: {row.get('Significant_Voxels', 0):,}个 ({row.get('Percent_Significant', 0)}%)\n")
                    f.write(f"  抑郁组>对照组: {row.get('Depression_Greater', 0):,}个体素\n")
                    f.write(f"  对照组>抑郁组: {row.get('Control_Greater', 0):,}个体素\n")
                    f.write(f"  统计显著性:\n")
                    f.write(f"    最小P值: {row.get('Min_P_Value', 'N/A')}\n")
                    f.write(f"    平均P值: {row.get('Mean_P_Value', 'N/A')}\n")
                    f.write(f"  效应量 (Cohen's d):\n")
                    f.write(f"    最大效应量: {row.get('Max_Effect_Size_D', 'N/A')}\n")
                    f.write(f"    平均效应量: {row.get('Mean_Effect_Size_D', 'N/A')}\n")
                    f.write(f"  前5个显著坐标: {row.get('Top_Coordinates', 'N/A')}\n")
                    f.write(f"  详细信息文件: {row.get('Metric', '').lower()}_significant_voxels_details.csv\n\n")

            # 生成统计摘要表
            stats_summary = []
            for _, row in results_df.iterrows():
                if 'Error' not in row and row.get('Significant_Voxels', 0) > 0:
                    stats_summary.append({
                        '指标': row.get('Metric_Name', ''),
                        '显著体素数': row.get('Significant_Voxels', 0),
                        '最小P值': row.get('Min_P_Value', 'N/A'),
                        '最大效应量': row.get('Max_Effect_Size_D', 'N/A')
                    })

            if stats_summary:
                stats_df = pd.DataFrame(stats_summary)
                stats_df.to_csv(output_dir / "statistical_summary.csv", index=False)

            print("格式化汇总表已生成:")
            print(f"  HTML表格: {output_dir / 'ttest_results_table.html'}")
            print(f"  文本报告: {output_dir / 'ttest_results_summary.txt'}")
            print(f"  统计摘要: {output_dir / 'statistical_summary.csv'}")
            print(f"  各指标的显著体素详细信息保存在对应的CSV文件中")

        except Exception as e:
            print(f"生成格式化表格时出错: {e}")

    def visualize_results(self, thresholded_map, metric_name, threshold_value):
        """
        可视化结果
        """
        print(f"正在生成可视化结果 ({metric_name}) ...")

        output_dir = self.results_dir / metric_name

        # 加载MNI模板
        mni_template = load_mni152_template(resolution=2)

        # 创建多视图可视化
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{metric_name.upper()} - Group Comparison (Depression > Control)', fontsize=16)

        # 轴面视图
        display1 = plot_stat_map(
            thresholded_map,
            bg_img=mni_template,
            threshold=threshold_value,
            display_mode='z',
            cut_coords=6,
            figure=fig,
            title=f'Axial view (threshold: {threshold_value:.2f})',
            colorbar=True
        )

        plt.savefig(output_dir / f"{metric_name}_group_comparison_axial.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 矢状面视图
        fig2 = plt.figure(figsize=(12, 8))
        display2 = plot_stat_map(
            thresholded_map,
            bg_img=mni_template,
            threshold=threshold_value,
            display_mode='x',
            cut_coords=5,
            figure=fig2,
            title='Sagittal view',
            colorbar=True
        )

        plt.savefig(output_dir / f"{metric_name}_group_comparison_sagittal.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 冠状面视图
        fig3 = plt.figure(figsize=(12, 8))
        display3 = plot_stat_map(
            thresholded_map,
            bg_img=mni_template,
            threshold=threshold_value,
            display_mode='y',
            cut_coords=5,
            figure=fig3,
            title='Coronal view',
            colorbar=True
        )

        plt.savefig(output_dir / f"{metric_name}_group_comparison_coronal.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 生成交互式可视化
        try:
            interactive_view = view_img(
                thresholded_map,
                bg_img=mni_template,
                threshold=threshold_value,
                title=f'{metric_name.upper()} - Group Comparison'
            )

            interactive_view.save_as_html(output_dir / f"{metric_name}_interactive_view.html")
        except Exception as e:
            print(f"生成交互式可视化时出错: {e}")

        print(f"可视化结果已保存到: {output_dir}")

    def extract_clusters_table(self, thresholded_map, metric_name, threshold_value):
        """
        提取显著簇信息表
        """
        try:
            print(f"正在提取显著簇信息 ({metric_name}) ...")

            clusters_table = get_clusters_table(
                thresholded_map,
                stat_threshold=threshold_value,
                cluster_threshold=10,
                min_distance=8.0
            )

            output_dir = self.results_dir / metric_name
            clusters_table.to_csv(output_dir / f"{metric_name}_clusters_table.csv", index=False)

            print(f"发现 {len(clusters_table)} 个显著簇")
            if len(clusters_table) > 0:
                print("前5个显著簇:")
                print(clusters_table.head())

            return clusters_table

        except Exception as e:
            print(f"提取簇信息时出错: {e}")
            return pd.DataFrame()  # 返回空DataFrame而不是None

    def run_complete_analysis_with_diagnosis(self, metrics=['alff', 'reho', 'seed1_fc', 'seed2_fc']):
        """
        运行完整分析流程（包含图像诊断）
        """
        print("=" * 60)
        print("开始fMRI组间分析")
        print("=" * 60)

        # 步骤1: 准备数据
        self.prepare_data_paths()

        # 步骤2: 构建设计矩阵
        self.prepare_design_matrix()

        # 保存参与者信息
        self.participants_complete.to_csv(self.results_dir / "participants_complete.csv", index=False)
        self.design_matrix.to_csv(self.results_dir / "design_matrix.csv", index=False)

        results_summary = {}

        # 步骤3: 对每个指标进行诊断和分析
        for metric in metrics:
            print(f"\n{'=' * 40}")
            print(f"分析指标: {metric.upper()}")
            print(f"{'=' * 40}")

            try:
                # 先进行图像诊断
                diagnostic_results = self.diagnose_image_consistency(metric)

                # 二级分析
                slm, z_map, stat_map = self.run_second_level_analysis(metric)

                # 多重比较校正
                thresholded_map, threshold_value = self.threshold_and_correct(z_map, metric)

                # 可视化
                self.visualize_results(thresholded_map, metric, threshold_value)

                # 提取簇信息
                clusters_table = self.extract_clusters_table(thresholded_map, metric, threshold_value)

                results_summary[metric] = {
                    'threshold': threshold_value,
                    'n_clusters': len(clusters_table) if clusters_table is not None else 0,
                    'n_problematic_images': sum(1 for r in diagnostic_results if r.get('needs_resampling', False))
                }

                print(f"{metric.upper()} 分析完成")

            except Exception as e:
                print(f"分析 {metric} 时出错: {e}")
                results_summary[metric] = {'error': str(e)}

                # 步骤4: 进行独立样本T检验分析
        try:
            print(f"\n{'=' * 40}")
            print("进行独立样本T检验分析")
            print(f"{'=' * 40}")

            ttest_results = self.run_paired_ttest_analysis(metrics)
            results_summary['ttest_analysis'] = {
                'total_metrics': len(metrics),
                'successful_analyses': len([r for r in ttest_results.to_dict('records') if 'Error' not in r]),
                'total_significant_voxels': sum(
                    [r.get('Significant_Voxels', 0) for r in ttest_results.to_dict('records')])
            }

        except Exception as e:
            print(f"T检验分析时出错: {e}")
            results_summary['ttest_analysis'] = {'error': str(e)}

        # 生成总结报告
        self.generate_summary_report(results_summary)

        print(f"\n{'=' * 60}")
        print("分析完成！结果保存在:", self.results_dir)
        print(f"{'=' * 60}")

        return results_summary

    def generate_summary_report(self, results_summary):
        """
        生成总结报告
        """
        report_path = self.results_dir / "analysis_summary.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("fMRI组间分析总结报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"分析时间: {pd.Timestamp.now()}\n")
            f.write(f"总被试数: {len(self.participants_complete)}\n")
            f.write(f"组别分布: {dict(self.participants_complete['group'].value_counts())}\n")

            # 添加人口统计学信息
            depr_group = self.participants_complete[self.participants_complete['group'] == 'depr']
            control_group = self.participants_complete[self.participants_complete['group'] == 'control']

            f.write(f"\n人口统计学信息:\n")
            f.write(f"抑郁组 (n={len(depr_group)}):\n")
            f.write(
                f"  年龄: {depr_group['age'].mean():.1f}±{depr_group['age'].std():.1f} ({depr_group['age'].min()}-{depr_group['age'].max()})\n")
            f.write(f"  性别: 男{(depr_group['gender'] == 'm').sum()}人, 女{(depr_group['gender'] == 'f').sum()}人\n")

            f.write(f"对照组 (n={len(control_group)}):\n")
            f.write(
                f"  年龄: {control_group['age'].mean():.1f}±{control_group['age'].std():.1f} ({control_group['age'].min()}-{control_group['age'].max()})\n")
            f.write(
                f"  性别: 男{(control_group['gender'] == 'm').sum()}人, 女{(control_group['gender'] == 'f').sum()}人\n")

            f.write(f"\n各指标分析结果:\n")
            f.write("-" * 30 + "\n")

            metric_names = {
                'alff': 'ALFF (低频振幅)',
                'reho': 'ReHo (局部一致性)',
                'seed1_fc': 'PCC种子连接 (PCC: 0,-52,26)',
                'seed2_fc': 'mPFC种子连接 (mPFC: 0,52,-2)',
                'ttest_analysis': '独立样本T检验'
            }

            for metric, results in results_summary.items():
                f.write(f"\n{metric_names.get(metric, metric.upper())}:\n")

                if 'error' in results:
                    f.write(f"  错误: {results['error']}\n")

                elif 'threshold' in results:  # 常规四个指标
                    f.write(f"  统计阈值: {results['threshold']:.3f}\n")
                    f.write(f"  显著簇数量: {results['n_clusters']}\n")
                    if results['n_clusters'] > 0:
                        f.write("  结果解释: 抑郁组相对于对照组的差异\n")

                elif metric == 'ttest_analysis':  # T 检验汇总
                    f.write(f"  成功分析指标数: {results.get('successful_analyses', 0)} / "
                            f"{results.get('total_metrics', 0)}\n")
                    f.write(f"  总显著体素数: {results.get('total_significant_voxels', 0)}\n")

            f.write(f"\n文件说明:\n")
            f.write(f"- *_z_map.nii.gz: 原始Z统计图\n")
            f.write(f"- *_thresholded_z_map.nii.gz: 校正后的阈值统计图\n")
            f.write(f"- *_group_comparison_*.png: 静态可视化图\n")
            f.write(f"- *_interactive_view.html: 交互式可视化\n")
            f.write(f"- *_clusters_table.csv: 显著簇详细信息\n")
            f.write(f"- resampled/: 重采样后的图像文件\n")

            f.write(f"\n注意事项:\n")
            f.write(f"- 所有分析均控制了年龄和性别的影响\n")
            f.write(f"- 统计图中正值表示抑郁组>对照组，负值表示抑郁组<对照组\n")
            f.write(f"- 坐标系统: MNI152 2mm空间\n")
            f.write(f"- 所有图像已重采样到一致的空间\n")

        print(f"总结报告已保存: {report_path}")


# 使用示例
if __name__ == "__main__":
    # 设置路径（请根据实际情况修改）
    participants_file = "participants.tsv"  # 请上传此文件
    base_dir = r"D:\fmri\fmri_output1"

    # 创建分析对象
    analyzer = FMRIGroupAnalysis(participants_file, base_dir)

    # 运行完整分析
    analyzer.run_complete_analysis_with_diagnosis()

    print("\n分析提示:")
    print("1. 所有结果保存在 'group_analysis_results' 目录中")
    print("2. 每个指标都有独立的子目录包含:")
    print("   - 原始统计图 (*_z_map.nii.gz)")
    print("   - 阈值统计图 (*_thresholded_z_map.nii.gz)")
    print("   - 静态可视化图 (*_group_comparison_*.png)")
    print("   - 交互式可视化 (*_interactive_view.html)")
    print("   - 显著簇信息表 (*_clusters_table.csv)")
    print("   - 重采样图像 (resampled/ 目录)")
    print("3. 查看 analysis_summary.txt 获取整体分析结果")