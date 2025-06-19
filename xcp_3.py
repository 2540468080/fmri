#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单被试层面：
  • 计算 ReHo（已改为输出 z-score，脑外 NaN）
  • 计算 ALFF（FFT 幅值先 /t，再做被试内归一化，脑外 NaN）
  • Seed FC、Atlas-mPFC FC 保持原逻辑
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import input_data, datasets
from nilearn.image import load_img, resample_to_img, math_img
from nilearn.image.resampling import coord_transform
from scipy.stats import rankdata
from scipy.fft import fft
from joblib import Parallel, delayed
from tqdm import tqdm

# -------------------- 通用工具 -------------------- #
def resample_mask_to_bold(mask_file, bold_file):
    mask_img = load_img(mask_file)
    bold_img = load_img(bold_file)
    return resample_to_img(
        mask_img, bold_img, interpolation="nearest",
        force_resample=True, copy_header=True
    )

# -------------------- ReHo -------------------- #
def compute_reho(bold_file, mask_file, n_jobs=-1):
    bold_img       = load_img(bold_file)
    resampled_mask = resample_mask_to_bold(mask_file, bold_file)

    data = bold_img.get_fdata()
    mask = resampled_mask.get_fdata().astype(bool)

    x, y, z, t = data.shape
    reho_map = np.full((x, y, z), np.nan, dtype=np.float32)

    # 中心体素需在脑内；其 26 个邻居不强制全在脑内（REST 的做法）
    voxels = [(i, j, k) for i in range(1, x-1)
                          for j in range(1, y-1)
                          for k in range(1, z-1)
                          if mask[i, j, k]]

    def voxel_reho(i, j, k):
        # 取 3×3×3 小方块（含自身共 ≤27 个体素）
        nb_ts = data[i-1:i+2, j-1:j+2, k-1:k+2, :]

        # 用同尺寸掩模去掉脑外体素
        nb_mask = mask[i-1:i+2, j-1:j+2, k-1:k+2]
        nb_ts   = nb_ts[nb_mask, :]              # shape:(n_vox, t)
        n = nb_ts.shape[0]
        if n < 2:
            return i, j, k, 0.0

        # 正确的排名：在 **每个时间点 t** 上，对 n_vox 体素排序
        ranks = np.apply_along_axis(rankdata, 0, nb_ts)  # axis=0 !!!

        # Kendall W 公式
        Ri = np.sum(ranks, axis=1)                       # 每个体素秩次和
        S  = np.sum((Ri - Ri.mean())**2)
        W  = 12 * S / (n**2 * (t**3 - t))
        return i, j, k, W

    results = Parallel(n_jobs=n_jobs)(
        delayed(voxel_reho)(i, j, k) for i, j, k in tqdm(voxels, desc="ReHo 计算中")
    )
    # ===== ① 打印 Kendall W 原始数值范围 =====
    W_vals = np.array([w for _, _, _, w in results], dtype=np.float32)
    print(f"[ReHo debug] W  min/mean/max: {W_vals.min():.4f} / "
          f"{W_vals.mean():.4f} / {W_vals.max():.4f}")

    for i, j, k, w in results:
        reho_map[i, j, k] = w

    # --- 温和归一化：除以脑内均值 ---
    valid_idx = mask & np.isfinite(reho_map)
    mean_reho = reho_map[valid_idx].mean()
    reho_map[valid_idx] /= mean_reho

    # 可选：打印一下范围
    vals = reho_map[valid_idx]
    print(f"[ReHo debug] 归一化后 min/mean/max: {vals.min():.2f}/{vals.mean():.2f}/{vals.max():.2f}")

    return nib.Nifti1Image(reho_map, affine=bold_img.affine)


# -------------------- ALFF -------------------- #
def compute_alff(
    bold_file, mask_file,
    tr        = 2.0,
    low_freq  = 0.01,
    high_freq = 0.1,
    n_jobs    = -1
):
    """
    1. 先把每个体素的时间序列 z-score；
    2. FFT 取幅值后在目标频段求均值，得到 ALFF；
    3. 不再 /t，也不再除脑内均值，只保留 0.5~1.5 左右的原值域；
    4. 打印三次数值范围：FFT 频段前、归一化后（如果启用）。
    """
    bold_img       = load_img(bold_file)
    resampled_mask = resample_mask_to_bold(mask_file, bold_file)

    data = bold_img.get_fdata()
    mask = resampled_mask.get_fdata().astype(bool)
    x, y, z, t = data.shape

    # 频率索引
    freqs      = np.fft.fftfreq(t, d=tr)
    band_mask  = (freqs > low_freq) & (freqs < high_freq)

    # 结果数组（脑外 NaN）
    alff_map = np.full((x, y, z), np.nan, dtype=np.float32)

    voxels = [(i, j, k) for i in range(x)
                          for j in range(y)
                          for k in range(z) if mask[i, j, k]]

    def voxel_alff(i, j, k):
        ts = data[i, j, k, :].astype(np.float32)
        std = ts.std()
        if std == 0:
            return i, j, k, np.nan
        ts = (ts - ts.mean()) / std                 # z-score
        power = np.abs(fft(ts))
        return i, j, k, power[band_mask].mean()

    # --- 计算 ALFF ---
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(voxel_alff)(i, j, k)
        for i, j, k in tqdm(voxels, desc="ALFF 计算中")
    )

    for i, j, k, a in results:
        alff_map[i, j, k] = a

    # --- ① 打印原始 ALFF 值域 ---
    raw_vals = alff_map[mask & np.isfinite(alff_map)]
    print(f"[ALFF debug] raw  min/mean/max: "
          f"{raw_vals.min():.4f}/{raw_vals.mean():.4f}/{raw_vals.max():.4f}")

    # === 可选：被试内归一化（若想让脑内均值 = 1） ===
    # 建议只启用一次，不要再除以 t
    do_norm = True     # ← 若需归一化，把 False 改 True
    if do_norm:
        norm = raw_vals.mean()
        if norm > 0:
            alff_map[mask] /= norm
            norm_vals = alff_map[mask & np.isfinite(alff_map)]
            print(f"[ALFF debug] norm min/mean/max: "
                  f"{norm_vals.min():.4f}/{norm_vals.mean():.4f}/{norm_vals.max():.4f}")

    return nib.Nifti1Image(alff_map, affine=bold_img.affine)

# -------------------- Seed FC -------------------- #
def compute_seed_fc(bold_file, mask_file, confounds_file, seeds, tr=2.0):
    bold_img = load_img(bold_file)
    resampled_mask = resample_mask_to_bold(mask_file, bold_file)

    # ——1. 过滤掩模外的种子 —— #
    def filter_seeds(seeds_3d, mask_img):
        affine = mask_img.affine
        data = mask_img.get_fdata()
        kept = []
        for seed in seeds_3d:
            i, j, k = np.round(coord_transform(*seed, np.linalg.inv(affine))).astype(int)
            if (0 <= i < data.shape[0]) and (0 <= j < data.shape[1]) and (0 <= k < data.shape[2]):
                if data[i, j, k]:
                    kept.append(seed)
                else:
                    print(f"[!] Seed {seed} 在掩模之外")
            else:
                print(f"[!] Seed {seed} 超出 FOV")
        return kept

    seeds_in = filter_seeds(seeds, resampled_mask)
    if not seeds_in:
        raise ValueError("所有种子点都在掩模之外")

    # ——2. 读取伪移位/旋转等混淆 —— #
    confounds = pd.read_csv(confounds_file, sep='\t')
    conf = confounds[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]].fillna(0)

    # ——3. Seed-ts & Whole-brain ts —— #
    seed_masker = input_data.NiftiSpheresMasker(
        seeds_in, radius=6, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=tr, mask_img=resampled_mask
    )
    seed_ts = seed_masker.fit_transform(bold_img, confounds=conf)

    brain_masker = input_data.NiftiMasker(
        mask_img=resampled_mask, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=tr
    )
    brain_ts = brain_masker.fit_transform(bold_img, confounds=conf)

    # ——4. 相关图 —— #
    # ---------- 修正版相关计算 ----------
    fc_imgs = []
    n_time = brain_ts.shape[0]  # 时间点数 (TR 数)

    for i in range(seed_ts.shape[1]):  # 对每个种子
        # 1) 计算皮尔逊相关 r
        r = np.dot(brain_ts.T, seed_ts[:, i]) / n_time  # shape: (n_vox,)

        # 2) 防止 r = ±1 导致 atanh → inf
        r = np.clip(r, -0.999, 0.999)

        # 3) Fisher-z 变换，让分布更接近正态
        z = np.arctanh(r)  # shape: (n_vox,)

        # 4) 可选调试打印
        print(f"[Seed{i + 1} debug] r  min/mean/max:",
              r.min(), r.mean(), r.max())
        print(f"[Seed{i + 1} debug] z  min/mean/max:",
              z.min(), z.mean(), z.max())

        # 5) 还原成 3D 图并保存到列表
        fc_imgs.append(brain_masker.inverse_transform(z))

    return fc_imgs

# -------------------- ROI FC (Atlas) -------------------- #
def compute_roi_fc_from_atlas(bold_file, confounds_file, mask_file, tr=2.0):
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img, labels = load_img(atlas.maps), atlas.labels

    label_name = 'Frontal Medial Cortex'
    label_idx = labels.index(label_name)
    mPFC_mask = math_img(f"img == {label_idx}", img=atlas_img)

    confounds = pd.read_csv(confounds_file, sep='\t')
    conf = confounds[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]].fillna(0)

    mPFC_masker = input_data.NiftiMasker(
        mask_img=mPFC_mask, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=tr
    )
    mPFC_ts = mPFC_masker.fit_transform(bold_file, confounds=conf)

    resampled_mask = resample_mask_to_bold(mask_file, bold_file)
    brain_masker = input_data.NiftiMasker(
        mask_img=resampled_mask, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=tr
    )
    brain_ts = brain_masker.fit_transform(bold_file, confounds=conf)

    fc_map = np.dot(brain_ts.T, mPFC_ts[:, 0]) / mPFC_ts.shape[0]
    return brain_masker.inverse_transform(fc_map)

# -------------------- 批处理主程序 -------------------- #
if __name__ == '__main__':
    base_dir = 'D:/fmri/fmri_output1'

    subject_dirs = sorted([
        d for d in os.listdir(base_dir)
        if d.startswith('sub-') and os.path.isdir(os.path.join(base_dir, d))
    ])

    # 这里示例只跑两名受试者，可按需改为 subject_dirs[:] 或别的切片
    subjects = subject_dirs

    for sid in subjects:
        print(f"\n处理受试者: {sid}")
        func_dir = os.path.join(base_dir, sid, 'func')

        bold_file = os.path.join(func_dir, f'{sid}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        mask_file = os.path.join(func_dir, f'{sid}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
        conf_file = os.path.join(func_dir, f'{sid}_task-rest_desc-confounds_timeseries.tsv')

        if not all(map(os.path.exists, [bold_file, mask_file, conf_file])):
            print(f"[!] {sid} 缺文件，跳过")
            continue

        # —— ReHo —— #
        print("计算 ReHo...")
        reho_img = compute_reho(bold_file, mask_file)
        reho_img.to_filename(os.path.join(func_dir, f'{sid}_reho.nii.gz'))

        # —— ALFF —— #
        print("计算 ALFF...")
        alff_img = compute_alff(bold_file, mask_file)
        alff_img.to_filename(os.path.join(func_dir, f'{sid}_alff.nii.gz'))

        # —— Seed FC —— #
        print("计算 Seed FC...")
        seeds = [(0, -52, 26), (0, 52, -2)]
        try:
            fc_imgs = compute_seed_fc(bold_file, mask_file, conf_file, seeds)
            for i, fc in enumerate(fc_imgs, 1):
                fc.to_filename(os.path.join(func_dir, f'{sid}_seed{i}_fc_map.nii.gz'))
        except ValueError as e:
            print(f"[!] {e}，跳过该受试者 Seed FC")

        # —— ROI FC —— #
        print("计算 mPFC ROI FC...")
        roi_fc = compute_roi_fc_from_atlas(bold_file, conf_file, mask_file)
        roi_fc.to_filename(os.path.join(func_dir, f'{sid}_atlas_mPFC_fc_map.nii.gz'))

        print(f"受试者 {sid} 计算完成！")

    print("\n所有受试者计算完成！")
