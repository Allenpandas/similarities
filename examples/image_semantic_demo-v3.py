# -*- coding: utf-8 -*-
"""
Code modified to calculate similarity between a fixed image A and all images in a directory B.
"""
import glob
import os
from PIL import Image
import torch
import numpy as np

# 假设 ImageHashSimilarity, SiftSimilarity, ClipSimilarity 等类已正确导入
from similarities import ImageHashSimilarity, SiftSimilarity, ClipSimilarity

# 计算相似度的函数
def calculate_similarity(image_path_a, image_paths_b):
    img_a = Image.open(image_path_a)
    sim_scores = []

    # 遍历目录B中的所有图片
    for image_path_b in image_paths_b:
        img_b = Image.open(image_path_b)
        sim_score = ClipSimilarity().similarity(img_a, img_b)
        sim_scores.append(sim_score[0][0])  # 假设我们只关心第一个得分
        print(f'sim scores between {image_path_a} and {image_path_b}: ', sim_score[0][0])

    return sim_scores

def main(a_path, b_dir):
    # 存储相似度值
    similarities = []

    # 确保A是单个文件路径
    if not os.path.isfile(a_path):
        print(f"Error: {a_path} is not a valid file path.")
        return

    # 获取目录B下所有图片的路径
    image_paths_b = glob.glob(os.path.join(b_dir, "*.png"))  # 假设图片格式为PNG

    # 计算A和B目录下所有图片的相似度
    sim_scores = calculate_similarity(a_path, image_paths_b)
    similarities.extend(sim_scores)

    # 打印并保存结果
    print("All similarity scores:", similarities)


    # 如果有相似度得分，计算平均值
    if similarities:
        stacked_tensor = torch.tensor(similarities)
        mean_value = torch.mean(stacked_tensor)
        print("Mean similarity score:", mean_value.item())
    else:
        print("No similarities calculated.")

if __name__ == "__main__":
    # 替换以下变量为你的图片路径和目录
    A_PATH = '/home/wuyalun/project/similarities/examples/aaai-25/org/原始图片_540-360.png'
    B_DIR = '/home/wuyalun/project/similarities/examples/aaai-25/img2img'

    main(A_PATH, B_DIR)