# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import glob
import sys
import torch

from PIL import Image

sys.path.append('..')
from similarities import ImageHashSimilarity, SiftSimilarity, ClipSimilarity


def sim_and_search(m):
    sim_scores = m.similarity(imgs1, corpus_imgs)
    print('sim scores: ', sim_scores)
    
    mean_score = torch.mean(sim_scores)
    print(f"The mean similarity score is: {mean_score.item()}")
    
    


if __name__ == "__main__":
    image_fps1 = ['data/image1.png']
#     image_fps2 = ['data/image12-like-image1.png', 'data/image10.png', 'data/image3.png']
    imgs1 = [Image.open(i) for i in image_fps1]
#     imgs2 = [Image.open(i) for i in image_fps2]
    corpus_fps = glob.glob('cutout1x/*.png')
    corpus_imgs = [Image.open(i) for i in corpus_fps]
#     print(corpus_imgs)
#     exit()

    # 2. image and image similarity score
    sim_and_search(ClipSimilarity())  # the best result
#     sim_and_search(ImageHashSimilarity(hash_function='phash'))
#     sim_and_search(SiftSimilarity())
