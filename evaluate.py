import os
import numpy as np
from torch.distributions import Beta
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
import torch

def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def mse(tf_img1, tf_img2):
    return mean_squared_error(tf_img1, tf_img2)


def psnr(tf_img1, tf_img2):
    return peak_signal_noise_ratio(tf_img1, tf_img2, data_range=1)


def ssim(tf_img1, tf_img2):
    return structural_similarity(tf_img1, tf_img2)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


net = 'RMIGANwoMI'
location = 'output/'
direction = 'A2B'


def main():
    path = net + '/' + location + direction + '/'
    mode = net + '_' + direction
    file_list = os.listdir(path)
    original_list = []
    contrast_list = []
    for i, file in enumerate(file_list):
        if i % 2 == 0:
            original_list.append(file)
        else:
            contrast_list.append(file)
    list_psnr = []
    list_ssim = []
    list_mse = []
    with open('result.txt', 'a') as f:
        for i, item in enumerate(original_list):
            t1 = read_img(os.path.join(path, item))
            t2 = read_img(os.path.join(path, contrast_list[i]))
            result1 = np.zeros(t1.shape, dtype=np.float32)
            result2 = np.zeros(t2.shape, dtype=np.float32)
            cv2.normalize(t1, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.normalize(t2, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            w = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
            w1 = w.sample().item()
            print(f'w1{w1}')
            #w2 = w.sample().item()
            result2 = result1 * 0.1 + result2 * 0.9
            mse_num = mse(result1, result2)
            psnr_num = psnr(result1, result2)
            ssim_num = ssim(result1, result2)
            list_psnr.append(psnr_num)
            list_ssim.append(ssim_num)
            list_mse.append(mse_num)
            # 输出每张图像的指标：
            print()
            print("image: " + str(item) + ' & ' + str(contrast_list[i]))
            print("PSNR:" + str(psnr_num))
            print("SSIM:" + str(ssim_num))
            print("MSE:" + str(mse_num))
        # 输出平均指标：
        print()
        print(mode + "  PSNR:" + str(np.mean(list_psnr)))
        print(mode + "  SSIM:" + str(np.mean(list_ssim)))
        print(mode + "  MSE:" + str(np.mean(list_mse)))
        f.write(mode + "  PSNR:" + str(np.mean(list_psnr)))
        f.write('\n')
        f.write(mode + "  SSIM:" + str(np.mean(list_ssim)))
        f.write('\n')
        f.write(mode + "  MSE:" + str(np.mean(list_mse)))
        f.write('\n')
        f.write('\n')


if __name__ == '__main__':
    main()
