#tai thu vien pip install opencv-python numpy scikit-image
import os
import cv2
import numpy as np


def approx_multiply(a, b):
   
    a_int = int(round(a))
    b_int = int(round(b))
    a_approx = a_int & ~0b11
    b_approx = b_int & ~0b11
    return float(a_approx * b_approx)


approx_mult_vec = np.vectorize(approx_multiply)



#Gausian blur 3x3
def task_gaussian_3x3(img):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0
    
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    img_pad = np.pad(img, 1, mode='edge').astype(np.float32)
    
    for r in range(h):
        for c in range(w):
            region = img_pad[r:r+3, c:c+3]
            out[r, c] = np.sum(approx_mult_vec(region, kernel))
            
    return np.clip(out, 0, 255).astype(np.uint8)


#sharpen 3x3
def task_sharpen_3x3(img):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    img_pad = np.pad(img, 1, mode='edge').astype(np.float32)
    
    for r in range(h):
        for c in range(w):
            region = img_pad[r:r+3, c:c+3]
            out[r, c] = np.sum(approx_mult_vec(region, kernel))
            
    return np.clip(out, 0, 255).astype(np.uint8)


#dct 8x8
def get_dct_matrix_8x8():
    C = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            if i == 0:
                C[i, j] = 1.0 / np.sqrt(8)
            else:
                C[i, j] = np.sqrt(2.0 / 8) * np.cos((2 * j + 1) * i * np.pi / 16.0)
    return C

DCT_MAT = get_dct_matrix_8x8()

def matrix_mult_approx(A, B):
    N, M = A.shape
    M2, K = B.shape
    C = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        for j in range(K):
            C[i, j] = np.sum(approx_mult_vec(A[i, :], B[:, j]))
    return C

def task_jpeg_dct_8x8(img):
    h, w = img.shape
    h_8 = (h // 8) * 8
    w_8 = (w // 8) * 8
    
    img_cropped = img[:h_8, :w_8].astype(np.float32) - 128.0
    out_img = np.zeros((h_8, w_8), dtype=np.float32)
    
    C = DCT_MAT
    C_T = DCT_MAT.T
    
    for r in range(0, h_8, 8):
        for c in range(0, w_8, 8):
            block = img_cropped[r:r+8, c:c+8]
            
            # Forward DCT: D = C * block * C_T
            temp1 = matrix_mult_approx(C, block)
            dct_block = matrix_mult_approx(temp1, C_T)
            
            # Inverse DCT: R = C_T * dct_block * C
            temp2 = matrix_mult_approx(C_T, dct_block)
            idct_block = matrix_mult_approx(temp2, C)
            
            out_img[r:r+8, c:c+8] = idct_block
            
    out_img = out_img + 128.0
    return np.clip(out_img, 0, 255).astype(np.uint8)


#in ket qua
if __name__ == "__main__":
    IMAGE_PATH = "input.png"  
    #random image neu chua co file png
    if not os.path.exists(IMAGE_PATH):
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    else:
        img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    print("Đang xử lý Tác vụ 1: Gaussian 3x3...")
    res_gaussian = task_gaussian_3x3(img)
    cv2.imwrite("out_gaussian.png", res_gaussian)

    print("Đang xử lý Tác vụ 2: Sharpen 3x3...")
    res_sharpen = task_sharpen_3x3(img)
    cv2.imwrite("out_sharpen.png", res_sharpen)

    print("Đang xử lý Tác vụ 3: JPEG DCT 8x8...")
    res_dct = task_jpeg_dct_8x8(img)
    cv2.imwrite("out_dct.png", res_dct)

    print("Hoàn thành! Đã xuất 3 file: out_gaussian.png, out_sharpen.png, out_dct.png")