from PIL import Image
import numpy as np
import cv2

laplacian_kernel = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

sobelX_kernel = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

sobelY_kernel = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

sharpen_kernel = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

emboss_kernel = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")

blur_kernel = np.array((
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625]), dtype="int")

outline_kernel = np.array((
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]), dtype="int")





def otsu_thresholding(img):
    # Otsu's thresholding
    t, out = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return out


def global_thresholding(img):
    # global thresholding
    t, out = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return out


def gaussian_otsu_thresholding(img):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    t, out = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return out


def bw_image(img):
    gray = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #gray = cv2.medianBlur(gray, 3)
    return gray


def wide_edge_detection(img):
    return cv2.Canny(img, 10, 200)


def tight_edge_detection(img):
    return cv2.Canny(img, 225, 250)


# with auto lower and upper threshold detection
def auto_edge_detection(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    return edged

# percentage_scale
# keeps aspect ratio
def percentage_scale(img,percentage):
    x = img.shape[1]
    y = img.shape[0]
    new_x = x + ((x/100)*percentage)
    ratio = new_x / x
    new_y = ratio * y
    dim = (new_x, new_y)
    resized_img = cv2.resize(img,dim,interpolation = cv2.INTER_LANCZOS4)
    return resized_img


def convolution_2d(img):
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img,-1,kernel)


def laplacian(img):
    return cv2.Laplacian(img,cv2.CV_64F)


def sobelX(img):
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    return np.uint8(abs_sobel64f)


def sobelY(img):
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    return np.uint8(abs_sobel64f)


def blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def showImg(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def toGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
