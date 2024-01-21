import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

img = cv2.imread('altay.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255  #original image
# точечный (соль/перец) шум
salt_paper = random_noise(img, mode='s&p', amount=0.4, salt_vs_pepper=0.6).astype(np.float32)
#Аддитивный шум
additive = random_noise(img, mode='gaussian', mean=0, var=0.08).astype(np.float32)
#Мультипликативный шум
specle = random_noise(img, mode='speckle', mean=0, var=0.08).astype(np.float32)
#Шум квантования
quant = random_noise(img, mode='poisson').astype(np.float32)
#Генерация шумов
def noise_generator(i):
    #Визуализация
    f, ax = plt.subplots(3,2, figsize=(10,15))
    ax[0,0].imshow(i, cmap='gray')
    ax[0,0].set_title('Оригинальное изображение')
    ax[0,1].imshow(i, cmap='gray')
    ax[0,1].set_title('Оригинальное изображение')
    ax[1,0].imshow(salt_paper, cmap='gray')
    ax[1,0].set_title('Импульсный шум (соль/перец)')
    ax[1,1].imshow(additive, cmap='gray')
    ax[1,1].set_title('Нормальный шум')
    ax[2,0].imshow(specle, cmap='gray')
    ax[2,0].set_title('Спекл-шум')
    ax[2,1].imshow(quant, cmap='gray')
    ax[2,1].set_title('Шум Пуассона')
    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')
    ax[2,0].axis('off')
    ax[2,1].axis('off')
    plt.subplots_adjust(wspace=0)
    plt.show()
#Фильтрация изображений
#Фильтр Гаусса
def lowfreq_filtering(i, sp, a, s, q):
    sigma = 2.5
    ksize = (0, 0)
    filter_gauss_1 = cv2.GaussianBlur(i, ksize, sigma)
    filter_gauss_2 = cv2.GaussianBlur(sp, ksize, sigma)
    filter_gauss_3 = cv2.GaussianBlur(a, ksize, sigma)
    filter_gauss_4 = cv2.GaussianBlur(s, ksize, sigma)
    filter_gauss_5 = cv2.GaussianBlur(q, ksize, sigma)

    f, ax = plt.subplots(5, 2, figsize=(10, 25))
    f.suptitle('Фильтр Гаусса', fontsize=19, ha='center', fontweight='bold')
    ax[0, 0].imshow(i, cmap='gray')
    ax[0, 0].set_title('Оригинальное изображение')
    ax[0, 1].imshow(filter_gauss_1, cmap='gray')
    ax[0, 1].set_title('Отфильтрованное изображение')

    ax[1, 0].imshow(sp, cmap='gray')
    ax[1, 0].set_title('Шум "Соль" и "Перец"')
    ax[1, 1].imshow(filter_gauss_2, cmap='gray')
    ax[1, 1].set_title('Отфильтрованное изображение')

    ax[2, 0].imshow(a, cmap='gray')
    ax[2, 0].set_title('Нормальный шум')
    ax[2, 1].imshow(filter_gauss_3, cmap='gray')
    ax[2, 1].set_title('Отфильтрованное изображение')

    ax[3, 0].imshow(s, cmap='gray')
    ax[3, 0].set_title('Спекл-шум')
    ax[3, 1].imshow(filter_gauss_4, cmap='gray')
    ax[3, 1].set_title('Отфильтрованное изображение')

    ax[4, 0].imshow(q, cmap='gray')
    ax[4, 0].set_title('Шум Пуассона')
    ax[4, 1].imshow(filter_gauss_5, cmap='gray')
    ax[4, 1].set_title('Отфильтрованное изображение')

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    ax[4, 0].axis('off')
    ax[4, 1].axis('off')
    plt.subplots_adjust(top=0.93, wspace=0)
    plt.show()

#Контргармонический фильтр
def Contrgarmonic_filter(i, Q, ksize):
    eps = 1e-11
    I_Q = (i + eps) ** Q
    kernel = np.ones(ksize, dtype=np.float32)
    numerator = cv2.filter2D(I_Q * i, -1, kernel)
    demoninator = cv2.filter2D(I_Q, -1, kernel) + eps
    return np.clip((numerator / demoninator).astype(np.float32), 0, 1)


def salt_or_paper(i):
    salt = random_noise(i, mode='salt')
    pepper = random_noise(i, mode='pepper')
    pepper_filter = Contrgarmonic_filter(pepper, Q=2, ksize=(3, 3))
    salt_filter = Contrgarmonic_filter(salt, Q=-2, ksize=(3, 3))

    f, ax = plt.subplots(2, 2, figsize=(12, 12))
    f.suptitle('Коррекция шума "соль" и "перец" с использованием контргармонического филтра', ha='center',
               fontsize=19, fontweight='bold')
    ax[0, 0].imshow(salt, cmap='gray')
    ax[0, 0].set_title('Шум "Соль"')

    ax[0, 1].imshow(salt_filter, cmap='gray')
    ax[0, 1].set_title('Отфильтрованное изображение')

    ax[1, 0].imshow(pepper, cmap='gray')
    ax[1, 0].set_title('Шум "Перец""')

    ax[1, 1].imshow(pepper_filter, cmap='gray')
    ax[1, 1].set_title('Отфильтрованное изображение')

    ax[0, 0].axis('off')
    ax[1, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

#Нелинейная фильтрация
def unlinear_filtering(i, sp, a, s, q, ksize):
    filter_m_1 = cv2.medianBlur(i, ksize)
    filter_m_2 = cv2.medianBlur(sp, ksize)
    filter_m_3 = cv2.medianBlur(a, ksize)
    filter_m_4 = cv2.medianBlur(s, ksize)
    filter_m_5 = cv2.medianBlur(q, ksize)

    f, ax = plt.subplots(5, 2, figsize=(10, 25))
    f.suptitle('Медианный фильтр', fontsize=19, ha='center', fontweight='bold')
    ax[0, 0].imshow(i, cmap='gray')
    ax[0, 0].set_title('Оригинальное изображение')
    ax[0, 1].imshow(filter_m_1, cmap='gray')
    ax[0, 1].set_title('Отфильтрованное изображение')

    ax[1, 0].imshow(sp, cmap='gray')
    ax[1, 0].set_title('Шум "Соль" и "Перец"')
    ax[1, 1].imshow(filter_m_2, cmap='gray')
    ax[1, 1].set_title('Отфильтрованное изображение')

    ax[2, 0].imshow(a, cmap='gray')
    ax[2, 0].set_title('Нормальный шум')
    ax[2, 1].imshow(filter_m_3, cmap='gray')
    ax[2, 1].set_title('Отфильтрованное изображение')

    ax[3, 0].imshow(s, cmap='gray')
    ax[3, 0].set_title('Спекл-шум')
    ax[3, 1].imshow(filter_m_4, cmap='gray')
    ax[3, 1].set_title('Отфильтрованное изображение')

    ax[4, 0].imshow(q, cmap='gray')
    ax[4, 0].set_title('Шум Руассона')
    ax[4, 1].imshow(filter_m_5, cmap='gray')
    ax[4, 1].set_title('Отфильтрованное изображение')

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    ax[4, 0].axis('off')
    ax[4, 1].axis('off')
    plt.subplots_adjust(top=0.93, wspace=0)
    plt.show()

#Взвешенный медианный фильтр, ранговый фильтр
def weight_rank_filter(i, k_size, kernel, rank):
    out = np.copy(i)
    copy = cv2.copyMakeBorder(i, int((k_size[0] - 1) / 2), int(k_size[0] / 2), int((k_size[1] - 1) / 2), int(k_size[1] / 2), cv2.BORDER_REPLICATE)
    rows, cols = copy.shape[0:2]
    kernel_fl = kernel.flatten()
    index_center_kernel = (k_size[0] // 2, k_size[1] // 2)
    for i in range(index_center_kernel[0], rows - index_center_kernel[0]):
        for j in range(index_center_kernel[1], cols - index_center_kernel[1]):
            window = copy[i-index_center_kernel[0]:i+index_center_kernel[0]+1, j-index_center_kernel[1]:j+index_center_kernel[1]+1].flatten()
            pixels = np.sort(np.repeat(window, kernel_fl))
            out[i-index_center_kernel[0], j-index_center_kernel[1]] = pixels[rank]
    return out


def weight_median_filter(i, sp, a, s, q):
    k_size = (5, 7)  # размер маски для медианы
    kernel = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 2, 1, 1], [1, 2, 3, 4, 3, 2, 1], [1, 1, 2, 2, 2, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)  # матрица весов для медианного фильтра 5x7
    rank = 20  # ранг рангового фильтра

    filter_m_1 = weight_rank_filter(i, k_size, kernel, rank)
    filter_m_2 = weight_rank_filter(sp, k_size, kernel, rank)
    filter_m_3 = weight_rank_filter(a, k_size, kernel, rank)
    filter_m_4 = weight_rank_filter(s, k_size, kernel, rank)
    filter_m_5 = weight_rank_filter(q, k_size, kernel, rank)

    f, ax = plt.subplots(5, 2, figsize=(10, 15))
    f.suptitle('Взвешенная медианная фильтрация', fontsize=19, ha='center', fontweight='bold')
    ax[0, 0].imshow(i, cmap='gray')
    ax[0, 0].set_title('Оригинальное изображение')
    ax[0, 1].imshow(filter_m_1, cmap='gray')
    ax[0, 1].set_title('Отфильтрованное изображение')

    ax[1, 0].imshow(sp, cmap='gray')
    ax[1, 0].set_title('Шум "Соль" и "Перец"')
    ax[1, 1].imshow(filter_m_2, cmap='gray')
    ax[1, 1].set_title('Отфильтрованное изображение')

    ax[2, 0].imshow(a, cmap='gray')
    ax[2, 0].set_title('Нормальный шум')
    ax[2, 1].imshow(filter_m_3, cmap='gray')
    ax[2, 1].set_title('Отфильтрованное изображение')

    ax[3, 0].imshow(s, cmap='gray')
    ax[3, 0].set_title('Спекл-шум')
    ax[3, 1].imshow(filter_m_4, cmap='gray')
    ax[3, 1].set_title('Отфильтрованное изображение')

    ax[4, 0].imshow(q, cmap='gray')
    ax[4, 0].set_title('Шум Пуассона')
    ax[4, 1].imshow(filter_m_5, cmap='gray')
    ax[4, 1].set_title('Отфильтрованное изображение')

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    ax[4, 0].axis('off')
    ax[4, 1].axis('off')
    plt.subplots_adjust(top=0.93, wspace=0)
    plt.show()


def adaptive_median_filter(I, s_max):
    I_out = np.copy(I)
    I_copy = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_REPLICATE) #border padding = 1
    rows, cols = I_copy.shape[0:2]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):   #проход по изобр. с паддингом, поэтому обход не по всему изображению
            # шаг 1
            flag = False  #индикатор того, дошел ли s до s_max
            for s in range(3, s_max+1, 2): #цикл по размеру фильтра
                index_center_kernel = (s // 2, s // 2) #кордината центра фильтра
                # выходим из цикла подбора размера фильтра,
                # если фильтр не может увеличиваться из-за достижения края картинки
                if (i - index_center_kernel[0]) < 0 or (j-index_center_kernel[1]) < 0:
                    break
                window = I_copy[i-index_center_kernel[0]:i+index_center_kernel[0]+1,
                                j-index_center_kernel[1]:j+index_center_kernel[1]+1]
                z_min = window.min()
                z_max = window.max()
                z_med = np.median(window)
                A_1 = z_med - z_min
                A_2 = z_med - z_max
                #выходим из цикла подбора размера фильтра, если достигнуто необх. условие
                if A_1 > 0 and A_2 < 0:
                    break
                # если все плохо и мы не вышли из цикла, s=s_max - не меняем текущий пиксель
                if s == s_max:
                    I_out[i-index_center_kernel[0], j-index_center_kernel[1]] = I_copy[i, j]
                    flag = True
            # шаг 2
            if not flag:       #если flag=False, то мы еще не дали пикселю значение
                B_1 = I_copy[i, j] - z_min
                B_2 = I_copy[i, j] - z_max
                if B_1 > 0 and B_2 < 0:
                    I_out[i-index_center_kernel[0], j-index_center_kernel[1]] = I_copy[i, j]
                else:
                    I_out[i-index_center_kernel[0], j-index_center_kernel[1]] = z_med
    return I_out


def am_filter(i, sp, a, s, q):
    s_max = 31
    filter_am_1 = adaptive_median_filter(i, s_max)
    filter_am_2 = adaptive_median_filter(sp, s_max)
    filter_am_3 = adaptive_median_filter(a, s_max)
    filter_am_4 = adaptive_median_filter(s, s_max)
    filter_am_5 = adaptive_median_filter(q, s_max)

    f, ax = plt.subplots(5, 2, figsize=(10, 15))
    f.suptitle('Адаптивная медианная фильтрация', fontsize=19, ha='center', fontweight='bold')
    ax[0, 0].imshow(i, cmap='gray')
    ax[0, 0].set_title('Оригинальное изображение')
    ax[0, 1].imshow(filter_am_1, cmap='gray')
    ax[0, 1].set_title('Отфильтрованное изображение')

    ax[1, 0].imshow(sp, cmap='gray')
    ax[1, 0].set_title('Шум "Соль" и "Перец"')
    ax[1, 1].imshow(filter_am_2, cmap='gray')
    ax[1, 1].set_title('Отфильтрованное изображение')

    ax[2, 0].imshow(a, cmap='gray')
    ax[2, 0].set_title('Нормальный шум')
    ax[2, 1].imshow(filter_am_3, cmap='gray')
    ax[2, 1].set_title('Отфильтрованное изображение')

    ax[3, 0].imshow(s, cmap='gray')
    ax[3, 0].set_title('Спекл-шум')
    ax[3, 1].imshow(filter_am_4, cmap='gray')
    ax[3, 1].set_title('Отфильтрованное изображение')

    ax[4, 0].imshow(q, cmap='gray')
    ax[4, 0].set_title('Шум Пуассона')
    ax[4, 1].imshow(filter_am_5, cmap='gray')
    ax[4, 1].set_title('Отфильтрованное изображение')

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    ax[4, 0].axis('off')
    ax[4, 1].axis('off')
    plt.subplots_adjust(top=0.93, wspace=0)
    plt.show()


def wiener_filter(I, k_size, var_noise):
    eps = 1e-11
    I_out = np.copy(I)
    I_copy = cv2.copyMakeBorder(I, int((k_size[0] - 1) / 2), int(k_size[0] / 2),
                                int((k_size[1] - 1) / 2), int(k_size[1] / 2), cv2.BORDER_REPLICATE)
    I_copy_power = I_copy ** 2

    rows, cols = I_copy.shape[0:2]
    index_center_kernel = (k_size[0] // 2, k_size[1] // 2)
    for i in range(index_center_kernel[0], rows - index_center_kernel[0]):
        for j in range(index_center_kernel[1], cols - index_center_kernel[1]):
            window = I_copy[i - index_center_kernel[0]:i + index_center_kernel[0] + 1,
                     j - index_center_kernel[1]:j + index_center_kernel[1] + 1]
            window_power = I_copy_power[i - index_center_kernel[0]:i + index_center_kernel[0] + 1,
                           j - index_center_kernel[1]:j + index_center_kernel[1] + 1]
            m = np.sum(window) / (k_size[0] * k_size[1])
            var = np.sum(window_power - m ** 2) / (k_size[0] * k_size[1])
            I_out[i - index_center_kernel[0], j - index_center_kernel[1]] = m + ((var - var_noise) / (var + eps)) * (
                        I_copy[i, j] - m)
    return np.clip(I_out, 0, 1).astype(np.float32)


def wiener(sp, a, s, q):
    k_size = (5, 5)
    var_noise = 0.08  # такая же дисперсия модели шума, как и при генерации шумов аддитивный и мультипликативный по Гауссу

    filter_w_2 = wiener_filter(sp, k_size, var_noise)
    filter_w_3 = wiener_filter(a, k_size, var_noise)
    filter_w_4 = wiener_filter(s, k_size, var_noise)
    filter_w_5 = wiener_filter(q, k_size, var_noise)

    f, ax = plt.subplots(4, 2, figsize=(12, 12))
    f.suptitle(f'Винеровская фильтрация', fontsize=19, ha='center', fontweight='bold')

    ax[0, 0].imshow(sp, cmap='gray')
    ax[0, 0].set_title('Шум "Соль" и "Перец"')
    ax[0, 1].imshow(filter_w_2, cmap='gray')
    ax[0, 1].set_title('Отфильтрованное изображение')

    ax[1, 0].imshow(a, cmap='gray')
    ax[1, 0].set_title('Нормальный шум')
    ax[1, 1].imshow(filter_w_3, cmap='gray')
    ax[1, 1].set_title('Отфильтрованное изображение')

    ax[2, 0].imshow(s, cmap='gray')
    ax[2, 0].set_title('Спекл-шум')
    ax[2, 1].imshow(filter_w_4, cmap='gray')
    ax[2, 1].set_title('Отфильтрованное изображение')

    ax[3, 0].imshow(q, cmap='gray')
    ax[3, 0].set_title('Шум Пуассона')
    ax[3, 1].imshow(filter_w_5, cmap='gray')
    ax[3, 1].set_title('Отфильтрованное изображение')

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    plt.subplots_adjust(top=0.93, wspace=-0.5)
    plt.show()

#Высокочастотная фильтрация
def highfreq_filtering(I):
    Gx = np.array([[1, 0], [0, -1]])
    Gy = np.array([[0, 1], [-1, 0]])
    I_x = cv2.filter2D(I, -1, Gx)
    I_y = cv2.filter2D(I, -1, Gy)
    I_roberts = cv2.magnitude(I_x, I_y)

    sobelx = cv2.Sobel(I, -1, 1, 0)
    sobely = cv2.Sobel(I, -1, 0, 1)
    I_sobel = cv2.magnitude(sobelx, sobely)

    I_laplacian = np.absolute(cv2.Laplacian(I, -1))

    Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    I_x = cv2.filter2D(I, -1, Gx)
    I_y = cv2.filter2D(I, -1, Gy)
    I_previtt = cv2.magnitude(I_x, I_y)

    I_int = np.round(255 * I, 0).astype(np.uint8)
    I_canny = cv2.Canny(I_int, 90, 200)

    f, ax = plt.subplots(5, 1, figsize=(5, 20))
    f.suptitle('Высокочастотная фильтрация', fontsize=14, ha='center', fontweight='bold')

    ax[0].imshow(I_roberts, cmap='gray')
    ax[0].set_title('фильтр Робертса')

    ax[1].imshow(I_sobel, cmap='gray')
    ax[1].set_title('фильтр Собеля')

    ax[2].imshow(I_laplacian, cmap='gray')
    ax[2].set_title('фильтр Лапласа')

    ax[3].imshow(I_previtt, cmap='gray')
    ax[3].set_title('фильтр Превитта')

    ax[4].imshow(I_canny, cmap='gray')
    ax[4].set_title('фильтр Кэнни')

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    plt.subplots_adjust(top=0.95, wspace=-0.5)
    plt.show()

#noise_generator(img)
#lowfreq_filtering(img, salt_paper, additive, specle, quant)
#salt_or_paper(img)
#unlinear_filtering(img, salt_paper, additive, specle, quant, ksize = 3)
#weight_median_filter(img, salt_paper, additive, specle, quant)
#am_filter(img, salt_paper, additive, specle, quant)
#wiener(salt_paper, additive, specle, quant)
highfreq_filtering(img)