import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

histSize = 256
histRange = (0, 256)
my_image = cv2.imread('horse.jpg', cv2.IMREAD_COLOR)
my_image_gray = cv2.imread('horse.jpg', cv2.IMREAD_GRAYSCALE)
proection = cv2.imread('lenin.jpg')
bar_code = cv2.imread('bar_code.jpg')


def histograms(i, hS, hR):
    i_bgr = cv2.split(i)
    colors = ('b', 'g', 'r')
    bgr = []
    for k, color in enumerate(colors):
        hist = cv2.calcHist(i_bgr, [k], None, [hS], hR)
        bgr.append(hist)
        plt.plot(hist, color = color)
    numRows, numCols = i.shape[0], i.shape[1]
    CHb = np.cumsum(bgr[0]) / (numRows * numCols)
    CHb = CHb[:, np.newaxis].astype(np.float32)
    CHg = np.cumsum(bgr[1]) / (numRows * numCols)
    CHg = CHg[:, np.newaxis].astype(np.float32)
    CHr = np.cumsum(bgr[2]) / (numRows * numCols)
    CHr = CHr[:, np.newaxis].astype(np.float32)
    CH = np.concatenate([CHb, CHg, CHr], axis=1)
    # Вывод гистограмм
    plt.title("Гистограммы по каналам")
    plt.show()

    return CH


def dynamic_range_stretching(i, hS, hR):
    if i.dtype == np.float32:
        i = np.round(i * 255, 0).astype(np.uint8)
    alpha = [_ * 0.125 for _ in range(1, 17)]
    for k in alpha:
        #Преобразование к другому типу
        if i.dtype == np.uint8:
            i_new = i.astype(np.float32) / 255
        else:
            i_new = i
        #Расслоение
        i_bgr = cv2.split(i_new)
        inew_bgr = []
        for layer in i_bgr:
            i_min = np.min(layer)
            i_max = np.max(layer)
            i_new = np.clip((((layer - i_min) / (i_max - i_min)) ** k), 0, 1)
            inew_bgr.append(i_new)
        #Объединение взад-назад
        i_new = cv2.merge(inew_bgr)
        # Обратное преобразование к uint если нужно
        if i.dtype == np.uint8:
            i_new = (255 * i_new).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f'{str(k/0.125)} image where alpha=' + str(k) + '.jpg', i_new)

        bgr = cv2.split(i_new)
        colors = ('b', 'g', 'r')
        #Расчет гистограммы для каждого слоя
        for j, color in enumerate(colors):
            hist = cv2.calcHist(bgr, [j], None, [hS], hR)
            plt.plot(hist, color = color)
        #Вывод гистограмм
        plt.title(f'Изображение гистограмм при альфа = {k}')
        plt.show()


def uniform_transform(i, CH):
    if i.dtype == np.uint8:
        i = i.astype(np.float32) / 255
        i_bgr = cv2.split(i)

    i_new_bgr = []
    i_new_bgr_cv = []

    for j, layer in enumerate(i_bgr):
        i_min = layer.min()
        i_max = layer.max()
        i_new = np.clip((i_max - i_min) * CH[(np.round(layer * 255, 0).astype(np.uint8)), j] + i_min, 0, 1)
        i_new_bgr.append(i_new)
        i_new_cv = cv2.equalizeHist((np.round(layer * 255, 0).astype(np.uint8)))
        i_new_bgr_cv.append(i_new_cv)

    i_new = cv2.merge(i_new_bgr)
    inew_cv = cv2.merge(i_new_bgr_cv)

    f, ax = plt.subplots(1, 3, figsize=(20, 8))
    ax[0].imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    ax[0].set_title('оригинал')
    ax[1].imshow(cv2.cvtColor(i_new, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f'равномерное преобразование \n (handmade)')
    ax[2].imshow(cv2.cvtColor(inew_cv, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'равномерное преобразование \n using cv2.equalizeHist()')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.show()


def exp_transform(i, CH):
    if i.dtype == np.uint8:
        i = i.astype(np.float32) / 255
        i_bgr = cv2.split(i)

    alpha = [_ * 0.5 for _ in range(1, 7)]
    image_ts = np.empty_like(i, dtype=np.float32)
    eps = 1e-7

    for k in alpha:
        for j, layer in enumerate(i_bgr):
            i_min = layer.min()
            i_max = layer.max()
            image_ts[:, :, j] = np.clip(i_min - (1 / k) * np.log(1 - CH[(np.round(layer * 255, 0).astype(np.uint8)), j] + eps), 0, 1)

        f, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original')
        ax[1].imshow(cv2.cvtColor(image_ts, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f'Transformed, alpha = {k}')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()


def relay_transform(i, CH):
    if i.dtype == np.uint8:
        i = i.astype(np.float32) / 255
        i_bgr = cv2.split(i)

    alpha = [_ for _ in [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0]]
    image_ts = np.empty_like(i, dtype=np.float32)
    eps = 1e-7

    for k in alpha:
        for j, layer in enumerate(i_bgr):
            i_min = layer.min()
            i_max = layer.max()
            image_ts[:, :, j] = np.clip(i_min + (2 * k**2 * np.log(1 / (1 - CH[(np.round(layer*255, 0).astype(np.uint8)),j]+eps)))**0.5,0,1)

        f, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original')
        ax[1].imshow(cv2.cvtColor(image_ts, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f'Transformed, alpha = {k}')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()


def two_third(i, CH):
    if i.dtype == np.uint8:
        i = i.astype(np.float32) / 255
        i_bgr = cv2.split(i)

    image_ts = np.empty_like(i, dtype=np.float32)

    for j, layer in enumerate(i_bgr):
        image_ts[:, :, j] = np.clip(np.power(CH[(np.round(layer * 255, 0).astype(np.uint8)), j], 2. / 3.), 0, 1)

    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original')
    ax[1].imshow(cv2.cvtColor(image_ts, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Transformed')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()


def hyperbolic_trnsform(i, hR, hS):
    Hist_gray = cv2.calcHist([i], [0], None, [hS], hR)
    CH_gray = np.cumsum(Hist_gray) / (i.shape[0] * i.shape[1])
    if i.dtype == np.uint8:
        i = i.astype(np.float32) / 255

    image_ts = np.empty_like(i, dtype=np.float32)

    #alpha = 0.05
    coef = [0.00, 0.01, 0.02, 0.05, 0.09, 0.1, 0.2, 0.3, 0.4]
    for alpha in coef:
        image_ts = np.clip(np.power(alpha, CH_gray[(np.round(i * 255, 0).astype(np.uint8))]), 0, 1)

        f, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(i, cmap='gray')
        ax[0].set_title('Original')
        ax[1].imshow(image_ts, cmap='gray')
        ax[1].set_title('Transformed')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()


def clahe(i):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(i)

    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(i, cmap='gray')
    ax[1].imshow(clahe_image, cmap='gray')
    ax[0].set_title('original gray-scaled image')
    ax[1].set_title('transformed clahe')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()


def proection_of_image(i):
    if i.ndim == 2:
        ProjI_y = np.sum(i, 1) / 255 / i.shape[1]
        ProjI_x = np.sum(i, 0) / 255 / i.shape[0]
    else:
        ProjI_y = np.sum(i, (1, 2)) / 255 / i.shape[1]
        ProjI_x = np.sum(i, (0, 2)) / 255 / i.shape[0]

    f, ax = plt.subplots(2, 2, figsize=(5, 5))
    ax[0, 0].imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title('original image')
    ax[0, 1].imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title('original image')
    ax[1, 0].plot(ProjI_y)
    ax[1, 0].set_title('Oy projection')
    ax[1, 0].set_xlim(0, i.shape[0])
    ax[1, 1].plot(ProjI_x)
    ax[1, 1].set_title('Ox projection')
    ax[1, 1].set_xlim(0, i.shape[1])
    plt.tight_layout()
    plt.show()


def profile(i):
    i = i.astype(np.float32) / 255
    grid = i[round(i.shape[0] / 2), :]

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(hspace=-0.2)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.imshow(i, cmap='gray')
    ax1.set_title('Изображение')
    ax1.axis('off')
    ax2.plot(grid)
    ax2.set_title('Профиль')
    ax2.set_xlim(0, i.shape[1])
    plt.show()


CH = histograms(my_image, histSize, histRange)                  #Гистограммы изображения
#dynamic_range_stretching(my_image, histSize, histRange)        #Растяжение динамического диапазона
#uniform_transform(my_image, CH)                                #Равномерное преобразование
#exp_transform(my_image, CH)                                    #Экспоненциальное преобразование
relay_transform(my_image, CH)                                  #Пробразование Рэлея
#two_third(my_image, CH)                                        #Преобразование две трети
#hyperbolic_trnsform(my_image_gray, histRange, histSize)        #Гиперболическое преобразование
#clahe(my_image_gray)
#proection_of_image(proection)                                  #Проекция изображения
#profile(bar_code)                                               #Профиль изображения