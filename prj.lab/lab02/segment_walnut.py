import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile

class SegmentWalnut:
    # ─────────────────────────────────────────────────────────────
    # Вспомогательные методы
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_step_edge_threshold(img, bright_quantile=99.5, step_fraction=0.03):
        """Адаптивный порог: отступаем вниз от заданного перцентиля яркости."""
        q_val = np.percentile(img, bright_quantile)
        min_val = img.min()
        threshold = q_val - step_fraction * (q_val - min_val)
        return np.clip(threshold, min_val, q_val)

    @staticmethod
    def _get_largest_component(binary_mask):
        """Оставляет только самую крупную связную область (убирает шум/артефакты)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        if num_labels <= 1:
            return np.zeros_like(binary_mask, dtype=np.uint8)
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        return (labels == largest_idx).astype(np.uint8) * 255

    # ────────────────────────────────────────────────────────────
    # Основной метод сегментации
    # ─────────────────────────────────────────────────────────────
    def segment_walnut(self, image_path, bright_quantile=99.5, step_fraction=0.03):
        # 1. Загрузка 32-битного TIFF
        img_32 = tifffile.imread(image_path)
        
        # 2. Конвертация в оттенки серого, если нужно
        if len(img_32.shape) == 3:
            img_32 = cv2.cvtColor(img_32, cv2.COLOR_RGB2GRAY)
            
        # 3. Нормализация в 0-255 (uint8)
        img = cv2.normalize(img_32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if img is None:
            print("Error loading image")
            return

        # --- PREPROCESSING ---
        denoised = cv2.medianBlur(img, 7)
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        working_img = normalized.copy()

        # --- STEP 1: SHELL ---
        print("Processing Shell...")
        
        # 1. Адаптивный порог
        thresh_shell = self._compute_step_edge_threshold(
            working_img, bright_quantile, step_fraction
        )
        shell_bin = (working_img >= thresh_shell).astype(np.uint8) * 255
        
        # 2. Замыкаем разрывы кольца
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        shell_bin_closed = cv2.morphologyEx(shell_bin, cv2.MORPH_CLOSE, kernel_close)
        
        # 3. Оставляем только самую крупную компоненту (основа маски)
        shell_mask_core = self._get_largest_component(shell_bin_closed)
        
        # 4. ️ СОЗДАЕМ ДВЕ ВЕРСИИ МАСКИ
        # Версия А: ТОНКАЯ (для красоты, отчетов и визуализации)
        shell_mask_thin = shell_mask_core
        
        # Версия Б: ЖИРНАЯ (только для математического вычитания из working_img)
        # Расширяем на 3-4 пикселя, чтобы гарантированно стереть ВСЮ толщину скорлупы
        kernel_wipe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        shell_mask_wipe = cv2.dilate(shell_mask_core, kernel_wipe, iterations=1)
        
        # 5. ️ ВЫЧИТАЕМ ЖИРНУЮ МАСКУ
        # Это гарантирует, что от скорлупы в working_img не останется ни пикселя
        working_img[shell_mask_wipe > 0] = 0
        
        # 6. Используем тонкую маску для дальнейших расчетов и вывода
        shell_mask = shell_mask_thin

        # --- STEP 2: KERNEL & PARTITIONS (Strategy: Thin the Gap) ---
        print("Processing Kernel & Partitions...")
        
        # 1. Общая маска внутреннего содержимого (Ядро + Перегородки)
        _, inner_mask = cv2.threshold(working_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Разрываем мосты (OPEN)
        # Ядро (5,5) разрывает перегородки, но мы потом вернем пиксели обратно
        kernel_separate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_broken = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, kernel_separate)
        
        # 3. "Жирные" перегородки = Всё - Разбитое ядро
        # Эта маска содержит перегородки + "объеденные" края лопастей (гало)
        partitions_fat = cv2.subtract(inner_mask, kernel_broken)
        
        # 4. ️ ИСТОНЬШАЕМ ПЕРЕГОРОДКИ (Erode)
        # Мы сужаем "жирные" перегородки, возвращая пиксели "гало" обратно ядру.
        # Ядро (3,3) для эрозии делает зазор тоньше, но оставляет линию перегородки.
        kernel_erode_part = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        partitions_thin = cv2.erode(partitions_fat, kernel_erode_part, iterations=1)
        
        # 5. ️ ФИНАЛЬНОЕ ЯДРО = Всё - Тонкие перегородки
        # Ядро становится полным (лопасти целые), минус только тонкие линии перегородок
        kernel_final = cv2.subtract(inner_mask, partitions_thin)
        
        # 6. Фильтр шума для ядра (оставляем только крупные лопасти)
        # largest_component не подойдет, так как ядро разбито на части
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(kernel_final, connectivity=8)
        min_area = 200 # Минимальная площадь лопасти (настройка)
        kernel_mask = np.zeros_like(kernel_final)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                kernel_mask[labels == i] = 255
        
        # 7. Финализация перегородок
        # Замыкаем разрывы в самих линиях перегородок, чтобы они были цельными
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        partitions_final = cv2.morphologyEx(partitions_thin, cv2.MORPH_CLOSE, kernel_cross)
        
        # Обрезаем перегородки по контуру скорлупы (убираем внешний шум)
        contours, _ = cv2.findContours(shell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            filled_interior = np.zeros_like(shell_mask)
            cv2.drawContours(filled_interior, [c], -1, 255, thickness=cv2.FILLED)
            partitions_final = cv2.bitwise_and(partitions_final, filled_interior)

        # Обновляем working_img для совместимости
        working_img[kernel_mask > 0] = 0

        # --- STEP 3: PARTITIONS (Preserve Thin Lines) ---
        print("Processing Partitions...")
        
        # 1. НИЗКИЙ ПОРОГ (30-40), чтобы захватить даже тусклые тонкие линии
        _, partitions_mask = cv2.threshold(working_img, 35, 255, cv2.THRESH_BINARY)
        
        # 2. Обрезаем по контуру скорлупы (убираем внешний шум)
        contours, _ = cv2.findContours(shell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            filled_interior = np.zeros_like(shell_mask)
            cv2.drawContours(filled_interior, [c], -1, 255, thickness=cv2.FILLED)
            partitions_mask = cv2.bitwise_and(partitions_mask, filled_interior)

        # 3. ️ TOP-HAT TRANSFORM: Выделяет тонкие светлые линии на тёмном фоне
        # Это ИДЕАЛЬНО для перегородок — они тонкие и яркие относительно фона
        kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        partitions_tophat = cv2.morphologyEx(partitions_mask, cv2.MORPH_TOPHAT, kernel_line)
        
        # 4. Порог после top-hat (он усиливает контраст линий)
        _, partitions_enhanced = cv2.threshold(partitions_tophat, 10, 255, cv2.THRESH_BINARY)
        
        # 5. ️ СКЕЛЕТИЗАЦИЯ: Делаем линии толщиной в 1 пиксель
        # Это сохраняет все детали, даже самые тонкие
        def skeletonize(img):
            size = np.size(img)
            skel = np.zeros(img.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            done = False
            temp_img = img.copy()
            while not done:
                eroded = cv2.erode(temp_img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(temp_img, temp)
                skel = cv2.bitwise_or(skel, temp)
                temp_img = eroded.copy()
                zeros = size - cv2.countNonZero(temp_img)
                if zeros == size:
                    done = True
            return skel
        
        # Применяем скелетизацию если есть что скелетизировать
        if cv2.countNonZero(partitions_enhanced) > 0:
            partitions_final = skeletonize(partitions_enhanced)
            
            # 6. Слегка утолщаем (опционально, чтобы линии были виднее)
            # partitions_final = cv2.dilate(partitions_final, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations=1)
        else:
            partitions_final = partitions_enhanced

        # --- VISUALIZATION ---
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 6, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 6, 2)
        plt.imshow(normalized, cmap='gray')
        plt.title("Normalized & Denoised")
        plt.axis('off')

        plt.subplot(1, 6, 3)
        plt.imshow(shell_mask, cmap='gray')
        plt.title("1. Shell Mask")
        plt.axis('off')

        plt.subplot(1, 6, 4)
        plt.imshow(kernel_mask, cmap='gray')
        plt.title("2. Kernel Mask")
        plt.axis('off')

        plt.subplot(1, 6, 5)
        plt.imshow(partitions_mask, cmap='gray')
        plt.title("3. Partitions Mask")
        plt.axis('off')
        
        plt.subplot(1, 6, 6)
        combined = cv2.bitwise_or(shell_mask, cv2.bitwise_or(kernel_mask, partitions_mask))
        plt.imshow(combined, cmap='gray')
        plt.title("Combined Result")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return img, normalized, shell_mask, kernel_mask, partitions_mask