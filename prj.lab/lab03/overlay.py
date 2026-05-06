import cv2
import numpy as np
import matplotlib.pyplot as plt

class Overlay:
    def __init__(self, original_img, normalized_img, shell_mask, kernel_mask, partitions_mask):
        """
        :param original_img: Исходное изображение (img)
        :param normalized_img: Предобработанное изображение (normalized)
        :param shell_mask: Маска скорлупы
        :param kernel_mask: Маска ядра
        :param partitions_mask: Маска перегородок
        """
        self.original = original_img
        self.normalized = normalized_img
        self.shell = shell_mask
        self.kernel = kernel_mask
        self.partitions = partitions_mask
        
        # Цвета для наложения (RGB)
        self.COLORS = {
            'shell': (255, 0, 0),      # Красный
            'kernel': (0, 255, 0),     # Зеленый
            'partitions': (0, 0, 255)  # Синий
        }

    def _prepare_rgb_image(self):
        """Приводит изображение к RGB для корректного отображения в matplotlib."""
        if len(self.original.shape) == 2:
            return cv2.cvtColor(self.original, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)

    def generate(self, alpha=0.4):
        """Генерирует только итоговую картинку с наложением."""
        img_rgb = self._prepare_rgb_image()
        overlay = np.zeros_like(img_rgb, dtype=np.float32)
        
        overlay[self.shell > 0] = self.COLORS['shell']
        overlay[self.kernel > 0] = self.COLORS['kernel']
        overlay[self.partitions > 0] = self.COLORS['partitions']

        blended = img_rgb.astype(np.float32) * (1 - alpha) + overlay * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def plot_dashboard(self, alpha=0.45):
        """
        Отрисовывает полную панель из 7 графиков, как в вашем примере.
        """
        # Генерируем оверлей для 7-го графика
        overlay_img = self.generate(alpha=alpha)
        
        # Комбинированная маска для 6-го графика
        combined = cv2.bitwise_or(self.shell, cv2.bitwise_or(self.kernel, self.partitions))

        plt.figure(figsize=(22, 4)) # Широкий формат для 7 колонок

        # 1. Original
        plt.subplot(1, 7, 1)
        plt.imshow(self.original, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # 2. Normalized & Denoised
        plt.subplot(1, 7, 2)
        plt.imshow(self.normalized, cmap='gray')
        plt.title("Normalized & Denoised")
        plt.axis('off')

        # 3. Shell Mask
        plt.subplot(1, 7, 3)
        plt.imshow(self.shell, cmap='gray')
        plt.title("1. Shell Mask")
        plt.axis('off')

        # 4. Kernel Mask
        plt.subplot(1, 7, 4)
        plt.imshow(self.kernel, cmap='gray')
        plt.title("2. Kernel Mask")
        plt.axis('off')

        # 5. Partitions Mask
        plt.subplot(1, 7, 5)
        plt.imshow(self.partitions, cmap='gray')
        plt.title("3. Partitions Mask")
        plt.axis('off')

        # 6. Combined Binary
        plt.subplot(1, 7, 6)
        plt.imshow(combined, cmap='gray')
        plt.title("Combined Binary")
        plt.axis('off')

        # 7. Segmentation Overlay
        plt.subplot(1, 7, 7)
        plt.imshow(overlay_img)
        plt.title("Segmentation Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
