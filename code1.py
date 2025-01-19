import cv2
import numpy as np
import os

# --------------------------- توابع کمکی ---------------------------
def divide_into_blocks(image, block_size=8):
    """
    تقسیم تصویر به بلوک‌هایی با اندازه مشخص.
    """
    blocks = []
    height, width = image.shape[:2]
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                blocks.append((i, j, block))
    return blocks

def detect_edges(image, threshold1=100, threshold2=200):
    """
    تشخیص لبه‌ها با استفاده از الگوریتم Canny.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    return edges

def classify_blocks(blocks, edges, block_size=8):
    """
    تقسیم بلوک‌ها به ساختاری و بافتی.
    """
    structural_blocks = []
    texture_blocks = []
    for (i, j, block) in blocks:
        block_edges = edges[i:i+block_size, j:j+block_size]
        if np.any(block_edges):
            structural_blocks.append((i, j, block))
        else:
            texture_blocks.append((i, j, block))
    return structural_blocks, texture_blocks

def compute_gradients(block):
    """
    محاسبه گرادیان بلوک.
    """
    gray_block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
    grad_x = np.gradient(gray_block, axis=1)
    grad_y = np.gradient(gray_block, axis=0)
    return grad_x, grad_y

def block_variance(block):
    """
    محاسبه واریانس پیکسل‌های بلوک.
    """
    gray_block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
    return np.var(gray_block)

def identify_removable_blocks(texture_blocks, threshold_variance=500, threshold_gradient=5):
    """
    شناسایی بلوک‌های قابل حذف.
    """
    removable_blocks = []
    for (i, j, block) in texture_blocks:
        grad_x, grad_y = compute_gradients(block)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        variance = block_variance(block)
        if avg_gradient < threshold_gradient and variance < threshold_variance:
            removable_blocks.append((i, j))
    return removable_blocks

def create_mask_and_modified_image(image_rgb, removable_blocks, block_size=8):
    """
    ایجاد ماسک و تصویر اصلاح‌شده.
    """
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    modified_image = image_rgb.copy()
    for (i, j) in removable_blocks:
        mask[i:i+block_size, j:j+block_size] = 255
        modified_image[i:i+block_size, j:j+block_size] = [255, 255, 255]  # سفید کردن
    return mask, modified_image

# --------------------------- کد اصلی ---------------------------
def main():
    # مسیر تصویر اصلی را مشخص کنید
    image_path = '/content/11.jpg'  # تغییر مسیر به مسیر تصویر خود
    if not os.path.exists(image_path):
        print(f"فایل تصویری با مسیر {image_path} یافت نشد!")
        return

    # محاسبه حجم فایل ورودی
    input_size = os.path.getsize(image_path)
    print(f"حجم تصویر ورودی: {input_size / 1024:.2f} KB")

    # خواندن تصویر و تبدیل به RGB
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("بارگذاری تصویر با مشکل مواجه شد!")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1) تقسیم تصویر به بلوک‌های 8x8
    blocks = divide_into_blocks(image_rgb, block_size=8)

    # 2) تشخیص لبه‌ها
    edges = detect_edges(image_rgb)

    # 3) طبقه‌بندی بلوک‌ها به ساختاری و بافتی
    structural_blocks, texture_blocks = classify_blocks(blocks, edges, block_size=8)

    # 4) شناسایی بلوک‌های قابل حذف
    removable_blocks = identify_removable_blocks(texture_blocks)

    # 5) ساخت mask و تصویر اصلاح‌شده (حذف بلوک‌های غیرضروری)
    mask, modified_image = create_mask_and_modified_image(image_rgb, removable_blocks, block_size=8)

    # 6) ذخیرهٔ mask و modified_image در فایل
    cv2.imwrite('mask.png', mask)  # ماسک تک‌کاناله است
    # ذخیره تصویر اصلاح‌شده به صورت JPEG با کیفیت 90
    cv2.imwrite('modified_image.jpg', cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # محاسبه حجم فایل‌های خروجی
    modified_image_size = os.path.getsize('modified_image.jpg')
    mask_size = os.path.getsize('mask.png')

    print(f"حجم فایل mask.png: {mask_size / 1024:.2f} KB")
    print(f"حجم فایل modified_image.jpg: {modified_image_size / 1024:.2f} KB")

    if modified_image_size < input_size:
        print("حجم تصویر خروجی (modified_image.jpg) کمتر از ورودی است.")
    else:
        print("حجم تصویر خروجی (modified_image.jpg) بیشتر از ورودی است. لطفاً پارامترهای فشرده‌سازی را تنظیم کنید.")

    print("فایل‌های mask.png و modified_image.jpg با موفقیت ذخیره شدند.")
    print("پایان کد اول.")

if __name__ == "__main__":
    main()
