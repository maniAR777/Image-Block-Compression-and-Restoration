import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def restore_image(modified_image_bgr, mask, inpaint_radius=3, method=cv2.INPAINT_TELEA):
    """
    بازسازی تصویر با استفاده از inpaint.
    """
    restored_bgr = cv2.inpaint(modified_image_bgr, mask, inpaintRadius=inpaint_radius, flags=method)
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    return restored_rgb

def main():
    # بررسی موجود بودن فایل‌های ورودی
    if not os.path.exists('mask.png'):
        print("فایل mask.png یافت نشد!")
        return

    if not os.path.exists('modified_image.jpg'):
        print("فایل modified_image.jpg یافت نشد!")
        return

    # محاسبه حجم فایل ورودی
    modified_image_size = os.path.getsize('modified_image.jpg')
    mask_size = os.path.getsize('mask.png')
    print(f"حجم فایل mask.png: {mask_size / 1024:.2f} KB")
    print(f"حجم فایل modified_image.jpg: {modified_image_size / 1024:.2f} KB")

    # بارگذاری ماسک (تک‌کاناله)
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    # بارگذاری modified_image (JPEG)
    modified_image_bgr = cv2.imread('modified_image.jpg')

    if mask is None or modified_image_bgr is None:
        print("مشکل در بارگذاری فایل‌های mask یا modified_image!")
        return

    # بازسازی
    restored_image_rgb = restore_image(modified_image_bgr, mask)

    # نمایش نتایج (اختیاری)
    plt.figure(figsize=(14,5))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(modified_image_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Modified Image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(restored_image_rgb)
    plt.title("Restored Image (Inpainting)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # ذخیرهٔ نتیجه به صورت JPEG با کیفیت 90
    cv2.imwrite('restored_image.jpg', cv2.cvtColor(restored_image_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # محاسبه حجم فایل بازسازی شده
    restored_image_size = os.path.getsize('restored_image.jpg')
    print(f"حجم فایل restored_image.jpg: {restored_image_size / 1024:.2f} KB")

    if restored_image_size < (modified_image_size + mask_size):
        print("حجم تصویر بازسازی‌شده (restored_image.jpg) کمتر از مجموع حجم فایل‌های ورودی است.")
    else:
        print("حجم تصویر بازسازی‌شده (restored_image.jpg) بیشتر از مجموع حجم فایل‌های ورودی است. لطفاً پارامترهای فشرده‌سازی را تنظیم کنید.")

    print("فایل restored_image.jpg ذخیره شد.")
    print("پایان کد دوم.")

if __name__ == "__main__":
    main()
