import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def detect_abnormalities(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ File not found or unsupported format: {image_path}")
        return
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original X-ray")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Detected Abnormalities")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
image_path = "Brain MRI.jpg"
detect_abnormalities(image_path)
