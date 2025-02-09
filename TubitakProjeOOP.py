import cv2
from PPEChecker import *

video_source = "Videos/ppe-2.mp4"
model_path = "Weights/ppe.pt"
img_source = "images/1.jpg"
img_source2 = "images/4.jpg"


checker = PPEChecker(model_path,video_source,img_source2)
checker.runVideo() # video Çalışacağı zaman 
#checker.runImage(img_source) # resim ile çalışacağı zaman 
checker.runImage() # resim ile çalışacağı zaman 
