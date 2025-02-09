from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("Videos/ppe-2.mp4")  # For Video

model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

# Her kişi için kontrol edilen güvenlik ekipmanları
no_mask_class = 'NO-Mask'
no_vest_class = 'NO-Safety Vest'
no_hardhat_class = 'NO-Hardhat'

# Etiket ve renk haritası
color_map = {
    'safe': (0, 255, 0),  # Yeşil (Güvenli)
    'unsafe': (0, 0, 255),  # Kırmızı (Güvensiz)
}
myColor = (0,0,0)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Her kişi için ayrı kontrol yap
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            label = model.names[cls]

            # Kişi kontrolü yap
            
            if conf > 0.5:
                
                    
                    
                if currentClass == 'Person':
                    person_has_no_mask = False
                    person_has_no_vest = False
                    person_has_no_hardhat = False

                    # Aynı kişiye ait diğer etiketleri kontrol et
                    for inner_box in boxes:
                        inner_cls = int(inner_box.cls[0])
                        inner_label = model.names[inner_cls]
                        inner_x1, inner_y1, inner_x2, inner_y2 = [int(coord) for coord in inner_box.xyxy[0]]

                        # Eğer bounding box'lar çakışıyorsa (aynı kişiyle ilgili), eksik ekipmanları kontrol et
                        if (inner_x1 >= x1 and inner_y1 >= y1 and inner_x2 <= x2 and inner_y2 <= y2):
                            if inner_label == no_mask_class:
                                person_has_no_mask = True
                            if inner_label == no_vest_class:
                                person_has_no_vest = True
                            if inner_label == no_hardhat_class:
                                person_has_no_hardhat = True

                    # Eğer eksik güvenlik ekipmanı varsa kutuyu kırmızı yap
                    if (person_has_no_mask and person_has_no_vest) or (person_has_no_hardhat and person_has_no_vest) or (person_has_no_mask and person_has_no_hardhat):
                        myColor = color_map['unsafe']  # Kırmızı
                        print("UNSAFE")
                        overlay = img.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (myColor), -1)
                        img = cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0)
                    else:
                        myColor = color_map['safe']  # Yeşil (güvenlik ekipmanları tam)
                        print("SAFE")
                        overlay = img.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (myColor), -1)
                        img = cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0)
                    # Kutuyu çiz
                    
                
                if currentClass == 'Hardhat' or  currentClass == 'Mask' or currentClass =="Safety Cone" or currentClass == "Safety Vest":
                    myColor = (0,255,0)
                if currentClass == 'NO-Hardhat' or  currentClass == 'NO-Mask' or currentClass =="NO-Safety Vest" :
                    myColor = (0,0,200)
                
                if currentClass != "Person":
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3 )
                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)

    # Görüntüyü göster
    cv2.imshow("Image", img)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
