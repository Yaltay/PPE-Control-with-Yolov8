import cv2
import cvzone
import math
from ultralytics import YOLO


class PPEChecker:
    def __init__(self, model_path, video_source = None , img_source = None):
        # Video kaynağını ve YOLO modelini başlat
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO(model_path)
        self.imgSystem = cv2.imread(img_source)
        self.proces_image = None
        

        # Sınıf isimleri ve renk haritaları
        self.class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                            'Safety Vest', 'machinery', 'vehicle']

        self.color_map = {
            'safe': (0, 255, 0),  # Yeşil (Güvenli)
            'unsafe': (0, 0, 255),  # Kırmızı (Güvensiz)
        }
        self.no_mask_class = 'NO-Mask'
        self.no_vest_class = 'NO-Safety Vest'
        self.no_hardhat_class = 'NO-Hardhat'

    def process_frame(self, img):
        # Sonuçları al
        results = self.model(img, stream=True)

        # Her kişi için ayrı kontrol yap
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding Box koordinatlarını al
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                # Güvenlik kontrolünü yap
                if conf > 0.5:
                    
                    self.draw_box_and_label(img, x1, y1, x2, y2, current_class, conf)

        return img

    def check_person_safety(self, img, boxes, x1, y1, x2, y2):
        
        """Her kişi için güvenlik ekipmanlarını kontrol eder."""
        person_has_no_mask = False
        person_has_no_vest = False
        person_has_no_hardhat = False

        # Aynı kişiye ait diğer etiketleri kontrol et
        for inner_box in boxes:
            inner_cls = int(inner_box.cls[0])
            inner_label = self.model.names[inner_cls]
            inner_x1, inner_y1, inner_x2, inner_y2 = [int(coord) for coord in inner_box.xyxy[0]]

            # Eğer bounding box'lar çakışıyorsa, eksik ekipmanları kontrol et
            if (inner_x1 >= x1 and inner_y1 >= y1 and inner_x2 <= x2 and inner_y2 <= y2):
                if inner_label == self.no_mask_class:
                    person_has_no_mask = True
                if inner_label == self.no_vest_class:
                    person_has_no_vest = True

                if inner_label == self.no_hardhat_class:
                    person_has_no_hardhat = True


        # Eğer eksik güvenlik ekipmanı varsa kırmızı, yoksa yeşil kutu yap
        if (person_has_no_mask and person_has_no_vest) or (person_has_no_hardhat and person_has_no_vest) or (
                person_has_no_mask and person_has_no_hardhat):
            my_color = self.color_map['unsafe']  # Kırmızı
            print("DİKKAT Tehlikede Personel Var")
        else:
            my_color = self.color_map['safe']  # Yeşil (güvenlik ekipmanları tam)

        # Yarı saydam dolgu ekle
        img = self.draw_transparent_box(img, x1, y1, x2, y2, my_color)
        return img

    def draw_transparent_box(self, img, x1, y1, x2, y2, color, alpha=0.5):
        """Yarı saydam bir kutu çizer."""
        print("Draw Çalıştı")
        #♦overlay = img.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), color)
        #img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img

    def draw_box_and_label(self, img, x1, y1, x2, y2, label, conf):
        """Etiketleri kutular ile çizer."""
        my_color = (0, 255, 0) if "NO-" not in label else (0, 0, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), my_color, 3)
        cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=my_color,
                           colorT=(255, 255, 255), colorR=my_color, offset=5)

    def runVideo(self):
        """Videoyu döngüye alıp işlem yapar."""
        while True:
            success, img = self.cap.read()
            if not success:
                break

            # Frame'i işleyelim
            img = self.process_frame(img)

            # Görüntüyü göster
            cv2.imshow("Image", img)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
    def runImage(self):
        """Videoyu döngüye alıp işlem yapar."""
    
        img = cv2.resize(self.imgSystem , (1280,720))
        # Frame'i işleyelim
        img = self.process_frame(img)
        self.proces_image = img
        # Görüntüyü göster
        cv2.imshow("Image", img)
        
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
    
    def saveImage(self):
        cv2.imwrite("savedImage.jpg", self.proces_image )

        


