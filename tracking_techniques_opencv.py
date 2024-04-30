import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

OPENCV_OBJECT_TRACKERS = {"csrt"      : cv2.TrackerCSRT_create,
                          "kcf"       : cv2.TrackerKCF_create,
                          "boosting"  : cv2.legacy.TrackerBoosting_create,
                          "mil"       : cv2.TrackerMIL_create,
                          "tld"       : cv2.legacy.TrackerTLD_create,
                          "medianflow": cv2.legacy.TrackerMedianFlow_create,
                          "mosse"     : cv2.legacy.TrackerMOSSE_create}

tracker_name = "boosting"
tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
print("Tracker:", tracker_name)

gt = pd.read_csv("gt_new.txt") # burada veri bir data frame'e dönüşür 
# print(gt.x) # -> Bu şekilde gt içerisindeki verilere erişebiliriz
video_path = "MOT17-13-SDP.mp4"
cap = cv2.VideoCapture(video_path)

# genel parametreler
initBB = None # Seçmiş olduğumuz bulduğumuz nesneyi bu değişkenin içerisinde depolayacağız
fps = 25
frame_number = []
f = 0
# car_gt = gt[gt.frame_no == 10] 
success_track_frame_success = 0
track_list = []
start_time = time.time() 

while True:
    
    time.sleep(0.01) #Videoyu daha yavaş akıtmak için
    
    ret, frame = cap.read()
    
    if ret:
        
        frame = cv2.resize(frame, dsize = (960,540))
        (H, W) = frame.shape[:2] # 540 ,960 ,3
        # gt
        car_gt = gt[gt.frame_no == f] 
        #Eğer f değeri ground truth'un frame numarası değerine eşitse o satırın hepsini car_gt'ye eşitle
        if len(car_gt) != 0:
            x = car_gt.x.values[0] #Burada x değerini alacağız ama car_gt.x dediğimizde seri(pandas veri tipi) olarak dönüyor
           #O yüzden .values diyerek direk değeri alıyoruz yoksa satır numarası ile birlikte değer döner
            y = car_gt.y.values[0]
            w = car_gt.w.values[0]
            h = car_gt.h.values[0]
            
            center_x = car_gt.center_x.values[0]
            center_y = car_gt.center_y.values[0]
            
            cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.circle(frame, (center_x, center_y), 2,(0,0,255),-1)
            
        # box -> initBB boş değilse 
        if initBB is not None:
            
            (success, box) = tracker.update(frame) #tracker her adımda update edilir
            
            # 450. frame geldikten sonra pencereyi çizdirmenin mantığı yok
            if f<=np.max(gt.frame_no):
                (x,y,w,h) = [int(i) for i in box]
                
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                success_track_frame_success = success_track_frame_success + 1
                track_center_x = int(x+w/2)
                track_center_y = int(y+h/2)
                track_list.append([f,track_center_x,track_center_y])
                #takip ettiğim nesne ile gerçek groun truth değerleri arasındaki farka bakmak için alıyorum
                # başarım kriteri belirliycez
                
                #info içerisinde başarılı bir takip mi başarısız bir takip mi onu göreceğiz
            info = [("Tracker",tracker_name),
                    ("Success", "Yes" if success else "No")]
            
            for (i,(o,p)) in enumerate(info):
                text = "{}: {} ".format(o,p)
                cv2.putText(frame, text, (10, H -(i*20)-10), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255),2)
            
        cv2.putText(frame,"Frame Num: "+str(f), (10,30),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)
        cv2.imshow("frame",frame)    
        # key
        key = cv2.waitKey(1) & 0xFF
           
        if key == ord("t"):
            #Kutucuk seçme -> ROI -> region of interest -> takip etmek istediğimiz bölge
            initBB = cv2.selectROI("Frame", frame, fromCenter = False) 
            # fromCenter=false diyerek merkezden seçmiyceğimiz köşeden seçeceğimizi belli ediyoruz
            #initBB boxımızı depoladığımız yer
            tracker.init(frame, initBB) #takip etme algoritmalarını başlatıyoruz
            
        elif key == ord("q"):break
        
        
        # frame
        frame_number.append(f)
        f = f + 1 #Burada her while döngüsünde frame sayısını artırıyoruz
        
    else: break
    
cap.release()
cv2.destroyAllWindows()            
            
stop_time = time.time()
time_diff = stop_time - start_time          
# degerlendirme

track_df = pd.DataFrame(track_list, columns =["frame_no","center_x","center_y"])

if len(track_df) != 0:
    print("Tracking method: ",tracker)
    print("Time: ",time_diff)
    print("Number of frame to track (gt): ",len(gt))
    print("Number of frame to track (track success): ",success_track_frame_success)
    
    track_df_frame = track_df.frame_no
    
    gt_center_x = gt.center_x[track_df_frame].values
    gt_center_y = gt.center_y[track_df_frame].values
    
    track_df_center_x = track_df.center_x.values
    track_df_center_y = track_df.center_y.values
    
    plt.plot(np.sqrt((gt_center_x-track_df_center_x)**2 + (gt_center_y-track_df_center_y)**2 ))
    plt.xlabel("frame")
    plt.ylabel("Öklid mesafesi btw gt ve track")
    error = np.sum(np.sqrt((gt_center_x-track_df_center_x)**2 + (gt_center_y-track_df_center_y)**2 ))
    print("Toplam hata: ",error)






















