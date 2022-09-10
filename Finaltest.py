from time import time
import cv2
import numpy as np
import exported_model_v4
import os
import math
#Initialize Object Detection
od=exported_model_v4.Model()
temp=True
temp2=False
#Split video into frame using OpenCV
cap= cv2.VideoCapture("FinalCropped.mp4")
# cap.set(cv2.CAP_PROP_FPS, 5)
# naming=10000000
count=0
BinsIn=0
BinsOut=0
centre_points_prev_frame=[]
# textfile=open("total_output.txt","a")
Bread_Bins_out=0
Bread_Bins_in=0
Empty_Bins_out=0
Empty_Bins_in=0
tracking_objects={}
tracking_distance={}
track_id=1
while True:
    ret,frame=cap.read()
    if ret==False:
        break
    if temp:
        h,w,nnnn=frame.shape
        # temp=False
    # print(hei , wid)
    # h1,h2,w1,w2=int(h),int(h/1.1),int(0),int(w)
    roi=frame[0:int(h/1.1),0:w]

    # print(count)
    count=count+1
    #points of current frame
    centre_points_cur_frame=[]
    # cv2.imwrite(("frame.jpg"),frame)
    param=cv2.resize(roi,(320,320))
    param=np.array(param, dtype=np.float32)[np.newaxis, :, :, :]
    # Detect objects on frame
    boxes,class_id,scores=od.predict(param=param)
    # os.remove("frame/frame.jpg")
    for box in boxes:
        left=int((box[0])*w)
        top=int((box[1])*h)
        height=int(((box[2])-(box[0])) * h)
        width=int(((box[3])-(box[1])) * w)
        # top=h1+top
        # left,top,height,width=left+w1,top+h1,height,width
        points=((left,top),(left+width,top),(left+width,top+height),(left,top+height),(left,top))
        cx=int((left+left+width)/2)
        cy=int((top+top+height)/2)
        centre_points_cur_frame.append((cx,cy))
        cv2.rectangle(frame,(left,top),(left+width,top+height),(0,255,0),2)
        # cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
    if count <=2:
        for pt in centre_points_cur_frame:
            # cv2.circle(frame,pt,5,(0,0,255),-1)
            for pt2 in centre_points_prev_frame:
                distance=math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                if distance<100:
                    tracking_objects[track_id]=pt
                    try:
                        tracking_distance[track_id][0]=tracking_distance[track_id][0]+distance
                    except:
                        tracking_distance[track_id]=[distance,class_id]
                    track_id+=1
                    temp2=True
                # print("count<2 distance brk: ",distance)

    else:
        if temp2:
            tracking_distance.pop(1)
            temp2=False
        tracking_objects_copy=tracking_objects.copy()
        centre_points_cur_frame_copy=centre_points_cur_frame.copy()

        for object_id,pt2 in tracking_objects_copy.items():
            object_exists=False
            for pt in centre_points_cur_frame_copy:
                distance=math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                #update object position
                if distance<100:
                    tracking_objects[object_id]=pt
                    if pt[1]<pt2[1]:
                        distance=-distance
                    try:
                        tracking_distance[track_id][0]=tracking_distance[track_id][0]+distance
                    except:
                        tracking_distance[track_id]=[distance,class_id]

                    object_exists=True
                    centre_points_prev_frame=centre_points_cur_frame.copy()
                    if pt in centre_points_cur_frame:
                        centre_points_cur_frame.remove(pt)
                    continue
            #remove lost id
            if not object_exists:
                # if tracking_distance[object_id+1][0]>0:
                #     BinsIn=BinsIn+1
                # else:
                #     BinsOut=BinsOut+1
                try:
                    if ((tracking_distance[object_id+1][0]>10) & (tracking_distance[object_id+1][1][0]=="Bread_Bin")):
                        Bread_Bins_in=Bread_Bins_in+1
                    elif ((tracking_distance[object_id+1][0]<-10) & (tracking_distance[object_id+1][1][0]=="Bread_Bin")):
                        Bread_Bins_out=Bread_Bins_out+1
                    elif ((tracking_distance[object_id+1][0]>10) & (tracking_distance[object_id+1][1][0]=="Empty_Bin")):
                        Empty_Bins_in=Empty_Bins_in+1
                    elif ((tracking_distance[object_id+1][0]<-10) & (tracking_distance[object_id+1][1][0]=="Empty_Bin")):
                        Bread_Bins_out=Bread_Bins_out+1
                    tracking_objects.pop(object_id)
                except:
                    pass
        #Add new ID found
        for pt in centre_points_cur_frame:
            tracking_objects[track_id]=pt
            track_id+=1

    for object_id,pt in tracking_objects.items():
        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
        cv2.putText(frame,str(object_id),(pt[0],pt[1]-7),0,1,(0,0,255),2)

    # naming+=1
    cv2.putText(frame,"Total Bins: "+str(track_id-1),(50,50),0,1,(0,0,255),2)
    cv2.putText(frame,"Bins In: "+str(Empty_Bins_in),(50,100),0,1,(0,0,255),2)
    cv2.putText(frame,"Bins Out: "+str(Bread_Bins_out),(50,150),0,1,(0,0,255),2)
    # cv2.putText(frame,"Empty Bins In: "+str(Empty_Bins_in),(50,200),0,1,(0,0,255),2)
    # cv2.putText(frame,"Bread Bins Out: "+str(Bread_Bins_out),(50,250),0,1,(0,0,255),2)
    # cv2.putText(frame,"Dough Bins Out: "+str(Empty_Bins_out),(50,300),0,1,(0,0,255),2)
    # cv2.putText(frame,"Total Bins Out: "+str(Bread_Bins_out+Empty_Bins_out),(50,350),0,1,(0,0,255),2)
    if temp:
        temp=False
        t=time()
    # cv2.putText(frame,"Bread Bins In: "+str(Bread_Bins_in),(50,350),0,1,(0,0,255),2)
    # cv2.imshow("Frame",frame)
    cv2.imshow("roi",roi)
    # cv2.imwrite(("FinalV2Trial/frame"+str(naming)+".jpg"),frame)
    #make a copy of points
    centre_points_prev_frame=centre_points_cur_frame.copy()
    key=cv2.waitKey(1)

    if key=='q':
        break
# textfile.write(("\n"+str(Bread_Bins_out)+","+str(Empty_Bins_out)+","+str(Bread_Bins_out+Empty_Bins_out)))
print(tracking_distance)
cap.release()
cv2.destroyAllWindows()
t2=time()
print(t2-t)

