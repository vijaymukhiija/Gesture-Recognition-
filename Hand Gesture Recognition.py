#Project on object detection using image processing

import cv2
import numpy as np

cv2.namedWindow("threshed",cv2.WINDOW_NORMAL)
cv2.namedWindow("contoured",cv2.WINDOW_NORMAL)
cv2.namedWindow("image1",cv2.WINDOW_NORMAL)

def nothing(x):
	pass

cap=cv2.VideoCapture(0)
#Trackbar to adjust HSV values
cv2.createTrackbar('lh','image1',0,179,nothing)
cv2.createTrackbar('ls','image1',0,179,nothing)
cv2.createTrackbar('lv','image1',0,255,nothing)
cv2.createTrackbar('hh','image1',0,255,nothing)
cv2.createTrackbar('hs','image1',0,255,nothing)
cv2.createTrackbar('hv','image1',0,255,nothing)

while(1):
	ret,frame=cap.read()
	Hl=cv2.getTrackbarPos('lh','image1')
	Sl=cv2.getTrackbarPos('ls','image1')
	Vl=cv2.getTrackbarPos('lv','image1')
	Hh=cv2.getTrackbarPos('hh','image1')
	Sh=cv2.getTrackbarPos('hs','image1')
	Vh=cv2.getTrackbarPos('hv','image1')
	image=frame
	#Filtering of an video to enhance the threshing
	blur1 = cv2.GaussianBlur(image,(3,3),0)
	morph_size = 2
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))
	cv2.morphologyEx(blur1,3, element)
	cv2.morphologyEx(blur1, 2, element)
	blur2= cv2.medianBlur(blur1,5)
	blur3 = cv2.bilateralFilter(blur2,9,75,75)
	#hsv conversion 
	hsv=cv2.cvtColor(blur3,cv2.COLOR_BGR2HSV)

	thresh = cv2.inRange(hsv,(Hl,Sl,Vl),(Hh,Sh,Vh)) 
	cv2.imshow("threshed",thresh)
	p = frame.shape
	#Draw lines
	cv2.line(frame,(p[1]/3,0),(p[1]/3,p[0]),(0,0,0),5)
	cv2.line(frame,(2*p[1]/3,0),(2*p[1]/3,p[0]),(0,0,0),5)
	cv2.line(frame,(p[1]/3,p[0]/2),(2*p[1]/3,p[0]/2),(0,0,0),5)

	max_area=0

	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#detecting the object
	for i in range(0,len(contours)):
			cnt=contours[i]
			area=cv2.contourArea(cnt)
			if(area>max_area):
				max_area=area
				j=i

	#finding the centre of mass & detecting the object
	if(max_area!=0):
		epsilon = 0.1*cv2.arcLength(contours[j],True)
		approx = cv2.approxPolyDP(contours[j],epsilon,True)
		(x,y),radius=cv2.minEnclosingCircle(contours[j])
		x=int(x)
		y=int(y)
		radius=int(radius)
		cv2.circle(frame,(x,y),radius,(0,0,255), 3)
		cx=x
		cy=y
		cv2.circle(frame,(cx,cy),10,(0,0,0),-1)

		font = cv2.FONT_HERSHEY_SIMPLEX
	
		if(cx<p[1]/3):
			cv2.putText(frame,'right',(cx,cy),font,1,(0,0,255),2)
		if(cx>2*p[1]/3):
			cv2.putText(frame,'left',(cx,cy),font,1,(0,0,255),2)
		if(cx>p[1]/3 and cx<2*p[1]/3 and cy<p[0]/2):
			cv2.putText(frame,'forward',(cx,cy),font,1,(0,0,255),2)
		if(cx>p[1]/3 and cx<2*p[1]/3 and cy>p[0]/2):
			cv2.putText(frame,'stop',(cx,cy),font,1,(0,0,255),2)


	cv2.imshow("contoured",frame)

	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

cap.release() 
cv2.destroyAllWindows()
