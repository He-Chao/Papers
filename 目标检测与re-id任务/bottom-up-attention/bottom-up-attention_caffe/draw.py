import cv2
import os
path='data\\CUHK-PEDES\\imgs\\cam_a'
count=0
for files in os.listdir(path):
	address=os.path.join(path,files)
	name=os.path.splitext(files)
	img1=cv2.imread(address)
	img2=cv2.imread(address)
	img3=cv2.imread(address)
#1
	for m in range(0,33,16):
		for n in range(0,87,43):
			imgmn=img1[n:n+43,m:m+16]
			cv2.imwrite(os.path.join('save1',name[0]+str(m)+str(n)+name[1]),imgmn)
 
#2
	img21=img2[0:25,0:48]
	img22=img2[26:50,0:48]
	img23=img2[51:75,0:48]
	img24=img2[76:101,0:48]
	img25=img2[102:128,0:48]
	
#3	
	img31=img3[0:28,0:48]
	img32=img3[29:60,0:48]
	img33=img3[50:128,0:48]
#4		
	for x in range(0,43,7):
		for y in range (0,127,21):
			if x==42 or y==126:
				imgxy=img1[y:148,x:48]	
			imgxy=img1[y:y+18,x:x+7]
			cv2.imwrite(os.path.join('save4',name[0]+str(x)+str(y)+name[1]),imgxy)
	

#2
	cv2.imwrite(os.path.join('save2',name[0]+'21'+name[1]),img21)
	cv2.imwrite(os.path.join('save2',name[0]+'22'+name[1]),img22)
	cv2.imwrite(os.path.join('save2',name[0]+'23'+name[1]),img23)
	cv2.imwrite(os.path.join('save2',name[0]+'24'+name[1]),img24)
	cv2.imwrite(os.path.join('save2',name[0]+'25'+name[1]),img25)
#3
	cv2.imwrite(os.path.join('save3',name[0]+'31'+name[1]),img31)
	cv2.imwrite(os.path.join('save3',name[0]+'32'+name[1]),img32)
	cv2.imwrite(os.path.join('save3',name[0]+'33'+name[1]),img33)

	
#	count=count+1
#	if count>10:
	break
