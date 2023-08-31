import cv2,time
import numpy as np
import telepot

np.random.seed(20)
class Detector:
    def __init__(self,videoPath,configPath,modelPath,classesPath):
        self.videoPath=videoPath
        self.configPath=configPath
        self.modelPath=modelPath
        self.classesPath=classesPath
        self.net=cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)
        self.readClasses()
    def readClasses(self):
        with open(self.classesPath,'r') as f:
            self.classesList=f.read().splitlines()
        self.classesList.insert(0,'__Background__')
        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList),3))#1
        # print(self.classesList)
    def onVideo(self):
        cap=cv2.VideoCapture(self.videoPath)
        if (cap.isOpened()==False):
            print("error ...")
            return
        (success,image)=cap.read()
        startTime=0#6
        while success:
            currentTime=time.time()#6
            fps=1/(currentTime-startTime)#6
            startTime=currentTime
            classLabelIDs,confidences,bboxs=self.net.detect(image,confThreshold=0.5)
            bboxs=list(bboxs)
            confidences=list(np.array(confidences).reshape(1,-1)[0])
            confidences=list(map(float,confidences))
            # bboxsIdx=cv2.dnn.NMSBoxes(bboxs,confidences,score_threshold=0.5,nms_threshold=0.2)
            bboxsIdx=cv2.dnn.NMSBoxes(bboxs,confidences,score_threshold=0.5,nms_threshold=0.8)#5

            if len(bboxsIdx)!=0:
                for i in range(0,len(bboxsIdx)):
                    bbox=bboxs[np.squeeze(bboxsIdx[i])]
                    classConfidence=confidences[np.squeeze(bboxsIdx[i])]
                    classLabelID=np.squeeze(classLabelIDs[np.squeeze(bboxsIdx[i])])
                    classLabel=self.classesList[classLabelID]
                    classColor=[int(c) for c in self.colorList[classLabelID]]#1
                    # displayText="{}:{:.4f}".format(classLabel,classConfidence)#2
                    displayText="{}:{:.2f}".format(classLabel,classConfidence)#2
                    x,y,w,h=bbox
                    x1=(x+w)/2
                    y1=(y+h)/2
                    b=(0,0)[0]<x1<(800,500)[0] and (0,0)[1]<y1<(800,500)[1]
                    if b:
                        token = "6275415240:AAF3yDdT45-VIn8GdBrQUHH0XmtMXo0MC28"
                        receiver_id=5877612764
                        bot = telepot.Bot(token)
                        a=cv2.imwrite("a.jpg",image)
                        bot.sendPhoto(receiver_id,photo=open("a.jpg", "rb"),caption="Có xâm nhập, nguy hiêm!")
                    cv2.rectangle(image,(x,y),(x+w,y+h),color=classColor,thickness=1)
                    cv2.rectangle(image,(0,0),(800,500),(0,255,0),2)
                    cv2.putText(image,displayText,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,classColor,2)#2
                    # lineWidth=30#3
                    lineWidth=min(int(w*0.3),int(h*0.3))#4
                    cv2.line(image,(x,y),(x+lineWidth,y),classColor,thickness=5)#3
                    cv2.line(image,(x,y),(x,y+lineWidth),classColor,thickness=5)#3

                    cv2.line(image,(x+w,y),(x+w-lineWidth,y),classColor,thickness=5)#3
                    cv2.line(image,(x+w,y),(x+w,y+lineWidth),classColor,thickness=5)#3

                    cv2.line(image,(x,y+h),(x+lineWidth,y+h),classColor,thickness=5)#3
                    cv2.line(image,(x,y+h),(x,y+h-lineWidth),classColor,thickness=5)#3

                    cv2.line(image,(x+w,y+h),(x+w-lineWidth,y+h),classColor,thickness=5)#3
                    cv2.line(image,(x+w,y+h),(x+w,y+h-lineWidth),classColor,thickness=5)#3
            cv2.putText(image,"FPS:"+str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)#6
            cv2.imshow("Result",image)
            key=cv2.waitKey(1)
            if key==ord('q'):
                break
            (success,image)=cap.read()
        cv2.destroyAllWindows()
