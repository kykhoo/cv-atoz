from flask import Flask, render_template, request
import numpy as np
import cv2
from base64 import b64decode, b64encode
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew

def rectContour(contours):
    #check if the contour is rectangle shape by checking is the
    #given contours provide 4 point of edges
    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def splitBoxes(img):
    rows = np.vsplit(img,5) #verticle split
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,5) #horizontal split
        for box in cols:
            boxes.append(box)
    return boxes

def drawGrid(img,questions=5,choices=5):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)

    return img

def showAnswers(img,myIndex,grading,ans,questions=5,choices=5):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)
    
    for x in range(0,questions):
        myAns= myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x]==1:
            myColor = (0,255,0)
            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
        else:
            myColor = (0,0,255)
            #cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img,((correctAns * secW)+secW//2, (x * secH)+secH//2),
            20,myColor,cv2.FILLED)
    
    return img

@app.route('/', methods=['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', user_image = '')
    else:
        txt64 = request.form['txt64']
        encoded_data = txt64.split(',')[1]
        encoded_data = b64decode(encoded_data)
        nparr = np.frombuffer(encoded_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        hands, img = detector.findHands(img)
        totalfingers=0
        if hands:
            fingers = detector.fingersUp(hands[0])
            totalfingers = fingers.count(1)
        
        cv2.putText(img,f'{int(totalfingers)}',(50,70), cv2.FONT_HERSHEY_PLAIN,5,(250,0,0),5)
            
        _, im_arr = cv2.imencode('.png', img)
        im_bytes = im_arr.tobytes()
        im_b64 = b64encode(im_bytes).decode("utf-8")

        return render_template('index.html', user_image = im_b64)
    
@app.route("/api/info", methods=['GET','POST'])
def api_info():
    txt64 = request.form.get("todo")
    encoded_data = txt64.split(',')[1]
    encoded_data = b64decode(encoded_data)
    nparr = np.frombuffer(encoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    hands, img = detector.findHands(img)
    totalfingers=0
    if hands:
        fingers = detector.fingersUp(hands[0])
        totalfingers = fingers.count(1)
    cv2.putText(img,f'{int(totalfingers)}',(50,70), cv2.FONT_HERSHEY_PLAIN,5,(250,0,0),5)
    _, im_arr = cv2.imencode('.png', img)
    im_bytes = im_arr.tobytes()
    im_b64 = b64encode(im_bytes).decode("utf-8")
    return im_b64

@app.route("/api/checkanswer", methods=['GET','POST'])
def check_answer():
    txt64 = request.form.get("todo")
    encoded_data = txt64.split(',')[1]
    encoded_data = b64decode(encoded_data)
    nparr = np.frombuffer(encoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    widthImg = 700
    heighImg =  700
    questions =5
    choices =5
    ans =[1,2,0,1,4]
    
    #preprocessing
    img = cv2.resize(img,(widthImg,heighImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 

    countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, countours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
    
    rectCon=rectContour(countours)
    biggestContour = getCornerPoints(rectCon[0]) #Answering/Marking Area
    gradePoints = getCornerPoints(rectCon[1]) #Grade Area
    
    if biggestContour.size != 0 and gradePoints.size != 0:
        cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)
        biggestContour = reorder(biggestContour)
        gradePoints = reorder(gradePoints)
        
        pt1 = np.float32(biggestContour)
        pt2 = np.float32([[0,0],[widthImg,0],[0,heighImg],[widthImg,heighImg]])
        matrix = cv2.getPerspectiveTransform(pt1,pt2)
        imgWarpColored = cv2.warpPerspective(img, matrix,(widthImg,heighImg))

        ptG1 = np.float32(gradePoints)
        ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
        matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG,(325,150))
        
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY) 
        imgThresh = cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

        boxes = splitBoxes(imgThresh)
        myPixelVal = np.zeros((questions,choices))

        countR = 0
        countC = 0
        
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC = countC+1
            if (countC==choices):
                countR = countR+1
                countC = 0

        myIndex = []
        for x in range(0,questions):
            arr = myPixelVal[x]
            #print('arr',arr)
            myIndexVal = np.where(arr==np.amax(arr))
            #print(myIndexVal[0])
            myIndex.append(myIndexVal[0][0])
        
        grading =[]
        for x in range(0,questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)

        score = (sum(grading)/questions) * 100 #final grade
        
        imgResult = imgWarpColored.copy()
        imgResult = showAnswers(imgResult,myIndex,grading,ans,questions,choices)
        imgRawDrawing = np.zeros_like(imgWarpColored)
        imgRawDrawing = showAnswers(imgRawDrawing,myIndex,grading,ans,questions,choices)
        invmatrix = cv2.getPerspectiveTransform(pt2,pt1)
        imgInvWarp = cv2.warpPerspective(imgRawDrawing, invmatrix,(widthImg,heighImg))
        imgRawGrade = np.zeros_like(imgGradeDisplay)
        cv2.putText(imgRawGrade, str(int(score))+"%",(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
        invmatrixG = cv2.getPerspectiveTransform(ptG2,ptG1)
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invmatrixG,(widthImg,heighImg))

        imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
        imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)
        
         
        
        


    
    _, im_arr = cv2.imencode('.png', imgFinal)
    im_bytes = im_arr.tobytes()
    im_b64 = b64encode(im_bytes).decode("utf-8")
    return im_b64
