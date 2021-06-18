"""
@author: Engin Tekin
"""

from enum import IntEnum
import numpy as np
from sklearn.cluster import DBSCAN
from imutils import build_montages, paths
import os
import shutil
import cv2
import dlib
import face_recognition
from tqdm.std import tqdm
from threading import Lock, Thread

classifier = IntEnum('classifier', ['CASCADE', 'HOG','DlibCNNv1'])

class FaceClustering:
    def __init__(self, args):
        self.idx = 1
        self.display = args.d
        self.register = args.r
        self.eyes = args.e
        self.classifiertype = args.ctype
        self.classifierPath = args.cpath
        self.predictorPath = args.ppath
        self.isVideo = True if args.video  is not None else False
        self.video = args.video
        self.image = args.image
        self.font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        self.cthreshold = args.cthreshold
        self.faceSize = (256,256)
        self.nCpu = args.ncpu
        self.lock = Lock() if self.nCpu > 1 else None
        self.classifier = IntEnum('classifier', ['CASCADE', 'HOG','DlibCNNv1'])
        self.encodings = list()
        self.processFps = args.pfps

        self.desiredLeftEye = (0.35, 0.35)
        self.alignFace = args.align
        self.resizeRatio = args.resizeratio

        if self.register:

            dirPath = os.getcwd() + '/faces'
            if os.path.isdir(dirPath) is True:
                shutil.rmtree(dirPath)    
            os.mkdir(dirPath)

    def bBox2rect(self, bbox):
        """
            Converts bbox[x,y,w,h] to Dlib rect[top right bottom left]
        """
        left = bbox[0]
        top = bbox[1]
        right = left + bbox[2]
        bottom = top + bbox[3] 
        return dlib.rectangle(left, top, right, bottom)

    def bBox2css(self, bbox, frameShape):
        x, y, w, h = bbox
        top = max(0, y)
        right = min(frameShape[1], x+w)
        bottom = min(frameShape[0], y+h)
        left = max(0, x)
        return top, right, bottom, left 

    def ccs2bBox(self, ccs, frameShape):
        x = max(0, ccs.left())
        y = max(0, ccs.top())
        w = min(frameShape[1], ccs.right()-ccs.left())
        h = min(frameShape[0], ccs.bottom()-ccs.top())
        return x, y, w, h

    def preProcessing(self, frame):
        """
        Given a RGB frame this method returns a preProcessed frame
        - RGB2Gray
        - Normalization
        """
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting to Gray Image
        frame = cv2.equalizeHist(frame) #Equalizing the Image so that max pixel: 255, min pixel: 0 
    
        if self.resizeRatio != 1:
            w = int(frame.shape[1] / self.resizeRatio)
            h = int(frame.shape[0] / self.resizeRatio)
            frame = cv2.resize(frame, (w, h)) 

        return frame

    def detectFaceHoG(self, frame, classifier):
        """
        Given a Grayscale frame this method returns a list containing the bounding boxes 
        using Histogram of Oriented Gradient for detected faces if any.
        input: - frame: HxW
        output: List of rectangles: [(x,y,w,h),(x2,y2,w2,h2),...]
        """

        dets, scores, idx = classifier.run(frame,1)

        faces = []
        for d in dets:
            bbox = self.ccs2bBox(d, frame.shape)
            faces.append(bbox)

        return faces, scores

    def detectFaceCascadeClassifier(self, frame, classifier):
        """
        Given a Grayscale frame this method returns a list containing the bounding boxes 
        using Cascade Classifier for detected faces if any.
        input: - frame: HxW
        output: List of rectangles: [(x,y,w,h),(x2,y2,w2,h2),...]
        """

        faces = classifier.detectMultiScale(frame,
                                                scaleFactor=1.2,
                                                minNeighbors=5,
                                                minSize=(20,20),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

        return faces, None 

    def detectFaceCNN(self, frame, classifier):

        """
        Given a Grayscale frame this method returns a list containing the bounding boxes 
        using CNN and MMOD for detected faces if any.
        input: - frame: HxW
        output: List of rectangles: [(x,y,w,h),(x2,y2,w2,h2),...]
        """

        dets = classifier(frame, 1)

        faces = []
        confidence = []
        for d in dets:
            bbox = self.ccs2bBox(d.rect, frame.shape)
            faces.append(bbox)
            confidence.append(d.confidence)

        return faces, confidence

    def drawBBox(self, frame, faces, confidence=None):
        """
        Given an frame this method returns draw bounding boxes in the frame
        input: - frame: HxWxC
        output: List of rectangles: [(x,y,w,h),(x2,y2,w2,h2),...]
        """
        for i,face in enumerate(faces):
            x,y,w,h = face
            if confidence is None:  
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

            elif confidence[i] > self.cthreshold:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame, str(round(confidence[i],3)), (x, y), self.font, 0.7, (255,0,0), 1)

        return frame

    def drawEyes(self, frame, eyesAll):
        """
        Given frame, face and eyes draws eyes with 5 points per eye
        """

        for eyes in eyesAll.values():
            for x, y in eyes:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        return frame

    def detectEyes(self, frame, faces, confidence, predictor):
        """
        Given frame, bbox and landmark predictor return eye positions
        """
        eyesAll = dict()
        for i, bbox in enumerate(faces):
            if self.cthreshold is None or confidence[i] > self.cthreshold:
                dlibRect = self.bBox2rect(bbox)
                landmarks = predictor(frame, dlibRect)

                eyes = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,48)]

                eyesAll[i] = eyes

        return eyesAll

    def detectFaceframe(self, frame, classifier):
        """
        Given a frame returns the detected faces from specified classifier
        """
        grayFrame = self.preProcessing(frame)

        if self.classifiertype == self.classifier.CASCADE:
            return self.detectFaceCascadeClassifier(grayFrame, classifier)
        elif self.classifiertype == self.classifier.HOG:
            return self.detectFaceHoG(grayFrame, classifier)
        elif self.classifiertype == self.classifier.DlibCNNv1:
            return self.detectFaceCNN(grayFrame, classifier)

    def align(self, frame ,ROI, eyes):
        """
        input: - ROI: HxWxBGR
               - eyes: [0:6] leftEye , [6:] rightEye
        Align Face Rotation, Translation, Scaling
        """

        eyes = np.array(eyes)
        leftEyeCenter = eyes[0:6].mean(axis=0).astype('int')
        rightEyeCenter = eyes[6:].mean(axis=0).astype('int')

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.faceSize[0]
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        
        eyesCenter = [eyesCenter[0].tolist(), eyesCenter[1].tolist()]

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        tX = self.faceSize[0] * 0.5
        tY = self.faceSize[1] * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        aligned = cv2.warpAffine(frame, M, self.faceSize,
            flags=cv2.INTER_CUBIC)

        return aligned

    def registerFace(self, frame, faces, confidence=None, eyesAll=None):
        """
        Given a frame and faces saves the ROI to file
        """
        for i, face in enumerate(faces):
            if confidence is None or confidence[i] > self.cthreshold:

                x,y,w,h = face
                x = np.clip(x, 0, min(x+w, frame.shape[1])-1)
                y = np.clip(y, 0, min(y+h, frame.shape[0])-1)
                x2 = np.clip(x+w, 0, frame.shape[1]-1)
                y2 = np.clip(y+h, 0, frame.shape[0]-1)
                ROI = frame[y:y2, x:x2]
                ROI = cv2.resize(ROI, self.faceSize, interpolation=cv2.INTER_AREA)

                if self.alignFace:
                    ROI = self.align(frame, ROI, eyesAll[i])

                if self.lock is not None:
                    self.lock.acquire()
                    idx = self.idx
                    self.idx += 1
                    self.lock.release()
                else:
                    idx = self.idx
                    self.idx += 1

                cv2.imwrite(f'faces/{idx}.jpg', ROI)
    
    def getLandmarkDetector(self):
        return dlib.shape_predictor(self.predictorPath)

    def getClassifier(self):

        if self.classifiertype == self.classifier.CASCADE:
            classifier = cv2.CascadeClassifier(self.classifierPath)

        elif self.classifiertype == self.classifier.HOG:
            classifier = dlib.get_frontal_face_detector()

        elif self.classifiertype == self.classifier.DlibCNNv1:
            classifier = dlib.cnn_face_detection_model_v1(self.classifierPath)

        else:
            print("Error: Unknown Classifier")
            exit(1)

        return  classifier

    def videoSingleProcessDetection(self, frameNos, pbar):
        """
            Single Process for multi-processing
        """
        video = cv2.VideoCapture(self.video)
        classifier = self.getClassifier()

        if self.alignFace:
            predictor = self.getLandmarkDetector()

        if video.isOpened() == False:
            print("Error: Cannot Open the Video")
            exit(1)

        for frameNo in frameNos:
            video.set(1, frameNo)
            ret, frame = video.read()
            if ret:
                faces, confidence = self.detectFaceframe(frame, classifier)

                eyesAll = None
                if self.eyes or self.alignFace:
                    eyesAll = self.detectEyes(frame, faces, confidence, predictor)

                if self.register:
                    self.registerFace(frame, faces, confidence, eyesAll)
                if self.display and self.nCpu == 1:
                    frame = self.drawBBox(frame, faces, confidence)
                    if self.eyes:
                        frame = self.drawEyes(frame, eyesAll)
                    cv2.imshow('Video',frame)
                    cv2.waitKey(1)

            pbar.update()

    def detectFaces(self):
        self.idx = 1 # person register index

        if self.isVideo:
            video = cv2.VideoCapture(self.video)
            frameCount = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = round(video.get(cv2.CAP_PROP_FPS))
            video.release()

            framesToProcess = np.arange(0, frameCount, fps//self.processFps if self.processFps else 1)
            pbar = tqdm(total=framesToProcess.shape[0], desc='Video Processing - Face Detection')

            if self.nCpu > 1:

                framesToProcess = np.array_split(framesToProcess, self.nCpu)
                threadPool = []

                for i in range(0, self.nCpu-1):
                    thread = Thread(target=self.videoSingleProcessDetection, args=(framesToProcess[i], pbar))
                    thread.start()
                    threadPool.append(thread)

                self.videoSingleProcessDetection(framesToProcess[self.nCpu-1], pbar)

                for thread in threadPool:
                    thread.join()   

            else:
                self.videoSingleProcessDetection(framesToProcess, pbar)    

        else:

            image = cv2.imread(self.image)

            if image is not None:
                faces, confidence = self.detectFaceframe(image, self.getClassifier())
                eyesAll = None
                if self.eyes or self.alignFace:
                    eyesAll = self.detectEyes(image, faces, confidence, self.getLandmarkDetector())

                if self.register:
                    self.registerFace(image, faces, confidence, eyesAll)
                if self.display:
                    image = self.drawBBox(image, faces, confidence)
                    if self.eyes:
                        image = self.drawEyes(image, eyesAll)
                    
                    cv2.imshow('Image',image)
                    cv2.waitKey(0)

    def getClusters(self):

        clt = DBSCAN(metric="euclidean", min_samples=1, n_jobs=self.nCpu)
        encodings = [d["encoding"] for d in self.encodings]

        clusters = clt.fit_predict(encodings)

        for i in range(0, len(self.encodings)):
            self.encodings[i]["ID"] = clusters[i]

    def faceRecognitionSingleProcess(self, imagePaths, pbar):

        classifier = self.getClassifier()

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces, _ = self.detectFaceframe(image, classifier) #[left top width height]
            faces = [self.bBox2css(face, image.shape) for face in faces]    

            encodings = face_recognition.face_encodings(rgb, faces)
            d = [{"imagePath": imagePath, "encoding": encoding} for encoding in encodings]
            if self.lock is not None:
                self.lock.acquire()
                self.encodings.extend(d)
                self.lock.release()
            else:
                self.encodings.extend(d)

            pbar.update()

    def faceRecognition(self):

        imagePaths = list(paths.list_images('faces'))

        pbar = tqdm(total=len(imagePaths), desc='Video Processing - Face Clustering')

        if self.nCpu > 1:

                framesToProcess = np.array_split(imagePaths, self.nCpu)
                threadPool = []

                for i in range(0, self.nCpu-1):
                    thread = Thread(target=self.faceRecognitionSingleProcess, args=(framesToProcess[i], pbar))
                    thread.start()
                    threadPool.append(thread)

                self.faceRecognitionSingleProcess(framesToProcess[self.nCpu-1], pbar)

                for thread in threadPool:
                    thread.join()   

        else:
            self.faceRecognitionSingleProcess(imagePaths, pbar) 

        self.getClusters()
        montage = None

        if self.register:

            parentPath = os.getcwd() + '/IDs'

            if os.path.isdir(parentPath) is True:
                shutil.rmtree('IDs')
                
            os.mkdir(parentPath)
            os.mkdir(os.path.join(parentPath, 'UniqueIDs'))
            os.mkdir(os.path.join(parentPath, 'all'))
            
            uniqueIDs = dict()

            for face in self.encodings:
                 uniqueIDs[face["ID"]] = list()

            for ID in uniqueIDs.keys():
                IDpath = os.path.join(parentPath + '/all', str(ID))
                os.mkdir(IDpath)

            for i, face in enumerate(self.encodings):
                uniqueIDs[face["ID"]].append(face["imagePath"])
                shutil.copyfile(face["imagePath"], f'IDs/all/{face["ID"]}/{i}.jpg')

            for ID, faces in uniqueIDs.items():
                shutil.copyfile(faces[0], f'IDs/UniqueIDs/{str(ID)}.jpg')

            
            nCols = 1; nRows = 1; row = True
            while (nRows * nCols) < len(uniqueIDs.keys()):
                if row:
                    nRows += 1
                    row = False
                else:
                    nCols += 1
                    row = True


            IDs = list()
            for IDPath in uniqueIDs.values():
                image = cv2.imread(IDPath[0])
                image = cv2.resize(image, (96, 96))
                IDs.append(image)

            montage = build_montages(IDs, (96, 96), (nRows,nCols))[0]
            cv2.imwrite(f'IDs/motage.jpg', montage)

        if self.display:
            if montage is None:
                uniqueIDs = dict()
                for face in self.encodings:
                    uniqueIDs[face["ID"]] = list()
                for i, face in enumerate(self.encodings):
                    uniqueIDs[face["ID"]].append(face["imagePath"])

                nCols = 1; nRows = 1; row = True
                while (nRows * nCols) < len(uniqueIDs.keys()):
                    if row:
                        nRows += 1
                        row = False
                    else:
                        nCols += 1
                        row = True

                IDs = list()
                for IDPath in uniqueIDs.values():
                    image = cv2.imread(IDPath[0])
                    image = cv2.resize(image, (96, 96))
                    IDs.append(image)

                montage = build_montages(IDs, (96, 96), (nRows,nCols))[0]
            

            cv2.imshow("Unique Faces", montage)
            cv2.waitKey(0)

            


