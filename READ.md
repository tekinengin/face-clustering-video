# Face Clustering Video

This is a framework to detect unique faces in videos. 

Built with [Dlib](http://dlib.net/)'s libraries. It is possible to use different face detection and recognition algorithms.

## Architecture

Input --> Preprocessing --> FaceDetection --> FaceLandmarks --> Face Alignment --> Feature Encoder --> Clustering --> UniqueIDs


## Pre-Trained Models

- OpenCV HaarCascadeClassifier: [pre-trained/haarcascade_frontalface_default.xml] 
- Dlib CNN Face Detector Modelv1: [pre-trained/mmod_human_face_detector.dat](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
- Dlib 68 Points Face Landmarks: [pre-trained/shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

## Parameters

- `--video` : Source Path
- `--ctype` : Face Detector(FD) Type 1: HaarCasCade, 2: HoG, 3: CNN
- `--cpath` : Pre-Trained FD Model Weights if any
- `--ppath` : Pre-Trained Feature Landmark Detector (Default Dlib-68-Points)
- `--ncpu`  : Number of CPUs for multi-threading
- `--cthreshold` : Threshold for face confidence
- `--resizeratio` : Resize the Input with 1/resizeratio
- `--pfps` : Processing F, Example 1 means Original Video 1 sn is 1 frame (Default Video Fps)
- `-d`     : Display Option only for --ncpus 1
- `-r`     : Saving Detected Faces and Clusters
- `-e`     : Eye Detection Option
- `-align` : Face Alignment

## Examples

- Clustering with Alignment`python main.py --ctype 2 --ncpu 4 --video="src/sampleVideo.mp4" --pfps 0.33 -r -align`

![aligned](raw.githubusercontent.com/face-clustering-video/master/examples/alignedmontage.jpg)

- Clustering without Alignment :`python main.py --ctype 2 --ncpu 4 --video="src/sampleVideo.mp4" --pfps 0.33 -r`

![nonaligned](raw.githubusercontent.com/face-clustering-video/master/examples/montage.jpg)

