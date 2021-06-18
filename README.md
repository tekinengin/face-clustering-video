# Video Based Face Clustering

This is a framework to detect unique faces in videos. 

Built with [Dlib](http://dlib.net/)'s libraries. It is possible to use different face detection and recognition algorithms.

## Architecture

Input --> Preprocessing --> FaceDetection --> FaceLandmarks --> Face Alignment (Affine) --> Feature Encoder --> Clustering --> UniqueIDs

## Pre-Trained Models

- OpenCV HaarCascadeClassifier: [pre-trained/haarcascade_frontalface_default.xml] 
- Dlib CNN Face Detector Modelv1: [pre-trained/mmod_human_face_detector.dat](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
- Dlib 68 Points Face Landmarks: [pre-trained/shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

## Installation 

 - `git clone https://github.com/tekinengin/face-clustering-video.git`
 - `curl http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 > pre-trained/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -d pre-trained/shape_predictor_68_face_landmarks.dat.bz2`
 
## Parameters

- `--video` : Source Path
- `--ctype` : Face Detector(FD) Type 1: HaarCasCade, 2: HoG, 3: CNN
- `--cpath` : Pre-Trained FD Model Weights if any
- `--ppath` : Pre-Trained Feature Landmark Detector (Default Dlib-68-Points)
- `--ncpu`  : Number of CPUs for multi-threading
- `--cthreshold` : Threshold for face confidence
- `--resizeratio` : Resize the Input with 1/resizeratio
- `--pfps` : Processing Fps, Example: skip *videoFps / pfps* frames (Default: Video Fps)
- `-d`     : Display Option only for --ncpus 1
- `-r`     : Saving Detected Faces and Clusters
- `-e`     : Eye Detection Option
- `-align` : Face Alignment

## Examples

- Clustering with Alignment (Affine) `python main.py --ctype 2 --ncpu 4 --video="src/sampleVideo.mp4" --pfps 0.33 -r -align`

![aligned](https://raw.githubusercontent.com/tekinengin/face-clustering-video/main/examples/alignedmontage.jpg)

- Clustering without Alignment :`python main.py --ctype 2 --ncpu 4 --video="src/sampleVideo.mp4" --pfps 0.33 -r`

![nonaligned](https://raw.githubusercontent.com/tekinengin/face-clustering-video/main/examples/montage.jpg)

