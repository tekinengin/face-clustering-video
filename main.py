import argparse
import cv2
import imageprocessing

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--video', help='Movie Path', type=str)
    parser.add_argument('--image', help='Movie Path', type=str)
    parser.add_argument('--ctype', help='Classifier Type: 1:cascade, 2:HoG, 3:Dlib CNN v1', type=int, default=imageprocessing.classifier.HOG)
    parser.add_argument('--cpath', help='Classifier Path', type=str,
        default='pre-trained/haarcascade_frontalface_default.xml')
    parser.add_argument('--ppath', help='Landmark Predictor Path', type=str,
        default='pre-trained/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--ncpu', help='Number of CPUs', type=int, default=1)
    parser.add_argument('--cthreshold', help='Confidence Threshold', type=float, default=0.3)
    parser.add_argument('--resizeratio', help='Resize Input Ratio [0 - n]', type=float, default=1)
    parser.add_argument('--pfps', help='Processing FPS', type=float, default=0)
    parser.add_argument('-d', action='store_true', help='To Display -d', default=False)
    parser.add_argument('-r', action='store_true', help='To Save Faces -r', default=False)
    parser.add_argument('-e', action='store_true', help='To Find Eyes', default=False)
    parser.add_argument('-align', action='store_true', help='To Align Faces', default=False)

    args = parser.parse_args()
    
    faceCluster = imageprocessing.FaceClustering(args)
    faceCluster.detectFaces()
    faceCluster.faceRecognition()

    cv2.destroyAllWindows()
