import cv2
import argparse
import pickle
import os
import face_recognition
from imutils import paths

# Command line args
ap=argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input dir of faces and images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="facial detecgtion model to use: hog or cnn")
args = vars(ap.parse_args())

# Get paths
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Init list of known encodings and names
knownEncodings = []
knownNames = []


