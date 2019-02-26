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

# Loop over img paths
for i, imagePath in enumerate(imagePaths):
    print("[INFO] Processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)
    # Load img and convert it from BGR to RGB
    rgb=cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    # Determines (x, y) coords for faces
    boxes=face_recognition.face_locations(rgb,model=args["detection_method"])

    # Compute facial map
    encodings=face_recognition.face_encodings(rgb,boxes)

    # Loop over encodings
    for encoding in encodings:
        # Pair encodings and names to sets of know encodings and names
        knownEncodings.append(encoding)
        knownEncodings.append(name)

    # Dump encodings
    print("[INFO] Serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()


