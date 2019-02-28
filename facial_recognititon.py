from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# Construct args and parse
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model to use: `hog` or `cnn`")
args = vars(ap.parse_args())

# Load data
print("[INIT] Loading data...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Start video feed and ready file writer
print("[INIT] Starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2)

# Loop over stream frames
while True:
    # Read frame
    frame = vs.read()

    # Convert colors of input then resize to 750px for better processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # Determine (x, y) coords of bounding boxes per face and compute
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop over faces
    for encoding in encodings:
        # Attempt to match
        try:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # Check for match
            if True in matches:
                # Find index of all match faces and use dictionary to store number of times a face was found
                matchedIdxs = [i for(i, b) in enumerate(matches) if b]
                counts = {}

                # Loop over matched indexes and maintain count per face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Determine face with largest number of votes
                name = max(counts, key=counts.get())

            # Update list of names
            names.append(name)
        except ValueError:
            print("[ERROR] No faces detected")

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # if the video writer is None *AND* we are supposed to write the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
