import numpy
from imutils import paths
from face_recognition import face_encodings, face_locations
import pickle
import cv2
import os

knownEncodings = []
knownNames = []
imagePaths = list(paths.list_images('Faces/faces'))   #('..\\faces\\'))
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    print("[INFO] processing image by {} - {}/{}".format(name, i + 1, len(imagePaths)))
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_locations(rgb)
    boxes_np = numpy.array(boxes)
    # compute the facial embedding for the face
    encodings = face_encodings(rgb, boxes_np)
    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)
print("[INFO] serializing encodings...")
data = {"encodings": numpy.array(knownEncodings), "names": numpy.array(knownNames)}
f = open("labels.pickle", "wb")
f.write(pickle.dumps(data))
f.close()