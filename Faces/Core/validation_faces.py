import face_recognition
import imutils
import pickle
import time
import cv2

print("[INFO] loading encodings...")
with open('labels.pickle', 'rb') as f:
    data = pickle.loads(f.read())
cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "/haarcascade_frontalface_alt.xml")
cont = 0
print('start')
while cv2.waitKey(1) != 27:
    inicio = time.time()
    # grab the frame from the threaded video stream
    frame = cap.read()[1]
    #frame = cv2.rotate(frame, 1)
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(imutils.resize(frame, width=250), cv2.COLOR_BGR2RGB)
    r = frame.shape[1] / float(rgb.shape[1])
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    # rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60))
    # boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    boxes = face_recognition.face_locations(rgb)
    if not boxes:
        continue
    tt, rr, bb, ll = boxes[0]
    crop = frame[tt:tt + bb, ll:ll + rr]
    cv2.imwrite("crop.png", crop)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            print(counts)
            name = max(counts, key=counts.get)
            if counts[name] < 120:
                name = "Unknown"

        # update the list of namesv   1
        names.append(name)
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 10)
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
