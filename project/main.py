import face_recognition
import os
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from imutils.video import VideoStream
from imutils import resize
import time
import cv2


def create_encodings_pickle():
    directory = os.fsencode("data/raw")
    data = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        person_directory = os.path.join("data/raw", filename)
        data[filename] = []
        print("Processing faces of {}.".format(filename))
        for image in tqdm(os.listdir(person_directory)):
            loaded_image = face_recognition.load_image_file(os.path.join(person_directory, image))
            encodings = face_recognition.face_encodings(loaded_image)
            encodings and data[filename].append(encodings[0])
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)


def load_encodings_from_pickle():
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
    X, y = [], []
    for key in data:
        for item in data[key]:
            X.append(item)
            y.append(key)
    return np.array(X), y


def create_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # LinearSVC_classifier = SklearnClassifier(SVC(kernel='linear',probability=True))
    clf = LinearSVC(random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Finished training, score {}".format(score))
    with open("classifier.h5", "wb") as f:
        pickle.dump(clf, f)


def load_classifier():
    with open("classifier.h5", "rb") as f:
        clf = pickle.load(f)
    loaded_image = face_recognition.load_image_file("data/test-maciek.png")
    encodings = face_recognition.face_encodings(loaded_image)
    prediction = None
    if encodings:
        prediction = clf.predict(encodings[0].reshape(1, -1))
    print(prediction)


def video_stream_detection():
    with open("classifier.h5", "rb") as f:
        clf = pickle.load(f)

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 400)
    video_capture.set(4, 600)

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]


        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                prediction = clf.predict(face_encoding.reshape(1, -1))[0]
                face_names.append(prediction)
        
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # if encodings:
                # prediction = clf.predict(encodings[0].reshape(1, -1))[0]
                # top, right, bottom, left = locations[0]
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, prediction, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # process_this_frame = not process_this_frame

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    # vs = VideoStream().start()
    # time.sleep(2.0)
    # while True:
    #     frame = vs.read()
    #     frame = resize(frame, width=400)
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # rects = detector(gray, 0)
    #     # for rect in rects:
    #     #     shape = predictor(gray, rect)
    #     #     shape = face_utils.shape_to_np(shape)
    #     #     for (x, y) in shape:
    #     #         cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    #     cv2.imshow("Frame", frame)
    #     if cv2.getWindowProperty("Frame", 0) >= 0:
    #         break
    # cv2.destroyAllWindows()
    # vs.stop()

video_stream_detection()
# load_classifier()
# pickle_x, pickle_y = load_encodings_from_pickle()
# create_classifier(pickle_x, pickle_y)