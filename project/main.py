import os
import pickle
import time

import cv2
import face_recognition
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm


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
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Finished training, score {}".format(score))
    with open("classifier.h5", "wb") as f:
        pickle.dump(clf, f)


def load_classifier():
    with open("classifier.h5", "rb") as f:
        clf = pickle.load(f)
    loaded_image = face_recognition.load_image_file("data/testing-adam.png")
    encodings = face_recognition.face_encodings(loaded_image)
    prediction = None
    if encodings:
        prediction = clf.predict_proba(encodings[0].reshape(1, -1))
    print(prediction)


def equalize_histogram(image):
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)


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
        # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            # img_rgb_eq = equalize_histogram(frame)
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                face_encoding_reshaped = face_encoding.reshape(1, -1)
                probability = clf.predict_proba(face_encoding_reshaped)[0]
                if any(p > 0.8 for p in probability):
                    prob_string = ""
                    for p in probability:
                        if p > 0.8:
                            prob_string = str(p)
                    prediction = clf.predict(face_encoding_reshaped)[0]
                    person_text = "{} {}".format(prediction, prob_string)
                else:
                    person_text = "Unknown"
                face_names.append(person_text)
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # top *= 2
            # right *= 2
            # bottom *= 2
            # left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (152, 0, 152), 3)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img=frame, text=name, org=(left + 6, bottom - 6), fontFace=font, 
                        color=(255, 255, 255), bottomLeftOrigin=False, fontScale=0.5)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


from tkinter import Tk, Button, Label, PhotoImage, Canvas

class Window:
    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")
        master.geometry("810x434")

        background_image = PhotoImage(file="bg.png")
        background_label = Label(self.master, image=background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.label = Label(master, text="Facial Recognition", fg="#00ff00", bg="#333248", font=("Helvetica", 16))
        self.label.grid(row=0, sticky='s')
        self.label = Label(master, text="Adam Przywarty, Maciej Korzeniewski", fg="#00ff00", bg="#333248", font=("Helvetica", 12))
        self.label.grid(row=1, sticky='s')
        self.greet_button = Button(master, text="RUN WEBCAM!", command=self.greet, fg="#333248", bg="#00ff00", font=("Helvetica", 12))
        self.greet_button.grid(row=2, sticky='s')

        self.master.mainloop()

    def greet(self):
        video_stream_detection()

root = Tk()
Window(root)

# video_stream_detection()
# load_classifier()
# pickle_x, pickle_y = load_encodings_from_pickle()
# create_classifier(pickle_x, pickle_y)