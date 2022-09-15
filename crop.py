import os

import cv2

path_dir = "/Users/younglimmm/PycharmProjects/FindingIdeal/image/validation/cat/"
file_list = os.listdir(path_dir)
print(len(file_list))
file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg", ""))


def cutting_face_save(image, names):
    face_cascade = cv2.CascadeClassifier(
        '/Users/younglimmm/PycharmProjects/FindingIdealAI/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cropped = image[y: y + h, x: x + w]
        resize = cv2.resize(cropped, (180, 180))

        cv2.imwrite(f"images/validation/dinosaur/{names}.jpg", resize)


for name in file_name_list:
    img = cv2.imread("/Users/younglimmm/PycharmProjects/FindingIdealAI/image/validation/dinosaur/"+name+".jpg")
    print(img)
    cutting_face_save(img, name)

