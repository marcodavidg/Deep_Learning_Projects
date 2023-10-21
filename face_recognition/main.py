import cv2
import time

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


if __name__ == '__main__':
    mode = "yolo"
    if mode == "yolo":
        net = cv2.dnn.readNetFromDarknet('models/yolov3.cfg', 'models/yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ln = net.getLayerNames()
        while True:
            img = cv2.imread('images/horse.jpg')
            cv2.imshow('window', img)
            cv2.waitKey(1)

            # construct a blob from the image
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            r = blob[0, 0, :, :]

            breakpoint()

            cv2.imshow('blob', r)
            text = f'Blob shape={blob.shape}'
            cv2.displayOverlay('blob', text)
            cv2.waitKey(1)

            net.setInput(blob)
            t0 = time.time()
            outputs = net.forward(ln)
            t = time.time()

            cv2.displayOverlay('window', f'forward propagation time={t - t0}')
            cv2.imshow('window', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        while True:

            face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            video_capture = cv2.VideoCapture(0)

            result, video_frame = video_capture.read()  # read frames from the video
            if result is False:
                break  # terminate the loop if the frame is not read successfully

            faces = detect_bounding_box(
                video_frame
            )  # apply the function we created to the video frame

            cv2.imshow(
                "My Face Detection Project", video_frame
            )  # display the processed frame in a window named "My Face Detection Project"

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()

