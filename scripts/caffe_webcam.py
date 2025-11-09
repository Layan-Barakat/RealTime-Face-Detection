import time
import cv2

def main():
    # load pretrained Caffe model (make sure the files are in models/)
    net = cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel"
    )

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    frame_count, start_time = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        (h, w) = frame.shape[:2]

        # prepare image for the DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # feed the image to the model
        net.setInput(blob)
        detections = net.forward()

        # draw boxes on detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1, y1, x2, y2 = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{confidence*100:.1f}%",
                    (x1, max(10, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        # calculate FPS
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # show the frame
        cv2.imshow("DNN Face Detection (Caffe)", frame)

        # press ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
