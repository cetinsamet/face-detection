import cv2
import imageio
import sys


face_cascade    = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')
#eye_cascade     = cv2.CascadeClassifier('../cascade/haarcascade_eye.xml')

def detect(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces   = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(200, 0, 0), thickness=3)
    return frame

def main(argv):

    if len(argv) != 2:
        print("Usage: python3 face_det.py inpath outpath")
        exit()

    inp, out    = argv[0], argv[1]
    reader      = imageio.get_reader(inp)
    fps         = reader.get_meta_data()['fps']
    writer      = imageio.get_writer(out, fps=fps)

    for i, frame in enumerate(reader):
        frame = detect(frame)
        writer.append_data(frame)
        print("processing %dth frame" % i)
    print("Processed video is saved to %s" % out)
    writer.close()


if __name__ == '__main__':
    main(sys.argv[1:])
