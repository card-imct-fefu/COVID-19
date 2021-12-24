from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from centroidtracker import CentroidTracker
import datetime
import time

protopath = "important/MobileNetSSD_deploy.prototxt"
modelpath = "important/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmon7itor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

t = time.time()

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

prototxtPath = r"important\deploy.prototxt"
weightsPath = r"important\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("important/mask_detector.model")

vs = VideoStream(src=0).start()

tDif = 0
fps_start_time = datetime.datetime.now()
print(fps_start_time)
fps = 0
total_frames = 0
lpc_count = 0
opc_count = 0
object_id_list = []
name = str(fps_start_time)
name = name.replace(':', '.')
f = open(f'results/total_{name}.txt', 'w')

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=1400)
	total_frames = total_frames + 1

	(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

	detector.setInput(blob)
	person_detections = detector.forward()
	rects = []
	for i in np.arange(0, person_detections.shape[2]):
		confidence = person_detections[0, 0, i, 2]
		if confidence > 0.5:
			idx = int(person_detections[0, 0, i, 1])

			if CLASSES[idx] != "person":
				continue

			person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = person_box.astype("int")
			rects.append(person_box)

	boundingboxes = np.array(rects)
	boundingboxes = boundingboxes.astype(int)
	rects = non_max_suppression_fast(boundingboxes, 0.3)

	objects = tracker.update(rects)
	for (objectId, bbox) in objects.items():
		x1, y1, x2, y2 = bbox
		x1 = int(x1)
		y1 = int(y1)
		x2 = int(x2)
		y2 = int(y2)

		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
		text = "ID: {}".format(objectId)
		cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

		if objectId not in object_id_list:
			object_id_list.append(objectId)

	fps_end_time = datetime.datetime.now()
	time_diff = fps_end_time - fps_start_time
	if time_diff.seconds == 0:
		fps = 0.0
	else:
		fps = (total_frames / time_diff.seconds)

	fps_text = "{:.0f}".format(fps)

	cv2.putText(frame, fps_text, (0, H-3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (140, 140, 140), 1)

	lpc_count = len(objects)
	opc_count = len(object_id_list)

	lpc_txt = "Frame: {}".format(lpc_count)
	opc_txt = "Total: {}".format(opc_count)

	cv2.putText(frame, lpc_txt, (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
	cv2.putText(frame, opc_txt, (5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
	tEnd = time.time()
	tDif = tEnd - t
	if tDif > 60:
		print(str(datetime.datetime.now()) + ' - Total persons detected: ' + str(opc_count))
		f.write(str(datetime.datetime.now()) + ' - Total persons detected: ' + str(opc_count) + '\n')
		t = time.time()

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		cv2.putText(frame, label, (startX + 4, endY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

	cv2.imshow("Happy New Year :)", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("z"):
		break

cv2.destroyAllWindows()
vs.stop()