import cv2

classnames = [] # Array
classfile = 'files/coco.names'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0) # تحديد رقم الكاميرا، 0 تعني الكاميرا الافتراضية

while True:
    ret, frame = cap.read() # قراءة إطار من الكاميرا

    results = net.detect(frame, confThreshold=0.48)
    if isinstance(results, tuple):
            classIds = results[0]
            confs = results[1]
            bbox = results[2]
    else:
        classIds, confs, bbox = results
    numCars = 0
    Command = 'OFF'

    if len(classIds) > 0:
            for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                

                if classId > 0 and classId <= len(classnames) and classnames[classId - 1] == 'car':
                    cv2.rectangle(frame, box, color=(255, 0, 0), thickness=2)
                    cv2.putText(frame, 'Car', (box[0] + 10, box[1] + 20), cv2.FONT_ITALIC, 0.9, (0, 0, 255), thickness=2)
                    numCars += 1

            cv2.putText(frame, 'Car Numbers: ' + str(numCars), (10, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=3)
						
            if numCars <= 6:
                Command = 'OFF'
                print(Command)
            elif numCars >= 10:
                Command = 'ON'
                cv2.putText(frame, 'Car Numbers: ' + str(numCars), (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), thickness=3)
                print(Command)

            cv2.imshow('program', frame)

            if cv2.waitKey(1) == ord('q'): # الضغط على q لإنهاء البرنامج
                break

