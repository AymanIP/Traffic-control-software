# إستيراد المكتبات المطلوبه
import cv2 # مكتبة OpenCV لمعالجة الصور والفيديو

classnames = [] # إعداد قائمة فارغة لأسماء الفئات
classfile = 'files/coco.names' # إنشاء متغير لتحديد ملف الفئات الذي يحتوي على أسماء الكائنات المعروفة

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

p = 'files/frozen_inference_graph.pb' # تحديد مسار ملفات النموذج المجمد والتكوين
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v) # كشف وفحص وتحديد النموذج المجمد وإسم الكائن في كل إطار
net.setInputSize(320, 320)         # تحديد عرض وإرتفاع النموذج المجمد
net.setInputScale(1.0 / 127.5)     # تحديد القياس
net.setInputMean((127.5, 127.5, 127.5)) # تعيين متوسط القيم للبيانات المدخله
net.setInputSwapRB(True)           # نظام الألوان

cap = cv2.VideoCapture(0) # تحديد رقم الكاميرا، 0 تعني الكاميرا الافتراضية

# حلقة تكراريه لكل إطار في الفيديو
while True:
    ret, frame = cap.read() # قراءة إطار من الكاميرا

    results = net.detect(frame, confThreshold=0.48) # تنفيذ الكشف عن السيارات باستخدام نموذج الكشف المتدرج وتم تحديد عتبة الثقة بقيمة 0.48
    if isinstance(results, tuple):
            classIds = results[0]
            confs = results[1]
            bbox = results[2]
    else:
        classIds, confs, bbox = results
    numCars = 0 # عدد السيارات المبدئي
    Command = 'OFF' # الأمر المبدئي

    if len(classIds) > 0:
            for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                

                if classId > 0 and classId <= len(classnames) and classnames[classId - 1] == 'car': # للتحقق من أن الكائن المكتشف هو سيارة
                    cv2.rectangle(frame, box, color = (255, 0, 0), thickness = 2 ) # رسم مربع أزرق يحيط بالكائن
                    cv2.putText(frame, 'Car', (box[0] + 10, box[1] + 20), cv2.FONT_ITALIC, 0.9, (0, 0, 255), thickness=2) # عرض اسم الكائن باللون الأحمر وتحديد مكان ظهور النص
                    numCars += 1

            cv2.putText(frame, 'Car Numbers: ' + str(numCars), (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), thickness=3) # عرض نص ثابت  (عدد السيارات المكتشفه) باللون الأخضر 
						
            if numCars <= 6: # تحديد الأمر الذي يتم إرساله بناءً على عدد السيارات المكتشفة، إما تشغيل أو إيقاف
                Command = 'OFF'
                print(Command)
            elif numCars >= 10:
                Command = 'ON'
                cv2.putText(frame, 'Car Numbers: ' + str(numCars), (10, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=3) # تغيير لون نص عدد السيارات المكتشفه إلى الأخضر
                print(Command)

            cv2.imshow('program', frame) # عرض الإطار امعالج

            if cv2.waitKey(1) == ord('q'): # الضغط على q لإنهاء البرنامج
                break
