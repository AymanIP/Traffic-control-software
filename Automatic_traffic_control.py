import cv2 # مكتبة OpenCV لمعالجة الصور والفيديو
import RPi.GPIO as GPIO  # استيراد مكتبة RPi.GPIO للتحكم بمواقع GPIO في Raspberry Pi
import time # استيراد مكتبة الوقت لاستخدام التأخير والوقت

# تهيئة متغيرات الملفات والقوائم
classnames = [] # إعداد قائمة فارغة لأسماء الفئات
classfile = 'files/coco.names' # إنشاء متغير لتحديد ملف الفئات الذي يحتوي على أسماء الكائنات المعروفة

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

# تهيئة متغيرات النموذج وتحديد النموذج
p = 'files/frozen_inference_graph.pb' # تحديد مسار ملفات النموذج المجمد والتكوين
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v) # كشف وفحص وتحديد النموذج المجمد وإسم الكائن في كل إطار
net.setInputSize(320, 320)         # تحديد عرض وإرتفاع النموذج المجمد
net.setInputScale(1.0 / 127.5)     # تحديد القياس
net.setInputMean((127.5, 127.5, 127.5)) # تعيين متوسط القيم للبيانات المدخله
net.setInputSwapRB(True)           # نظام الألوان

cap = cv2.VideoCapture(0) # فتح الكاميرا، 0 تعني الكاميرا الافتراضية

# تحديد مخرجات وإدخالات GPIO الخاصة بالمشروع
servo_pin = 17
trig_pin = 18
echo_pin = 24
led_pin = 25

# تهيئة مواقع GPIO والمحرك والمستشعر و LED
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(trig_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)
GPIO.setup(led_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # تهيئة PWM على المخرج servo_pin بتردد 50 هرتز
pwm.start(0)

# دالة لتعريف زاوية المحرك
def set_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

#  دالة لقياس المسافة بواسطة مستشعر الموجات فوق الصوتية
def measure_distance():
    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig_pin, GPIO.LOW)
    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()
    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    return distance

# تهيئة المتغيرات الأولية
Command = "OFF"
text_color = "Green"
motor_on = False
led_on = False

# حلقة تكراريه لكل إطار في الفيديو
while True:
    ret, frame = cap.read() # قراءة إطار من الكاميرا
    distance = measure_distance()
    results = net.detect(frame, confThreshold=0.30) # تنفيذ الكشف عن السيارات باستخدام نموذج الكشف المتدرج وتم تحديد عتبة الثقة بقيمة 0.48
    
    # تحليل نتائج الكشف عن السيارات
    if isinstance(results, tuple):
       classIds = results[0]
       confs = results[1]
       bbox = results[2]
    else:
        classIds, confs, bbox = results
    numCars = 0 # عدد السيارات المبدئي
	
    # حساب عدد السيارات ورسم المستطيلات
    if len(classIds) > 0:
            for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                

                if classId > 0 and classId <= len(classnames) and classnames[classId - 1] == 'car': # للتحقق من أن الكائن المكتشف هو سيارة
                    cv2.rectangle(frame, box, color = (255, 0, 0), thickness = 2 ) # رسم مربع أزرق يحيط بالكائن
                    cv2.putText(frame, 'Car', (box[0] + 10, box[1] + 20), cv2.FONT_ITALIC, 0.9, (0, 0, 255), thickness=2)
                    numCars += 1

            if text_color == "Red":
                cv2.putText(frame, 'Car Numbers: ' + str(numCars), (10, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=3)
            elif text_color == "Green":
                cv2.putText(frame, 'Car Numbers: ' + str(numCars), (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), thickness=3)
						
            if numCars <= 6: # تحديد الأمر الذي يتم إرساله بناءً على عدد السيارات المكتشفة، إما تشغيل أو إيقاف
                if Command == "ON":
                    Command = "OFF"
                    print(Command)
                text_color = "Green"
                
                if led_on:
                    GPIO.output(led_pin, GPIO.LOW)
                    led_on = False
                if motor_on:
                    set_angle(0)
                    motor_on = False

            elif numCars >= 10:
                if Command == "OFF":
                    Command = "ON"
                    print(Command)
                text_color = "Red"
                
                if not motor_on and distance > 10:
                    if led_on:
                        GPIO.output(led_pin, GPIO.LOW)
                        led_on = False
                    set_angle(90)
                    motor_on = True

                elif distance <= 10 and motor_on:
                    if led_on:
                        GPIO.output(led_pin, GPIO.LOW)
                        led_on = False

                elif not motor_on:
                    if not led_on:
                        GPIO.output(led_pin, GPIO.HIGH)
                    led_on = True
                    
                     
           
            cv2.imshow('program', frame) # عرض الإطار امعالج
            
            if cv2.waitKey(1) == ord('q'): # الضغط على q لإنهاء البرنامج
                break
pwm.stop()
GPIO.cleanup()
