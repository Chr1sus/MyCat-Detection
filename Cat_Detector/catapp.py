from curses import COLOR_BLACK
import cv2
import tensorflow as tf
import numpy as np 
import csv 


tflite_model_path = "/usr/bin/modelcatgray.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']



# Load image
cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
cat_dict = {0: "Hormiguita", 1:"Jandi",2: "Koneko",3:"Lin",4:"Macho"}

    # start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('cat.xml')
    #notgray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (64, 64)), -1),0)
        input_data = cropped_img 
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data_tflite = interpreter.get_tensor(output_details[0]['index'])

        maxindex= int(np.argmax(output_data_tflite,axis=1))
        #cat_dict[maxindex] es la variable con la que se sabe cual gato es, que abra solo a Jandi y Koneko. Para los demas no se abre. 

        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()