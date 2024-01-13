#!/usr/bin/env python

import cv2, rospy
import numpy as np
import smach_ros
from smach import State,StateMachine
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tensorflow.keras.models import load_model


# Variable para saber si se ha identificado a Luis
luis = False 
# Coordenadas de Luis para su seguimiento
x_luis = 0
y_luis = 0
w_luis = 0
h_luis = 0



# ESTADO DetectLuis: RECONOCE PERSONAS, DETECTA CARAS E IDENTIFICA A LUIS
class DectectLuis(State):
    def __init__(self):
        State.__init__(self, outcomes=['luis_detected'])

        # Variable que indica si se ha identificado a Luis
        self.luis_detected = False

        # Rutas a los archivos de configuracion y pesos de YOLO
        self.yolo_cfg = 'yolov3.cfg'
        self.yolo_weights = 'yolov3.weights'
        # Cargar el modelo YOLO
        self.net = cv2.dnn.readNetFromDarknet(self.yolo_cfg, self.yolo_weights)
        # Inicializar el puente entre OpenCV y ROS
        self.bridge = CvBridge()

        # Cargar el clasificador de caras preentrenado
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Publicador para la imagen con las detecciones
        self.image_pub = rospy.Publisher("person_tracking/image", Image, queue_size=1)

        # Subscribirse al topic de la imagen de la camara del robot
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)


    def execute(self, userdata):
        # Bucle de espera hasta que se identifique a Luis
        rate = rospy.Rate(10)
        while not self.luis_detected:
            rate.sleep()

        return 'luis_detected'


    def luis_model(img, modelo_path="clasificador_adam.pkl"):
        # Cargar el modelo preentrenado
        loaded_model = load_model(modelo_path)

        # Preprocesar la imagen
        img_array = cv2.resize(img, (224, 224))
        img_array = img_array.astype('float') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar la prediccion
        prediccion = loaded_model.predict(img_array)

        # Devolver True si la prediccion es 1 (es Luis), False si la prediccion es 0 (no es Luis)
        return bool(round(prediccion[0][0]))


    def image_callback(self, msg):
        global luis,x_luis,y_luis,w_luis,h_luis

        # Si no se ha identificado a Luis
        if not luis:
            # Convertir la imagen de ROS a un formato que OpenCV pueda manejar
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Obtener las dimensiones del frame
            height, width = frame.shape[:2]
            # Preprocesar el frame
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            # Obtener las capas de salida
            output_layers = self.net.getUnconnectedOutLayersNames()
            # Realizar las detecciones
            detections = self.net.forward(output_layers)

            # Para cada deteccion
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Si el objeto detectado supera un grado de confianza
                    if confidence > 0.95 and class_id == 0:  # 0 == person
                        # Dibujar un cuadro alrededor de la persona detectada
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)

                        if not luis:
                            # Nueva imagen solo con la persona detectada
                            person_rectangle = frame[int(y):int(y + h), int(x):int(x + w)].copy()
                            # Detectar caras en la imagen
                            faces = self.face_cascade.detectMultiScale(person_rectangle, scaleFactor=1.3, minNeighbors=5)
                            
                            # Para cada cara detectada
                            for (x_f, y_f, w_f, h_f) in faces:
                                x_face = int(x + x_f)
                                y_face = int(y + y_f)
                                w_face = int(w_f)
                                h_face = int(h_f)
                                # Dibujar rectangulo alrededor de la cara detectada
                                cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (255, 0, 0), 2)
                                # Nueva imagen solo con la cara detectada
                                cara = person_rectangle[int(y_f):int(y_f + h_f), int(x_f):int(x_f + w_f)].copy()

                                # Comprobar si la cara detectada se corresponde con Luis
                                #if self.luis_model(cara):
                                luis = True
                                # Guardar las coordendas del recuadro de Luis para el seguimiento
                                x_luis = x
                                y_luis = y
                                w_luis = w
                                h_luis = h

            # Publicar la imagen con las detecciones en el topic correspondiente
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            
            if luis:
                self.luis_detected = True

    

# ESTADO TrackLuis: EL ROBOT REALIZA EL SEGUIMIENTO DE LUIS
class TrackLuis(State):
    def __init__(self):
        State.__init__(self, outcomes=['base'])
        # Variable para saber si se ha perdido a Luis
        self.luis_lost = False
        # Tracker para realizar el seguimiento
        self.tracker = None
        # Publicador para la velocidad del robot
        self.pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        # Publicador para la imagen con las detecciones
        self.image_pub = rospy.Publisher("person_tracking/image", Image, queue_size=1)
        # Subscribirse al topic de la imagen de la camara del robot
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # Inicializar el puente entre OpenCV y ROS
        self.bridge = CvBridge()
        self.contador = 0
        # Rutas a los archivos de configuracion y pesos de YOLO
        self.yolo_cfg = 'yolov3.cfg'
        self.yolo_weights = 'yolov3.weights'
        # Cargar el modelo YOLO
        self.net = cv2.dnn.readNetFromDarknet(self.yolo_cfg, self.yolo_weights)


    def execute(self, userdata):
        # Bucle de espera hasta que se pierda a Luis
        rate = rospy.Rate(10)
        while not self.luis_lost:
            rate.sleep()

        return 'base'


    def image_callback(self, msg):
        global x_luis,y_luis,w_luis,h_luis
        persona = True

        # Si se ha identificado a Luis
        if luis:
            # Convertir la imagen de ROS a un formato que OpenCV pueda manejar
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if self.contador % 20 == 0:     # Solo se realiza el reconocimiento cada ciertas iteraciones
                # Preprocesar el frame
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.net.setInput(blob)
                # Obtener las capas de salida
                output_layers = self.net.getUnconnectedOutLayersNames()
                # Realizar las detecciones
                detections = self.net.forward(output_layers)
                persona = False     # Variable para saber si hay personas en el frame
                # Para cada deteccion
                for detection in detections:
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # Si el objeto detectado supera un grado de confianza
                        if confidence > 0.95 and class_id == 0:  # 0 == person
                            persona = True

            # Crear e inicializar el tracker si no se ha hecho ya
            if not self.tracker:
                # Recuadro con las coordenadas de Luis
                bbox = (int(x_luis), int(y_luis), int(w_luis), int(h_luis))
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, bbox)
            else:
                self.contador = self.contador + 1
                # Realizar el seguimiento con el tracker
                ok, bbox = self.tracker.update(frame)

                # Comprobar si se ha perdido a Luis
                if not persona:
                    self.luis_lost = True
                else:   # Control de velocidad en funcion del nuevo bbox
                    # Dibujar el rectangulo con las nuevas coordenadas en el frame
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                    # Publicar la imagen con las detecciones en el topic correspondiente
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

                    twist = Twist()
                    # Ajustar la velocidad lineal
                    diff_h = h_luis - bbox[3]              
                    twist.linear.x = 0.01 * diff_h          
                    # Obtener el centro x del rectangulo
                    center_x = bbox[0] + bbox[2] / 2
                    # Calcular el error en pixeles desde el centro de la imagen
                    error = center_x - (640 / 2)  # 640 es el ancho tipico de la imagen
                    # Ajustar la velocidad angular en funcion del error
                    twist.angular.z = -float(error) * 0.01

                    # Comprobar margenes de seguridad
                    if twist.linear.x > 0.2:
                        twist.linear.x = 0.2
                    elif twist.linear.x < -0.2:
                        twist.linear.x = -0.2

                    if twist.angular.z > 0.3:
                        twist.angular.z = 0.3
                    elif twist.angular.z < -0.3:
                        twist.angular.z = -0.3
                    
                    # Publicar la velocidad del robot
                    self.pub.publish(twist)



# ESTADO MoveToBase: EL ROBOT REGRESA A SU POSICION INICIAL
class MoveToBase(State):
    # Metodo de inicializacion del estado
    def __init__(self):
        State.__init__(self, outcomes=['end']) # Se define la transicion 'end'
    
    # Metodo que define el comportamiento del estado, mueve el robot a la posicion base
    def execute(self, userdata):
        # Definicion de la posicion objetivo
        goal_pose = MoveBaseGoal()
        goal_pose.target_pose.header.frame_id = 'map'
        goal_pose.target_pose.pose.position.x = 0  
        goal_pose.target_pose.pose.position.y = 0  
        goal_pose.target_pose.pose.orientation.w = 1.0 

        # Moverse a coordenadas especificas usando una SimpleActionState con la accion MoveBase
        move_base_state = smach_ros.SimpleActionState('move_base', MoveBaseAction, goal=goal_pose)
        outcome = move_base_state.execute(userdata)

        # Si la accion se ha realizado con exito, se cambia de estado
        if outcome == 'succeeded':
            return 'end'



if __name__ == '__main__':
    rospy.init_node("person_tracking_node",anonymous=True)
    sm = StateMachine(outcomes=['stop'])
    with sm:
        StateMachine.add('DETECT_LUIS', DectectLuis(), transitions={'luis_detected':'TRACK_LUIS'})
        StateMachine.add('TRACK_LUIS', TrackLuis(), transitions={'base':'MOVE_BASE'})
        StateMachine.add('MOVE_BASE', MoveToBase(), transitions={'end':'stop'})
    sm.execute()
    rospy.spin()