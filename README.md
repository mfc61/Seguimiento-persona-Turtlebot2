# SEGUIMIENTO DE UNA PERSONA CON EL TURTLEBOT2

## DESCRIPCIÓN 
El proyecto consiste en una secuencia de acciones que realizará el robot Turtlebot 2. En general, se identifica a una persona, en concreto Luis, y se realiza su seguimiento. 

Primero, se reconocen todas las personas presentes en cada frame de la cámara del robot. 
Luego, se detectan y extraen las caras de las personas reconocidas anteriormente. 
Posteriormente, se identifica a Luis de entre las caras detectadas. En el caso de que se identifique, el robot debe seguirle. 
En caso contrario, debe continuar su búsqueda. Por último, una vez se ha identificado a Luis y se ha iniciado el seguimiento, si deja de detectarlo, el robot deberá volver a la base.

## REQUISITOS

- Robot Turtlebot2 con cámara y sensor láser, junto con sus librerías correspondientes.
- ROS
- OpenCV
- TensorFlow
- Archivo con los pesos de YOLOv3 'yolov3.weights'

## USO

1. Conectarse al Turtlebot
```bash
echo "ROS_MASTER_URI=http://LA_IP_DEL TURTLEBOT:11311" >> ~/.bashrc
echo "ROS_HOSTNAME=LA_IP_DE_TU_PC" >> ~/.bashrc
ssh turtlebot@ip_del_robot
```

2. Poner en marcha los diferentes motores y sensores del robot
```bash
# Base del robot
roslaunch turtlebot_bringup minimal.launch
# Sensor láser
roslaunch turtlebot_bringup hokuyo_ust10lx.launch
# Cámara
roslaunch astra_launch astra.launch
```

3. Lanzar el nodo de SLAM del Turtlebot
```bash
export TURTLEBOT_3D_SENSOR=astra
roslaunch turtlebot_navigation gmapping_demo.launch
```

4. Lanzar y configurar rviz para visualizar el mapa, el robot y sus sensores.
```bash
rosrun rviz rviz
```

5. Lanzar el nodo de seguimiento
```bash
rosrun tutlebot_tracking track_Luis.py
```

## IDENTIFICAR A OTRAS PERSONAS
Para este proyecto, se ha entrenado una red neuronal para lograr identificar a una persona en concreto, en este caso Luis.
Si se quisiera identificar a otra persona, se tendría que recopilar una gran cantidad de imágenes tanto de la persona a identificar como de otras personas, así como etiquetarlas debidamente.

En el caso del proyecto, se utilizaron los ficheros 'train_luis_adam.py', 'train_luis.py' y 'train_luis2.py', junto con el archivo con las imágenes etiquetadas 'train.txt', para la obtención del modelo 'modelo_luis.h5'.
