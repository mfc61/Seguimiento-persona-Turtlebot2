TRACKING DE UNA PERSONA CON EL TURTLEBOT2

El proyecto consiste en una secuencia de acciones que realizará el robot Turtlebot 2. En general, se identifica a una persona, en concreto Luis, y se realiza su seguimineto.

Primero, se reconocen todas las personas presentes en cada frame de la cámara del robot. 
Luego, se detectan y extraen las caras de las personas reconocidas anteriormente. 
Posteriormente, se identifica a Luis de entre las caras detectadas. En el caso de que se identifique, el robot debe seguirle. 
En caso contrario, debe continuar su búsqueda. Por último, una vez se ha identificado a Luis y se ha iniciado el seguimiento, si deja de detectarlo, el robot deberá volver a la base.
