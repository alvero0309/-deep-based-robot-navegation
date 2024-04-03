
import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
#Import Hokuyo Laser Scan
from sensor_msgs.msg import LaserScan
# Nombre del archivo de texto
archivo_texto = "path.txt"

#Meta para llegar hacia la meta
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.4
TIME_DELTA = 1
angular_speed = 1
linear_speed = .7
#/home/milena/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/xacro/laser
rangeMaxOfLaser = 2

# Check if the random goal position is located out of the scenerario, this point is the door of car
def check_pos(x, y):
    goal_ok = True
    #Area of arena
    if -2.175> x > 2.175 and  -2.300 > y > 2.300:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        #velodyne crea un arreglo de unos y los multiplica por 10
        self.velodyne_data = np.ones(self.environment_dim) * 10
        #Laser data
        self.laser_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        self.distanceGoalAct = 0
        self.distancePre = 0
        self.timeStep=0
        self.timeStepPre=0
        self.point_Interest = []
        self.pointOfinterest = []

        self.distanceOrig = 0
        self.newStep=0
        #Angulos -90° a 90°
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        #Reset the world, the object allocate in the room in the initial state
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.publisher4 = rospy.Publisher("point_of_interest",MarkerArray)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )
        self.laserHokuyo = rospy.Subscriber(
            "/r1/front_laser/scan", LaserScan, self.laserScan_callback, queue_size=1
            )
        self.distanceGoalAct = .0001
        self.distanceOrig  = .0001
       

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        #print(data)
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        #Distancia si detecta algún obstaculo, espacio de estados depende del tamaño del entorno
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    #Recolección de puntos de interes

    def laserScan_callback (self,scan_data):

        # Ejemplo de cómo acceder a los datos del escáner láser
        ranges = scan_data.ranges  # Lista de distancias medidas por el escáner láser
        
        #Estos parametros estan definidos por el hardware del hokuyo
        angle_increment = scan_data.angle_increment  # Incremento angular entre cada punto de escaneo
       
        angle_min = scan_data.angle_min  # Ángulo mínimo del campo de visión del escáner láser

        # Tamaño de la lista de distancias del escáner láser
        num_ranges = len(ranges)
        #The num of ranges is in this configuration of 360 sizes 
   


        # Crear un arreglo para almacenar los datos del escáner láser procesados
        self.laser_data= np.ones(self.environment_dim) * 5  # Aquí define environment_dim según tus necesidades

        
     
        #Recorre cada elemento de la la lista
        for i in range(num_ranges):
            
            # Calcular el ángulo actual del punto de escaneo
            #print(i)
            angle = angle_min + i * angle_increment
            
            # Filtrar puntos de escaneo no deseados (opcional)
            #if ranges[i] < scan_data.range_min or ranges[i] > scan_data.range_max:
            #   continue  # Saltar al siguiente punto de escaneo si está fuera del rango válido


            if ranges[i] == float('inf'):
                #Alcance maximo esta definido por el harware
                #Estas coordenadas estan definidas con respecto al robot, es necesario 
                #hacer una transpolación a las coordenadas del mapa
               
                x = rangeMaxOfLaser * math.cos(angle)
                y = rangeMaxOfLaser * math.sin(angle)
                z = 0
                
                #Obtener coordenas del robot con respecto al mapa
                #Las cooredenadas del robot es odom_x y odomo_y

                #Los puntos de interes estan dados en cordenadas del mapa de la siguiente forma:
                x_map = x + self.odom_x
                y_map = y + self.odom_y
                coords = (x_map,y_map)


                self.point_Interest.append(coords)

            #Capturar los huecos por donde pueda pasar el robot    
            #if ranges[i] 1





            else:                

                
                # Calcular la posición relativa del punto de escaneo en coordenadas cartesianas con respecto al robot
                x = ranges[i] * math.cos(angle)
                y = ranges[i] * math.sin(angle)
                z = 0  # Suponiendo que el escáner láser está en el plano xy (horizontal)





                if z > -0.2:  # Filtrar puntos que están por debajo de una cierta altura
                    dot = x * 1 + y * 0
                    mag1 = math.sqrt(x ** 2 + y ** 2)
                    mag2 = math.sqrt(1 ** 2 + 0 ** 2)
                    beta = math.acos(dot / (mag1 * mag2)) * np.sign(y)
                    dist = math.sqrt(x ** 2 + y ** 2 + z ** 2)

                    for j in range(len(self.gaps)):
                        if self.gaps[j][0] <= beta < self.gaps[j][1]:
                            # Actualizar el arreglo de datos del escáner láser
                            self.laser_data[j] = min(self.laser_data[j], dist)
                            break  # Salir del bucle si se encuentra un intervalo adecuado
        #print(self.point_Interest)

        #Pintar puntos de interes en el mapa
        self.publish_markersPI(self.point_Interest)                    
        # Devolver los datos procesados del escáner láser
        # return velodyne_data




    def odom_callback(self, od_data):
        self.last_odom = od_data



    def moveForward(self,linear_speed):
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_speed
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
        action=[linear_speed,0.0]
        self.publish_markers(action)
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)
        # Detener el movimiento
        self.stop_movement()
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        return action

    def turn_right(self,angular_speed):
        rate = rospy.Rate(10)  # 10Hz
        initial_angular_speed = 0.0  # Velocidad angular inicial (en rad/s)
        final_angular_speed = angular_speed  # Velocidad angular final (en rad/s)
        turn_duration = 2.0  # Duración del giro (en segundos)
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0  # No avanzar
        #vel_cmd.angular.z = -angular_speed  # Velocidad angular negativa para girar hacia la derecha
        
        start_time = rospy.get_time()
        # Tiempo necesario para girar 90 grados (ajustar según la velocidad angular)
        turn_duration = 2.0
        """
        self.vel_pub.publish(vel_cmd)
        action=[0.0,-angular_speed]
        self.publish_markers(action)
        """
        action=[0.0,angular_speed]
        while rospy.get_time() - start_time < turn_duration:
            
            elapsed_time = rospy.get_time() - start_time

            rospy.wait_for_service("/gazebo/unpause_physics")
            try:
                self.unpause()
            except (rospy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

            # Interpolación de la velocidad angular entre la inicial y la final
            interpolation_ratio = min(elapsed_time / turn_duration, 1.0)
            interpolated_speed = (1 - interpolation_ratio) * initial_angular_speed + interpolation_ratio * final_angular_speed
            vel_cmd.angular.z = interpolated_speed  # Velocidad angular negativa para girar hacia la derecha
            
            self.vel_pub.publish(vel_cmd)
             
            # propagate state for TIME_DELTA seconds
            time.sleep(1.11)

            # Detener el movimiento
            self.stop_movement()
            
            rospy.wait_for_service("/gazebo/pause_physics")
            try:
                pass
                self.pause()
            except (rospy.ServiceException) as e:
                print("/gazebo/pause_physics service call failed")
           

        return action
     
    def stop_movement(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
        # propagate state for TIME_DELTA seconds
       

    def turn_left(self,angular_speed):
        rate = rospy.Rate(10)  # 10Hz
        initial_angular_speed = 0.0  # Velocidad angular inicial (en rad/s)
        final_angular_speed = angular_speed # Velocidad angular final (en rad/s)
        
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0  # No avanzar
        #vel_cmd.angular.z = -angular_speed  # Velocidad angular negativa para girar hacia la derecha
        
        start_time = rospy.get_time()
        # Tiempo necesario para girar 90 grados (ajustar según la velocidad angular)
        turn_duration = 2.0
        """
        self.vel_pub.publish(vel_cmd)
        action=[0.0,-angular_speed]
        self.publish_markers(action)
        """
        action=[0.0,angular_speed]

        while rospy.get_time() - start_time < turn_duration:
            
            elapsed_time = rospy.get_time() - start_time

            rospy.wait_for_service("/gazebo/unpause_physics")
            try:
                self.unpause()
            except (rospy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

            # Interpolación de la velocidad angular entre la inicial y la final
            interpolation_ratio = min(elapsed_time / turn_duration, 1.0)
            interpolated_speed = (1 - interpolation_ratio) * initial_angular_speed + interpolation_ratio * final_angular_speed
           
            vel_cmd.angular.z = -interpolated_speed  # Velocidad angular negativa para girar hacia la derecha
            
            self.vel_pub.publish(vel_cmd)
            # propagate state for TIME_DELTA seconds
            time.sleep(1.17)
            # Detener el movimiento
            self.stop_movement()
            # Detener el movimiento
            self.stop_movement()
            
            rospy.wait_for_service("/gazebo/pause_physics")
            try:
                pass
                self.pause()
            except (rospy.ServiceException) as e:
                print("/gazebo/pause_physics service call failed")
           

                """
                while not rospy.is_shutdown() and time.time() - start_time < turn_duration:
                   self.vel_pub.publish(vel_cmd)
                   self.publish_markers(action)"""
        return action


        """
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0  # No avanzar
        vel_cmd.angular.z = angular_speed  # Velocidad angular negativa para girar hacia la derecha
    
        self.vel_pub.publish(vel_cmd)
        action=[0.0,angular_speed]
        self.publish_markers(action)
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        return action"""



    # Perform an action and read a new state
    def step(self, actionD):
        target = False

        if actionD == 0:
           action = self.moveForward(linear_speed)
        if actionD == 1:
           action = self.turn_right(angular_speed)
        if actionD == 2:
           action = self.turn_left(angular_speed)

       


        #If see a collision the next epsodie starts

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        print("laser_data: ")
        print(self.laser_data)
        laser_state = [v_state]
        #print(laser_state)
        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y

       
        #Las cooredenadas almacenadas estan dadas sobre el escenario
        with open(archivo_texto, "a") as archivo:
             # Escribe el estado de las variables en el archivo
             archivo.write("{}".format(self.odom_x))
             archivo.write("{} \n".format(self.odom_y))





        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)


        #Aulocalization 

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        if .0001 == self.distanceGoalAct:
            self.distanceOrig = distance

        self.distancePre =self.distanceGoalAct
        self.distanceGoalAct = distance
        
      
        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        #Agregar un timeStep
        self.timeStepPre = self.timeStep
        self.timeStep = self.timeStep + 1

        robot_state = [distance, theta]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser,self.distanceGoalAct,self.distancePre,self.distanceOrig,self.timeStep,self.timeStepPre)
        #print(reward)
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
             self.reset_proxy()
             print("Prueba")

        except rospy.ServiceException as e:
             print("/gazebo/reset_simulation service call failed")
        #El angulo del robot cuando inicia su exploración esta dado por un muestreo aleatorio
        #angle = np.random.uniform(-np.pi, np.pi)
        angle = 0
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        #rESET THE ROBOT POSITION
        while not position_ok:
            x = np.random.uniform(-2.0, 1.8)
            y = np.random.uniform(-2.0, 1.8)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        #self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        #The laser scanner is a status provider.
        v_state[:] = self.laser_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        #Reset
        self.distanceGoalAct = .0001
        self.timeStep = 0
        #Se modifico la los ultimos estados  que indicaban la acción escogida
        robot_state = [distance, theta]
        state = np.append(laser_state, robot_state)
        return state

    
     




    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 6:
            self.upper += 0.004
        if self.lower > -6:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)


    #Intenta localizar los box para cambiarlos agregar un carrito en movimiento
    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)


        #Publica el arreglo de la odemetria
        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    def publish_markersPI(self,positions):

             # Publish visual data in Rviz
        markerArray = MarkerArray()
        i=0
        for position in (positions):
            i=i+1
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = 0
            marker.id=i
            markerArray.markers.append(marker)
        self.publisher4.publish(markerArray)
        
    #observa si existe una collision
    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        #print(min_laser)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

  

#Falta modelar las recompensas
    @staticmethod
    def get_reward(target, collision, action, min_laser,distanceActualy,distancePrev,distanceOrig,timeStep,timeStepPre):
        if target:
            
            return 100.0
        elif collision:
            
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0

            #Penaliza la proximidad de los obstaculos
            # Recompensa por disminución de la distancia con respecto a la meta
            #print(distanceActualy)
            #print(distancePrev)
            
            distance_reward = (distanceActualy - distancePrev)# 
            print("Distance Original "+ str(distanceOrig))
            print("Distance Actualy " + str(distanceActualy))

             #Por cada time Step se dvuelve una recompensa negativa de .02
            #print("Distancia: " + str(float(distance_reward)))
            if timeStepPre < timeStep:
                #Tasa de crecieminto
                b = .01
                rewTime =  -np.exp(b * timeStep) + 1
           
            #Recompensa Positiva devuelve un valor mientras se encuentra dentro de la recompensa positiva
            if distanceActualy < distanceOrig and distance_reward < 0:
                X = distanceOrig - distanceActualy
                #Función escalada entre 0 y 1, invertida
                rew =  np.minimum(np.maximum(0, X), distanceOrig) 
                print("Distancia anteriot " + str(distance_reward))
                print("RELU " + str(rew))
                print("RewTime " + str(rewTime))
                positiva = rewTime + rew  + (.2) - r3(min_laser) 
                print("Recompensa Positiva :" + str(positiva))
                return  positiva


                 #Devuelve un valor Negativo si se aleja del punto mas allá del punto de inicio
            if distanceActualy >= distanceOrig:
                X = distanceOrig - distanceActualy
                rew = (1 - (1 / (1 + np.exp(-(X + 5))))) * -1
                print("Negativa:" + str(rew))
                return  rew + rewTime  - r3(min_laser)
               
            elif distance_reward > 0:
                print("Negativa fuera del origen " + str(r3(min_laser)))
                return  rewTime  -(.1) - r3(min_laser)

            
            elif distance_reward < 0:
                print("Positiva fuera del origen" + str(r3(min_laser)))
                #rint(rewTime)
                return  rewTime + (.1) - r3(min_laser) 

            
            


   
            
