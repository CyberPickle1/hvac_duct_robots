import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt 
import math
from scipy.spatial.transform import Rotation as R 


class LPF(Node):
    def __init__(self):
        super().__init__('lpf_tuner')

        self.subscription_odom = self.create_subscription(Odometry,'/odom',self.odom_callback,10)
            # Low-pass filter parameters
        self.alpha = 0.05 # tune this number between 0.001 and 0.1  
        self.odom_xyz = {'x': 7.81, 'y': -1.1, 'z': 3.0}  # Assuming initial gravity

        # Initialize data lists 
        self.xl = []
        self.yl = []
        self.zl = []
        self.xxl = []
        self.yyl = []
        self.zzl = []
        self.th = []

    
        self.fig, self.ax = plt.subplots()
        self.accx, = self.ax.plot([],[],'r', markersize=1 , linestyle='dotted',label='x unfiltered')  # Red dotted line not filtered
        self.accy, = self.ax.plot([],[],'b',markersize=1, linestyle='dotted',label='y unfiltered')  # blue dottted line not filtered
        self.accz, = self.ax.plot([],[],'g',markersize=1, linestyle='dotted',label='z unfiltered')  # green dotted line not filtered
        self.fx, = self.ax.plot([],[],'r-',markersize=1,label='x filtered')  # Red line filtered
        self.fy, = self.ax.plot([],[],'b-',markersize=1,label='y filtered')  # blue line filtered
        self.fz, = self.ax.plot([],[],'g-',markersize=1,label='z filtered')  # green line filtered
        self.tth, = self.ax.plot([],[],'k',markersize=1,label='theta unfiltered')  #black line unfiltered

        self.ax.set_xlim(-25, 2500)
        self.ax.set_ylim(-25, 25)
        plt.ion()  # Interactive mode
        plt.legend()

        self.odom_offset_x = 0.0 
        self.odom_offset_y =  -0.05

    def odom_callback(self, msg): 
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        r = R.from_quat([qx, qy, qz, qw])
        rpy = r.as_euler('zyx')
        self.theta_current = rpy[0]

        self.th.append(self.theta_current)

        self.tth.set_data(range(len(self.th)),self.th)



        x = msg.pose.pose.position.x - self.odom_offset_x * math.cos(self.theta_current) - self.odom_offset_y * math.sin(self.theta_current)
        y = msg.pose.pose.position.y - self.odom_offset_x * math.sin(self.theta_current) + self.odom_offset_y * math.cos(self.theta_current)
        z = msg.pose.pose.position.z

        print(f"x : {x} , y : {y} , z : {z}" ) 

        self.xl.append(x)
        self.yl.append(y)
        self.zl.append(z)

        # Plotting unfiltered data
        self.accx.set_data(range(len(self.xl)),self.xl)  #range(len()) is used to give the input as a list as the function expects 2 lists and not an integer and list 
        self.accy.set_data(range(len(self.yl)),self.yl)
        self.accz.set_data(range(len(self.zl)),self.zl)


        self.odom_xyz['x'] = self.alpha*x + (1-self.alpha)*self.odom_xyz['x']
        self.odom_xyz['y'] = self.alpha*y + (1-self.alpha)*self.odom_xyz['y']
        self.odom_xyz['z'] = self.alpha*z + (1-self.alpha)*self.odom_xyz['z']

        # Appending values 
        self.xxl.append(self.odom_xyz['x'])
        self.yyl.append(self.odom_xyz['y'])
        self.zzl.append(self.odom_xyz['z'])

        # Plotting filtered data
        self.fx.set_data(range(len(self.xxl)),self.xxl)
        self.fy.set_data(range(len(self.yyl)),self.yyl)
        self.fz.set_data(range(len(self.zzl)),self.zzl)

        # Update plot
        plt.draw()
        plt.pause(0.01)


        self.x = self.odom_xyz['x']
        self.y = self.odom_xyz['y']
        self.z = self.odom_xyz['z']


def main(args=None):
    rclpy.init(args=args)
    node = LPF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    plt.show()  # Keep plot open after shutdown


if __name__ == '__main__':
    main()