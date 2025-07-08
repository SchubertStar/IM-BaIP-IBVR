/*
* File: distance_rplidar.cpp
* Purpose: ros rplidar listener node.
* Version: 1.0.0
* File Date: 12-07-2023
* Release Date: 14-07-2023
* URL: ...
* License: MIT License
* Copyright (c) Muhammad Ridho Rosa 2023
* Permission is not granted to any person obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
* modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
* WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <cmath>

#define QUEUE_SIZE 1000 //subscriber buffer size
#define RAD2DEG(x) ((x)*180./M_PI)


class DistanceRplidar{
public:
  DistanceRplidar();
private:
  void callBack(const sensor_msgs::LaserScan::ConstPtr& scan);

  ros::NodeHandle n;
  ros::Publisher pub;
  ros::Subscriber sub_lidar;
  ros::Subscriber sub_odom;

  int i_velLinear , i_velAngular;
  int range;
  int count;
  int count_inrange;
  float distance_array[1150];
  float degree_array[1150];
  float degree_object[5];
  float distance_object[5];
  int count_object;
  int count_size;
  double u1x,u1y;
  float e1,e2;
  float dist_1,dist_2;
  float desired_1 = 1;
  float desired_2 = 1;
  int m = 0;
  float accumulator_distance, accumulator_degree;
  float x1,x3,y1,y3,z1x,z1y,z2x,z2y;
  double pi = 3.14159265359;
};

DistanceRplidar::DistanceRplidar() // global variable:
{
  i_velLinear = 1;
	i_velAngular = 0;
	n.param("axis_linear", i_velLinear, i_velLinear);
	n.param("axis_angular", i_velAngular, i_velAngular);
	pub = n.advertise<geometry_msgs::Twist>("cmd_vel2", 1);
	sub_lidar = n.subscribe<sensor_msgs::LaserScan>("scan", QUEUE_SIZE, &DistanceRplidar::callBack, this);
  // odom_pub_ = n.subscribe<>
}


void DistanceRplidar::callBack(const sensor_msgs::LaserScan::ConstPtr& scan)
{
  geometry_msgs::Twist vel;
  count = scan->scan_time / scan->time_increment;
  count_inrange = 0;
  for(int i = 0; i < count; i++)
  {
    if(scan->ranges[i] < 1.25)
    {
      distance_array[count_inrange] = scan->ranges[i];
      degree_array[count_inrange] = RAD2DEG(scan->angle_min + scan->angle_increment * i);
      count_inrange++;
    }
  }
  //ROS_INFO("count inrange: %i",count_inrange);
  if(count_inrange > 0)
  {
    count_object = 0;
    accumulator_distance = 0;
    accumulator_degree = 0;
    count_size = 0;
    for(int j = 0; j < count_inrange; j++)
    {
      float abs_val = abs(degree_array[j] - degree_array[j+1]);
      //ROS_INFO("abs value: %f",abs_val);
      if ( (abs_val > 25) || (j == count_inrange -1) )
      {
        if ((count_size > 10) && (count_size < 100) && (accumulator_distance >0) )
        {
         	//ROS_INFO(" count size: %i",count_size);
          	degree_object[count_object] = accumulator_degree/count_size;
          	distance_object[count_object] = accumulator_distance/count_size;
          	count_object++;
        }
	count_size = 0;
        accumulator_distance = 0;
        accumulator_degree = 0;
      }
      else
      {
        if ( abs (distance_array[j] - distance_array[j-1]) < 0.03)
        {
		accumulator_distance += distance_array[j];
        	accumulator_degree += degree_array[j];
        	count_size++;
	}
      }
    }
  //ROS_INFO(" count object!!: %i",count_object);
  }

  // u = - B Dz Dz^T e
  // B = [1 0 -1;
  //     -1 1 0;
  //      0 -1 1]
  // u1 = -z1/||z1|| e1 + z3/||z3|| e3
  // u2 = -z2/||z2|| e2 + z1/||z1|| e1
  // u3 = -z3/||z3|| e3 + z2/||z2|| e2

  // now calculate z = bar(B)^T p
  // z1 = [(x1 - x2) ; (y1 - y2)]
  // z2 = [(x2 - x3) ; (y2 - y3)]
  // z3 = [(x3 - x1) ; (y3 - y1)]
  if (count_object > 1)
  {
	for (int k = 0; k < count_object; k++)
  	{
		if ( (degree_object[k] < 70) && (degree_object[k] > 0) )
		{
			y1 = distance_object[k] * cos(degree_object[k]*(pi/180));
			x1 = -distance_object[k] * sin(degree_object[k]*(pi/180));
		}
                if ( (degree_object[k] > 70) && (degree_object[k] < 150) )
                {
                	y3 = distance_object[k] * cos(degree_object[k]*(pi/180));
                	x3 = -distance_object[k] * sin(degree_object[k]*(pi/180));
                }
  	}
  	//z2x = 0 - x3;
  	//z2y = 0 - y3;
  	//z1x = x1 - 0;
  	//z1y = y1 - 0;
	z2x = -x3;
        z2y = -y3;
        z1x = x1;
        z1y = y1;

	dist_1 = sqrt( (pow(z1x,2)) + (pow(z1y,2)) );
        dist_2 = sqrt( (pow(z2x,2)) + (pow(z2y,2)) );
  	e1 = dist_1 - desired_1;
  	e2 = dist_2 - desired_2;
  	u1x = -(z2x/dist_2)*e2 + (z1x/dist_1)*e1;
  	u1y = -(z2y/dist_2)*e2 + (z1y/dist_1)*e1;

  	if (isnan(u1x)) { vel.linear.x = 0;
	}
  	else { vel.linear.x = 0.75*u1y;
  	}

  	if (isnan(u1y)) { vel.linear.y = 0;
  	}
  	else { vel.linear.y = 0.75*u1x;
  	}
  	vel.angular.z = i_velAngular;
  }

  // DEBUG
  /*for (int k = 0; k < count_object; k++)
  {
    ROS_INFO(" object degree: %f",degree_object[k]);
    ROS_INFO(" object distance: %f",distance_object[k]);
  }
  ROS_INFO(" x1,y1: [%f,%f]",x1,y1);
  ROS_INFO(" x3,y3: [%f,%f]",x3,y3);
  ROS_INFO(" velocity x: %f",vel.linear.x);
  ROS_INFO(" velocity y: %f",vel.linear.y);
  ROS_INFO(" dist_1: %f",dist_1);
  ROS_INFO(" dist_2: %f",dist_2);
  ROS_INFO(" e1: %f",e1);
  ROS_INFO(" e2: %f",e2);
  ROS_INFO(" z1x: %f",z1y);
  ROS_INFO(" z1y: %f",z1x);
  ROS_INFO(" z2x: %f",z2y);
  ROS_INFO(" z2y: %f",z2x);*/

  if (m ==  0)
  {
	ros::Duration(10).sleep();
	pub.publish(vel);
	m = 1;
  }
  else
  {
	pub.publish(vel);
  }

  // control ridho
  // u = P + (R-Rd)Dm P - B Dz Dz^T Dk e  ; e from lidar, Dm P from wheel_vel
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "distance_rplidar");
	DistanceRplidar teleop_robot2;
	ros::spin();
}
