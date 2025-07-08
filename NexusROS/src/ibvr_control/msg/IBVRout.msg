std_msgs/Header header

int32[]   ids             # agent IDs
float32[] error           # image‐area errors e_ij
float32[] distance        # predicted smoothed 3D distances D
float32[] depth           # filtered z_r
float32[] depth_est       # raw z_r_est

float32   cmd_vx
float32   cmd_vy
float32   cmd_wz