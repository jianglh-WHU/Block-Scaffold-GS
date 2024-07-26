exp_name='baseline'
voxel_size=0.001
update_init_factor=16
appearance_dim=32
ratio=1
gpu=-1
data_format='matrixcity'
scale=0.1
scene='matrixcity/small_city/street/pose/block_small/'

./train_matrixcity.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --data_format ${data_format} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} \
    --scale ${scale}