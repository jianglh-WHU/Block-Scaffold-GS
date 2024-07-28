exp_name='baseline'
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=-1
data_format='hier'
scene='hierarchical-gs/small_city_eval/'
depths='depths'
alpha_masks=""

./train_hier.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --depths ${depths} --data_format ${data_format} \
    --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}
    