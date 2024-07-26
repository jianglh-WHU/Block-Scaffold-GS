function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

lod=0
iterations=30_000
resolution=-1
scale=1.0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --voxel_size) vsize="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --repair_setting) repair_setting="$2"; shift ;;
        --resolution) resolution="$2"; shift ;;
        --lora_ckpt) lora_ckpt="$2"; shift ;;
        --base_model_path) base_model_path="$2"; shift ;;
        --guidance_mode) guidance_mode="$2"; shift ;;
        --data_format) data_format="$2"; shift ;;
        --scale) scale="$2"; shift ;;
        --sparse) sparse="$2"; shift ;;
        --iterations) iterations="$2"; shift ;;
        --position_lr_init) position_lr_init="$2"; shift ;;
        --position_lr_final) position_lr_final="$2"; shift ;;
        --train_chunk) train_chunk="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

python train.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} \
    --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir} \
    --resolution ${resolution} --data_format ${data_format} --scale ${scale} 