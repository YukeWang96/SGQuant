cd ../../
docker run --rm -it --init --runtime=nvidia --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 pyg /bin/bash