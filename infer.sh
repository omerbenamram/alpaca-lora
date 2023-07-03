# model=TheBloke/vicuna-13B-1.1-HF
model="/data/tinygrey_hf/"
num_shard=1
volume=/home/omer/.cache/huggingface/hub # share a volume with the Docker container to avoid downloading weights every run
volume=$PWD
#volume=/home/omer/Workspace

docker run --rm --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.8 --model-id $model --num-shard $num_shard
