# tf-best-practice

## Running on AWS

1. Select `nvidia-docker-171107 (ami-f8da7781)` as AMI and start a `p2.xlarge` (spot) instance.

2. Login to instance:
```bash
sudo service docker start
sudo usermod -a -G docker ubuntu
sudo reboot
```

3. Build docker from github
```bash
docker build https://github.com/hanxiao/tf-best-practice.git#code-gen
```

4. Get docker image ID and run with `nvidia-docker`:
```bash
docker images
nvidia-docker run -it --entrypoint /bin/bash ed73b939d215
```
Just do `sudo reboot` if you encounter `nvidia-docker | 2018/02/08 20:52:07 Error: nvml: Driver/library version mismatch` problem!

