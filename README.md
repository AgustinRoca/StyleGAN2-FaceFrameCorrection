# Thesis - Face generator app 2021S2
Face generator project using StyleGAN2. This is a continuation of another project that already
implements an API and Webapp to generate random faces using StyleGAN2.
This particular project aims to find directions in the latent space of the StyleGAN2 network
to allow morphing of generated faces.

## Setup
### 0. SSH connection
In order to connect to the server at ITBA, we need to configure a `ProxyJump` as it's private and only
available through another publicly disclosed server. This internal server is running Linux 18.04LTS with 
an NVIDIA Quadro RTX 8000.

```
#~/.ssh/config

Host pampero
    Hostname pampero.it.itba.edu.ar
    User <USER>
Host quadro
    Hostname quadro-deepmind.it.itba.edu.ar
    ProxyJump pampero
    User <USER>
    ForwardX11 yes
    Compression yes
```

This way, when executing `ssh quadro` a connection to this internal server will be made.

### 1. CUDA and CUDNN (non root)
This project uses `tensorflow-gpu==1.14`, which depends on `CUDA v10.0` and `CUDNN v7.4`, which are needed to execute code
on NVIDIA's GPUs. For more information about the compatible CUDA and CUDNN versions for each version of tensorflow, please
have a look [here](https://www.tensorflow.org/install/source#gpu).

#### CUDA v10.0
1. Head to the [CUDA v10.0 download page at NVIDIA](https://developer.nvidia.com/cuda-10.0-download-archive) from your local browser. If it isn't available, you can search it [here](https://developer.nvidia.com/cuda-toolkit-archive).
2. Select the version options depending OS.
3. Select the `runfile (local)` version. Copy the download link for `Base Installer` and download it in the destination machine running `wget <LINK>`.
4. Give execution permission to the downloaded file `chmod +x <CUDA_INSTALLER>` and run it `./<CUDA_INSTALLER>`.
5. Accept the `EULA` pressing `s` until reaching the end and type `accept`.
6. Type `no` to **NOT** install graphics drivers bundled with CUDA.
7. Type `yes` to install CUDA libraries, which is what we need. 
8. You are going to be asked for an installation directory. If no one is provided, a system-wide location will be used which requires root access. To install it without root permissions, give it a directory inside your home. We will refer it to `<CUDA>`.
9. Type `no` to **NOT** install CUDA samples, unless you want them.

#### CUDNN v7.4.1 (CUDA v10.0)
1. Head to the [CUDNN download page](https://developer.nvidia.com/rdp/cudnn-download) from your local browser.
2. Login or signup as an NVIDIA Developer.
3. Accept the EULA and go over to `Archived versions of CUDNN`.
4. Search for `cuDNN v7.4.1, for CUDA 10.0` and click on it.
5. Download `cuDNN Library for Linux`. As this link is authenticated, you need to download it using your browser and then upload it to the destination machine using, for example, `scp <LOCAL_INSTALLER_PATH> quadro:<CUDNN_INSTALLER>`.
6. In your destination machine, extract the archive `tar -xvzf <CUDNN_INSTALLER>`. This will extract the libraries in a path we will call `<CUDNN>`.

#### Link CUDA and CUDNN
To finish up the installation, it's necessary to copy some files and add the libraries to different env vars to make them visible for other programs.
```bash
cp -P <CUDNN>/cuda/include/cudnn.h <CUDA>/include/
cp -P <CUDNN>/cuda/lib64/libcudnn* <CUDA>/lib64
chmod a+r <CUDA>/include/cudnn.h <CUDA>/lib64/libcudnn*
```

Also you need to include this paths to your global variables. You can do this by adding this lines at the end of your `~/.bashrc`
```
export PATH=<CUDA>/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<CUDA>/lib64/
```

After this, you will have to run `source ~/.bashrc`

### 2. Webapp and API
Follow the setup sections of each of these:

- [Webapp](./webapp/README.md#Setup).
- [API](./api/README.md#Setup).

### 3. Test
```python
# Inside python terminal

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

**NOTE**: Some warnings may appear, discard them.

Search for somthing named `GPU` (don't get confused with `XLA_GPU`, they are two different things). If it appears then you are ready to go!

## Run
Follow the Run sections of each of these:

- [Webapp](./webapp/README.md#Run)
- [API](./api/README.md#Run).

It may be useful to set a SOCKS5 proxy to redirect traffic to the server. To do this, run
```bash
ssh -ND7070 quadro
```

This will create a SOCKS5 proxy listening on the local port `7070` (`<WEBAPP_PORT>`from now on).

- Fire up Firefox and configure it to redirect traffic to this proxy. 
- Open a new tab and go to `0.0.0.0:<API_PORT>/hello`. You should see a response from the API if everything is working correctly.
- Open a new tab and go to `localhost:<WEBAPP_PORT>`. You should see the home page if everything is working correctly.

**NOTE**: `0.0.0.0` will redirect the traffic to the SOCKS5 proxy. `localhost` or `127.0.0.1` will connect to the local computer where the browser is running.
