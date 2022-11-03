import os

# Checking if stylegan2 is installed
os.system('echo Checking if stylegan2 is installed...')
os.system('cd src/stylegan2')
os.system('ls')

# Checking hardware requirements
os.system('echo ')
os.system('echo Checking hardware requirements...')
os.system('nvcc src/stylegan2/test_nvcc.cu -o test_nvcc -run')
