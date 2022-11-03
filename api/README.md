# Face Generator API
Face Generator API built with Python using StyleGAN2.

## Setup
1. Move this folder (`api`) to your home folder.
```bash
mv ../api $HOME/api
```
2. Install `virtualenv` to keep all your dependencies in one place.
```bash
apt-get install virtualenv
```
3. Create a virtual environment and activate it
```bash
virtualenv venv --python=python3.6
source venv/bin/activate
```

4. Install the dependencies
```bash
pip install -r requirements.txt
```

5. Check hardware requirements are OK
```bash
python cli.py
```

## Run
```bash
cd $HOME/api
./run.sh
```

Generated images are saved to `$HOME/api/results/`

When you are done executing the API, deactivate the `virtualenv` running
```bash
deactivate
```
