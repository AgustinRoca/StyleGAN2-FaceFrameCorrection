#!/bin/bash
~/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 4800 face_generator:app
