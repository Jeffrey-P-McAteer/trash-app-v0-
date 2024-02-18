#!/usr/bin/env python

import os
import sys
import subprocess
import importlib
import time
import shutil

def ensure_packages_installed():
  modules_and_pkg_names = [
    ('cv2', 'opencv-python'),
    ('ultralytics', 'ultralytics'),
  ]

  if sys.platform.startswith('win'):
    modules_and_pkg_names.append(
      ('torch', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    )
  else:
    modules_and_pkg_names.append(
      ('torch', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    )

  # By including the version data in the path, we avoid the possibility that 2 copies of python on the same box
  # share the package folder, resulting in eg a copy of python 3.8 installing ancient numpy and then
  # a copy of python 3.10 running this same script and trying to import modules that do not match!
  vers_major = sys.version_info[0]
  vers_minor = sys.version_info[1]
  py_pkgs_folder = os.path.join(os.path.dirname(__file__), 'pyenv', f'{vers_major}_{vers_minor}')

  os.makedirs(py_pkgs_folder, exist_ok=True)

  if not py_pkgs_folder in sys.path:
    sys.path.insert(0, py_pkgs_folder)
  if not py_pkgs_folder in os.environ.get('PYTHONPATH', ''):
    os.environ['PYTHONPATH'] = py_pkgs_folder + os.pathsep + os.environ.get('PYTHONPATH', '')
  for module_name, pkg_names in modules_and_pkg_names:
    try:
      _ = importlib.import_module(module_name)
    except:
      try:
        import pip # prove we _have_ pip
        subprocess.run([
          sys.executable, '-m', 'pip', 'install', '--target={}'.format(py_pkgs_folder), *(pkg_names.split())
        ])
        _ = importlib.import_module(module_name)
      except:
        subprocess.run([
          sys.executable, '-m', 'ensurepip',
        ])
        subprocess.run([
          sys.executable, '-m', 'pip', 'install', '--target={}'.format(py_pkgs_folder), *(pkg_names.split())
        ])
        _ = importlib.import_module(module_name) # if this blows, up, fatal error installing package!



ensure_packages_installed()
import cv2
import torch
import ultralytics

def main(args=sys.argv):
  # See https://docs.ultralytics.com/tasks/detect/#models to download model files
  model_data_dir = os.path.join(os.path.dirname(__file__), 'model-data')
  os.makedirs(model_data_dir, exist_ok=True)
  license_plate_model = ultralytics.YOLO(os.path.join(model_data_dir, 'yolov8x.pt'))

  vidcap = None
  if len(args) > 1 and os.path.exists(args[1]):
    print(f'Opening the video file {args[1]}')
    vidcap = cv2.VideoCapture(args[1])
  elif len(args) > 1:
    print(f'Opening camera number {args[1]}')
    camera_num = int(args[1])
    vidcap = cv2.VideoCapture(camera_num)
  else:
    print(f'Opening the first camera')
    vidcap = cv2.VideoCapture(0)

  # Read frames & process each one w/ debug UI
  cv2.namedWindow('floatme', cv2.WINDOW_NORMAL)
  success = True
  while success:
    success, frame = vidcap.read()
    if not success:
      break
    # Process frame

    cv2.imshow('floatme', frame)

    key = cv2.waitKey(1)
    if (key == 27): # esc
        break

    time.sleep(0.08)

  # release resources
  try:
    cv2.destroyAllWindows()
    vidcap.release()
  except:
    pass




if __name__ == '__main__':
  main()
