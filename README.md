# GeomPT backend
Backend API for bridging the GeomPT frontend and backend, and interfacing with
the Firebase database.

## Usage
- Initialize a virtualenv with `python3 -m venv .venv`
- Install requirements with `pip install -r requirements.txt`
- Download secret key.json file from Firebase and adjust path to secret key in
  Credentials of `firebase_util.py`
- Run app with `python3 main.py`

