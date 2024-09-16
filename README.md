# *GeomPT* Backend
GeomPT is an AI-powered web platform for physical therapy providers to track and analyze patients' range of motion (ROM) over time. It addresses patient non-adherence and improves rehabilitation outcomes through data-driven insights and enhanced plan-of-care adherence.

<div align="center">
  <img src="https://github.com/user-attachments/assets/74b3948c-56f6-4f82-8bec-0bba67086dc6" alt="GeomPT GIF video">
</div>

## Stack
- Flask
- Firebase
- OpenCV
- Mediapipe (Pose Model)

## Project Structure
- **main.py**  
  The primary interface between the frontend and backend. It:
  - Processes video data from the frontend using websockets
  - Captures automatic measurements using angle thresholds
  - Processes the video data using helpers from `opencv_logic.py`
  - Stores the ROM results and exercise tracking data in Firebase using `firebase_util.py`

- **firebase_util.py**  
  Contains helper functions to:
  - Upload video and image data to Firebase
  - Track and store exercise data per user and workout in Firebase DB

- **opencv_logic.py**  
  Contains the computer vision logic for:
  - Processing video data using OpenCV and Mediapipe's Pose Model to analyze Range of Motion (ROM)
  - Extracting angles and key joint positions from patient videos for analysis and tracking over time
  - Draws vectors between limbs, computes angle, and smoothes data using rolling average 

## Usage
To run this project locally:

1. Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/geompt.git](https://github.com/GeomPT/backend.git)
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up Firebase:
    - Download your Firebase secret key (`key.json`) from the Firebase console.
    - Adjust the path to the secret key in the `Credentials` section of `firebase_util.py`.

5. Run the backend:
    ```bash
    python3 main.py
    ```

6. Run the frontend (follow instructions from the frontend README).

As you use the frontend to upload videos, the backend automatically processes these videos using OpenCV and Mediapipe, and uploads the results to Firebase for further analysis.

## Contribution
To contribute to this project:

1. Clone the repository locally:
    ```bash
    git clone https://github.com/yourusername/geompt.git
    ```

2. Set up a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the Firebase secret key (`blah.json`), and adjust the path in `firebase_util.py`.

5. Run the backend:
    ```bash
    python3 main.py
    ```

## Credit
- Hector Astrom
- Allen Liu
- Sandro Gopal
- Steven Zhang
