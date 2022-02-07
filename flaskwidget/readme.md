To run:

Basic requirements: python3 and git. Clone the repository.
git clone https://github.com/ShannonGFEMS/FLARE.git

Navigate inside that directory.

Create a virtual environment. This code is no longer being updated, so this will deconflict packages.
python -m venv env

Activate virtual environment.
source env/bin/activate

Install requirements.
pip install -r requirements.txt

Navigate inside the flask widget folder.

Export the application.
export FLASK_APP=GFEMSFlaskWidget3.py

Start the application.
flask run

Find it at 127.0.0.1:5000/home

crtl+C to stop.
