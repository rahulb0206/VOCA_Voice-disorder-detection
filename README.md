# VOCA - Voice Disorder Detection 
This repository contains the machine learning pipeline developed as part of the VOCA Health project at Carle Illinois College of Medicine. The goal of the project is to assist healthcare professionals in detecting voice disorders using machine learning techniques applied to vocal audio signals.

The system analyzes patient voice recordings and predicts whether the voice pattern indicates a normal or disordered vocal condition, enabling scalable and automated screening support for clinicians.

Full-stack iOS application with Flask backend.

## Structure
- `ios-app/`: iOS application
- `backend/`: Flask server

## Setup Instructions

### iOS App
1. Open `ios-app/ProjectName.xcodeproj`
2. Install dependencies (if using CocoaPods/SPM)
3. Build and run

### Backend
1. Create virtual environment: `python -m venv venv`
2. Activate venv: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run server: `python run.py`

Note:
The original dataset used in the VOCA Health project cannot be publicly shared.
