# Week 0 Solar Challenge

Analysis of solar farm data for MoonLight Energy Solutions to identify high-potential regions for solar investments.

## Project Overview

This repository contains the setup and analysis for Week 0 of the 10 Academy Data Science program. Tasks include Git setup, data profiling, EDA, cross-country comparison, and an optional Streamlit dashboard.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hallelujah-kib/solar-challenge-week0.git
   cd solar-challenge-week0
2. **Create and activate virtual environment**:
    python -m venv venv
    venv\Scripts\activate # macOS: source venv/bin/activate 
3. **Install dependencies**:
    pip install -r requirements.txt
4. **Verify setup**:
    python --version
    pip list

## Folder Structure
├── .vscode/

│   └── settings.json

├── .github/

│   └── workflows

│       ├── unittests.yml

├── .gitignore

├── requirements.txt

├── README.md

 |------ src/

├── notebooks/

│   ├── __init__.py

│   └── README.md

├── tests/

│   ├── __init__.py

└── scripts/
    ├── __init__.py

    └── README.md

## CI/CD
A GitHub Actions workflow (.github/workflows/unittests.yml) runs on push/PR, installing dependencies and verifying Python version.

## Next Steps
- Perform EDA on Benin, Sierra Leone, Togo datasets (Task 2).
- Compare solar potential across countries (Task 3).
- Develop Streamlit dashboard (Task 4).