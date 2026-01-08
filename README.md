# DragonMemory (BDH Model)

This repository contains the implementation of the BDH (Dragon Hatchling) model for narrative consistency checking.

## Setup Instructions

### 1. Create and Activate Virtual Environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Libraries

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Run the Model

Run the training/inference script:

```bash
python train.py
```

### Output

After running the script, the predictions will be saved to a file named **`results.csv`**.
