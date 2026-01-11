# DragonMemory (BDH Model)

This repository contains the implementation of the BDH (Dragon Hatchling) model for narrative consistency checking.

## Setup Instructions

> [!IMPORTANT]
> **Evaluator Note:** Please ensure the `test.csv` file is placed in the `data` folder before proceeding with the steps below.

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
python predict.py
```

### Output

After running the script, the predictions will be saved to a file named **`results.csv`**.

## Optional

In the `Optional` folder, I have tried with representation learning. Follow the above steps to run it same.


