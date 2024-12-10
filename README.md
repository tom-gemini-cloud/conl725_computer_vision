# Edge Detection and Tracing with OpenCV

This project demonstrates various edge detection techniques using OpenCV in Python. It includes methods like Canny, Sobel, Laplacian, Prewitt, and Scharr for detecting edges in images. The project also provides functionality to trace these edges and visualize them.

## Features

- **Edge Detection Methods**:

  - Canny
  - Sobel
  - Laplacian
  - Prewitt
  - Scharr

- **Edge Tracing**:
  - Detects and traces edges in an image and visualizes them.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- dotenv (for environment variable management)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:

On Mac

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows

```bash
   python -m venv .venv
   .venv/Scripts/activate
```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up the environment variables:
   - Create a `.env` file in the root directory.
   - Add the following line to specify the image directory:
     ```
     IMAGE_DIR=path/to/your/image/directory
     ```
