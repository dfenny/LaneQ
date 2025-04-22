<h1 align="center">
  <a href="https://github.com/dfenny/retro/tree/main">
    <img src="images/LaneQ_background.png" alt="LaneQ logo" width="450" height="187">
  </a>
</h1>

<!-- # LaneQ -->

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)



## About

**LaneQ** is a lightweight Python library that automatically evaluates the quality of lane line markings on roads, making it ideal for applications in autonomous driving, map data validation, and infrastructure monitoring. It quantifies the clarity, visibility, and consistency of lane markings, providing actionable insights for road safety analysis and maintenance.


<div align="center">
  <img src="images/day.jpg" alt="Day time image" width="400" style="margin-right: 10px;">
  <img src="images/night.jpg" alt="Image 2 description" width="400">
</div>

## Features
- Evaluate clarity, visibility, and consistency of lane markings
- Simple and flexible API for easy integration
- Supports both real-world and simulated images
- Easily integrable with larger computer vision pipelines
- Configurable models



## Installation

### Option 1: Install via pip
To install **LaneQ** from PyPI, you can simply run:

```bash
pip install laneq
```

This will install the latest version of the package and its dependencies.

### Option 2: Clone the Repository

If you'd prefer to work with the latest development version or contribute to the project, you can clone the repository directly:

Clone the repository:

```bash
git clone https://github.com/dfenny/retro.git
cd retro
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

(Optional) Set up the environment:

For better isolation, you can create and activate a virtual environment before installing the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Get started easily with:

```python
from laneq import DegradationDetector

dt = DegradationDetector("output")
dt.predict("path/to/dashcam/image.jpg")
```

Or to run the GUI:

```bash
python -m laneq
```

Example usage and description of available options.
<!-- <div align="center">
  <img src="images/architecture_with_example.png" alt="Day time image" width="800">
</div> -->



## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

---

## License

See `LICENSE` for more information.
