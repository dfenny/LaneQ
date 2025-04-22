Automated Lane Line Quality Assessment

![Model Architecture with Example](https://raw.githubusercontent.com/dfenny/LaneQ/main/images/architecture_with_example.png)

Install:

`python3 -m pip install laneq`

Get started easily with:

```python
from laneq import DegradationDetector

dt = DegradationDetector("output")
dt.predict("path/to/dashcam/image.jpg")
```

Or to run the GUI:

`python -m laneq`