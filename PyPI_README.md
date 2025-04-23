<h1 align="center">
  <a href="https://github.com/dfenny/retro/tree/main">
    <img src="images/LaneQ_background.png" alt="LaneQ logo" width="450" height="187">
  </a>
</h1>

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