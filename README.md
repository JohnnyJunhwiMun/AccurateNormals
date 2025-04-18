## Preparing Recording Files/Data from Depth Camera (Intel RealSense D435)

Bag files can be downloaded from the shared Google Drive folder:

📁 [Google Drive Link](https://drive.google.com/drive/folders/1m8xXoC6dJsAKacj06d-5bL6_e20H10J1?usp=drive_link)

- For the **component** bag file case, the desired angle is **45 degrees**.
- For the **compressor 4** bag file case, the desired angle should be set to **66 degrees**.  
  → *Note: Filter parameters should also be adjusted accordingly.*

---

## Modified/Improved Function

The original issue was that the **estimated normal vectors were fluctuating** due to real-time changes in the point cloud distribution.

To address this, an **`angle_threshold`** has been introduced:  
- When the angle difference between the current and previous normal vectors is less than **0.2 degrees**, the previous vector is reused.
- This minimizes jitter and stabilizes the normal estimation by **allowing only small fluctuations**.

Since the estimated normals are now nearly fixed in each frame, the **starting point of the recording** becomes critical. If recording begins while the normals are unstable, it can negatively impact the overall performance.

To ensure proper initialization, an **`angle_tolerance`** mechanism has been implemented:  
- It allows a deviation of **±2 degrees** from the **`desired_angle`**.
- If this condition is met continuously for **`required_tolerance_frames`** (15 frames in the current implementation), the **`angle_threshold`** function is activated.
- This helps the system begin with **stable normal vectors** for each target point.

---

## Results

Check out the real-time estimation:

https://github.com/user-attachments/assets/984aea3c-b87e-4962-8b40-fc4a6cf73b7b


Before applying the improved function, the estimated normals showed significant fluctuations, with some frames exceeding the threshold boundary.  
With the improved function, **100% accuracy** is achieved—no red markers are observed in the graph.

### Improved Version  
![frame_avg_20250418_131742](https://github.com/user-attachments/assets/6c004ed2-39a9-49ef-8ad2-bd773df3fa89)

### Original Version  
![high_k50_case2](https://github.com/user-attachments/assets/56d05a21-67b4-4683-bdf3-f50884c36e84)

---

## Future Work

To improve performance in real-world scenarios, it's important to ensure **robust and stable depth information** in **dynamic environments**, especially under challenging conditions such as **glare** and **changing lighting**.

Currently, a key limitation of this system is that it must know the **actual angle difference between normal vectors**, which can be difficult to estimate accurately in practice.
