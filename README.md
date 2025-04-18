## Preparing Recording Files/Data from Depth Camera (Intel RealSense D435)

Bag files can be downloaded from the shared Google Drive folder:

📁 [Google Drive Link](https://drive.google.com/drive/folders/1m8xXoC6dJsAKacj06d-5bL6_e20H10J1?usp=drive_link)

- For the **component** bag file case, the desired angle is **45 degrees**.
- For the **compressor 4** bag file case, the desired angle should be set to **66 degrees**.  
  → *Note: Filter parameters should also be adjusted accordingly.*

---

## Modified/Improved Function

The original issue was that the **estimated normal vectors were fluctuating** due to real-time changes in the point cloud distribution.

To address this, an **`angle_threshold`** has been introduced.  
- When the angle difference between the current and previous normal vectors is less than **0.2 degrees**, the previous vector is reused.
- This effectively reduces jitter and stabilizes the normal estimation by **allowing only minimal fluctuations**.

---

## Future Work

To improve performance in real-world scenarios, it is important to ensure **robust and stable depth information** in **dynamic environments**, especially under challenging conditions like **glare** and **changing lighting**.
