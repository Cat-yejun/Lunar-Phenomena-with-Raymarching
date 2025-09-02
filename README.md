# 🌙 Lunar Phenomena with Raymarching

 This project uses **Python + OpenGL(GLUT) + GLSL** to simulate the positions of the Sun, Earth and Moon, along with **lighting, shadow effects, texture mapping,** and **Ray Marching** rendering techniques.

⚠️ Currently, the code is designed for **Windows only**.

------------------------------------------------------------------------

## ✨ Features
-   **Ray Marching Rendering**
    -   Renders planets as spheres using Signed Distance Functions
        (SDF)
    -   Implements soft shadow effects with light attenuation
-   **Orbit Data Integration**
    -   Loads `data/high_dt/realX.npy`, `realY.npy`, `realZ.npy` to
        compute planetary positions
    -   Planetary position data are numerically computed by 4th order Runge-Kutta methods
-   **Texture Mapping**
    -   Applies textures to Earth (`earthmap.bmp`) and the Moon
        (`moon.bmp`)
-   **Camera Perspective**
    -   Dynamically adjusts the camera viewpoint based on
        Sun-Earth-Moon relations
    -   The camera always faces the Moon
 
-----------------------------------------------------------------------------------------

## 📂 Project Structure
    Lunar-Phenomena-with-Raymarching/
    │── data/
    │    └── high_dt/         # Planetary position data (realX.npy, realY.npy, realZ.npy)
    │── textures/
    │    ├── earthmap.bmp
    │    └── moon.bmp
    │── main.py               # Main execution file
    │── requirements.txt      # Required Python libraries
    │── README.md


---------------------------------------------------------------------------------------

## ⚙️ Requirements

-   Windows 10+
-   Python 3.8+
-   Python libraries:
    -   PyOpenGL\
    -   PyOpenGL_accelerate\
    -   PyGLM\
    -   Pillow (PIL)\
    -   Numpy

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Usage

``` bash
python main.py
```

When executed, an 800x800 OpenGL window will open,\
rendering a ray-marched simulation of the Sun-Earth-Moon system with
realistic orbits and shadows.

------------------------------------------------------------------------

## 🖼️ Screenshots

## Lunar Eclipse Simulation - 05/26/2021
<img width="844" height="689" alt="image" src="https://github.com/user-attachments/assets/78f31968-ca3e-453d-9f33-61295eb179ea" />
- data source: NASA JPL

https://github.com/user-attachments/assets/9d187639-502f-4a0d-af90-15d8ec06a994


## Lunar Phases

https://github.com/user-attachments/assets/b5484618-2d15-4922-bf9e-6fb92c377ba8

------------------------------------------------------------------------


## 🔮 Future Work

-  Incorporate Earth’s atmospheric scattering effect to simulate the reddish tint observed during lunar eclipses, making the rendering more physically realistic.
