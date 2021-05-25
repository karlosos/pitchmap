---
id: intro
title: Introduction
sidebar_position: 1
---

# Introduction ⚡

This project is about researching tracking of football players on the playfield captured by a moving camera. Main objective of **automatic camera calibration** was achieved with available models, exactly based on [DonsetPG/narya](https://github.com/DonsetPG/narya).

:::caution
**This project is research only proof of concept.<br/>
We do not suggest trying to use this code as it is poorly mantained.**
:::

## Automatic calibration 

Automatic camera calibration module was developed using [Segmentation Models](https://github.com/qubvel/segmentation_models) based on [DonsetPG/narya](https://github.com/DonsetPG/narya). It uses *EfficentNetb3* and *FeautrePyramidNetwork*. **Notebooks and scripts of camera calibration model are available in [karlosos/camera_calib](https://github.com/karlosos/camera_calib) repository.**

Training data was acquired with [kkoripl/NaryaKeyPointsDatasetCreator](https://github.com/kkoripl/NaryaKeyPointsDatasetCreator). We've collected XX training images from XX matches played on XX stadiums.

If you are interested about sports camera calibration we can recommend [Awesome Sports Camera Calibration](https://github.com/cemunds/awesome-sports-camera-calibration) on GitHub.

:::note
Described model cannot succesfully detect homography in middle sections of the pitch. This is because of not enough characteristic points in this area. That's why **camera movement analysis** is required for calibrating all frames in input videos.
:::

## Camera movement analysis

**Camera movement analysis is required step for finding homography for each frame in input video.** Camera movement modeling was done using dense optical flow which is calculated on the startup of the application.

With camera movement model we can better interpolate homographies in comparison to naive approach of linear interpolation between calibrated frames in time. **Interpolation is performed in domain on camera angles**.

Optical flow is calculated only for non pitch area which is found using color ranges.

:::note
We calculate optical flow only for non pitch area by color ranges. Because of that our system cannot be used in situations where non pitch area is not distinguished by color.
:::

<!-- TODO: add images of segmentation -->

## Players detection

Players detection is achieved with Yolo model available in [cvlib](https://www.cvlib.net/). We only use objects with `person` label.

Each player is classified into 3 classes: team A, team B and referee. However, we know that it would be better to classifie each player into 5 classes (adding two classes for goalkeepers). 

