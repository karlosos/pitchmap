---
sidebar_position: 3
---

# Development 👨🏻‍💻

## Requirements

Create virtual environment with tool of your choice (`virtualenv`, `pipenv`, `conda`) and activate it. Install requirements:

```
pip install -r requirements.clean.txt
```

:::caution
`mxnet` on windows requires [Microsoft C++ Build Tools](visualstudio.microsoft.com/visual-cpp-build-tools)
:::

## Environment variable

`$SM_FRAMERK` variable is required for proper working of deep neural netowrk model. Define it with:

```
$env:SM_FRAMEWORK="tf.keras"
```

Or with you IDE of choice.

## Running application

In `pitchmap/main.py` change input video path by modificating `self.video_name` field in `PitchMap` class. Footage should be stored in `data/` directory.

Run application with

```
python pitchmap/main.py
```

## Documentation

This website was built with [Docusaurus](https://docusaurus.io/). To start developing run:

```
cd website
npm start
```

Deploying to GitHub pages:

```
cmd /C 'set "GIT_USER=karlosos" && npm run deploy'
```
