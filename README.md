# BLIP-2 Multimodal Image Captioning, Q&A, and Retrieval System
![img1](https://github.com/user-attachments/assets/03384ae2-412f-45a9-8e66-7aa7a98f7043)
This project integrates Salesforce's BLIP-2 with OpenAI's CLIP to build a multimodal system that:
- Generates captions for images,
- Answers visual questions, and
- Retrieves similar images from a gallery based on the generated caption.

> **Note:** This project is designed to run on GPU-enabled environments. If you have a multi-GPU setup, please follow the configuration instructions below to avoid device assignment issues.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation & Configuration](#installation--configuration)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The system leverages:
- **BLIP-2** to perform image captioning and visual question answering.
- **CLIP** to extract semantic features for both text and images, enabling retrieval of similar images.
- **Gradio** to create a simple web UI for interaction.

## Features

- **Image Captioning:** Automatically generate a description for an uploaded image using BLIP-2.
- **Visual Question Answering:** Answer questions about the image via a prompt-based interface.
- **Image Retrieval:** Use the generated caption as a query to retrieve the top-K semantically similar images from a local gallery using CLIP.

## File Structure

```plaintext
blip2_multimodal/
├── demo.py                       # Main entry (Gradio UI)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── models/
│   └── blip2_wrapper.py         # BLIP-2 model wrapper (caption & Q&A)
├── utils/
│   └── retrieval.py             # CLIP-based image retrieval module
└── assets/
    └── gallery/                 # Directory containing gallery images (e.g., JPG/PNG)
```
## Running the Project
To start the Gradio web interface:
```plaintext
python demo.py
```
## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for bug fixes and new features.

## License
```plaintext
This project is licensed under the MIT License.
```
