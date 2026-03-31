---
title: AminoRice API
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_file: app.py
pinned: false
---

# 🌾 AminoRice API v2.1

Rice Quality Assurance API powered by **ConvNeXtV2-Nano** ONNX model with 16 prediction targets.

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Root / welcome |
| GET | `/health` | Health check |
| POST | `/register` | Create account |
| POST | `/login` | Get JWT token |
| GET | `/profile` | View profile |
| PUT | `/profile` | Update profile |
| POST | `/predict` | Analyze rice image |
| GET | `/scans` | Scan history |
| GET | `/scans/{id}` | Scan details |
| DELETE | `/scans/{id}` | Delete scan |
| POST | `/chat` | Rice expert chatbot |

## Interactive Docs

Visit `/docs` for the full Swagger UI.

## Model

- **Architecture**: ConvNeXtV2-Nano + Comment Embedding
- **Input**: 640×640 RGB image
- **Outputs**: 16 targets (grain counts, measurements, colour, rice type)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference