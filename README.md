# AminoRice API v2.1

Rice Quality Assurance API powered by a ConvNeXtV2-Nano ONNX model with 16 prediction targets.

## Deployment Target

This repository is configured for Vercel Python runtime using:

- `api/index.py` as the Vercel function entrypoint
- `vercel.json` for route/build config
- `app/main.py` as the FastAPI app module

## Endpoints

| Method | Route         | Description         |
| ------ | ------------- | ------------------- |
| GET    | `/`           | Root / welcome      |
| GET    | `/health`     | Health check        |
| POST   | `/register`   | Create account      |
| POST   | `/login`      | Get JWT token       |
| GET    | `/profile`    | View profile        |
| PUT    | `/profile`    | Update profile      |
| POST   | `/predict`    | Analyze rice image  |
| GET    | `/scans`      | Scan history        |
| GET    | `/scans/{id}` | Scan details        |
| DELETE | `/scans/{id}` | Delete scan         |
| POST   | `/chat`       | Rice expert chatbot |

## Required Environment Variables

- `MONGODB_URL`
- `DATABASE_NAME` (optional, default: `aminorice_db`)
- `USERS_COLLECTION` (optional, default: `users`)
- `SCANS_COLLECTION` (optional, default: `scans`)
- `SECRET_KEY`
- `ALGORITHM` (optional, default: `HS256`)
- `ACCESS_TOKEN_EXPIRE_MINUTES` (optional)
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`
- `OPENAI_API_KEY`
- `MODEL_PATH` (optional)
- `ONNX_MODEL_URL` (optional, direct downloadable URL to the .onnx file)
- `GDRIVE_ONNX_ID` (optional, used when model is downloaded on-demand)

## Vercel Notes

- Vercel serverless filesystem is read-only except `/tmp`.
- The API now defaults model storage to `/tmp/Final_Best_model.onnx` when running on Vercel.
- Startup is serverless-safe: MongoDB initializes lazily if lifespan hooks are skipped.

## Important Limitation

Large ONNX models can exceed Vercel function size/time/memory limits. If `/predict` times out or cold starts are too slow, move inference to a dedicated GPU/VM service and keep Vercel as the API gateway.
