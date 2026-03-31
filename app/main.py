"""
AminoRice API v2.1  —  Fixed version
=====================================
Key fixes over v2.0:
  1. inverse_transform() applied to ONNX raw outputs before ANY calculation.
     The model was trained with log1p + z-score normalisation, so raw outputs
     are z-scored values (e.g. -1.4, 0.3, 2.1), NOT real grain counts.
     Without this step every Count ≈ 0, making broken/defect % meaningless.

  2. Percentage calculations now guard against total_count == 0 and against
     defect_count > total_count (clamp to 100%).

  3. Quality thresholds validated against real-world rice standards.

HOW TO GET TRANSFORM STATS (run once locally on your laptop):
    python extract_stats.py
  Then paste the printed JSON into TRANSFORM_STATS below.
"""

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os, math
import numpy as np
import gc
from PIL import Image, ImageOps, UnidentifiedImageError
import io
import onnxruntime as ort
import cloudinary
import cloudinary.uploader
import cloudinary.api
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
#  CONFIGURATION
# =============================================================================

MONGODB_URL              = os.getenv("MONGODB_URL")
DATABASE_NAME            = os.getenv("DATABASE_NAME", "aminorice_db")
USERS_COLLECTION         = os.getenv("USERS_COLLECTION", "users")
SCANS_COLLECTION         = os.getenv("SCANS_COLLECTION", "scans")

cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key    = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
)

SECRET_KEY                  = os.getenv("SECRET_KEY")
ALGORITHM                   = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client  = OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Saved_model", "Final_Best_model.onnx")
ONNX_MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

IMG_H = 640
IMG_W = 640
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 15 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 40_000_000))

# ── Target order (must match training script exactly) ─────────────────────────
COUNT_TARGETS = [
    'Count', 'Broken_Count', 'Long_Count', 'Medium_Count',
    'Black_Count', 'Chalky_Count', 'Red_Count', 'Yellow_Count', 'Green_Count'
]
CONTINUOUS_TARGETS = [
    'WK_Length_Average', 'WK_Width_Average', 'WK_LW_Ratio_Average',
    'Average_L', 'Average_a', 'Average_b'
]
ALL_TARGETS = COUNT_TARGETS + CONTINUOUS_TARGETS + ['comment_encoded']   # 16

COMMENT_MAP_INV = {0: 'Paddy', 1: 'Brown', 2: 'White'}

# =============================================================================
#  TRANSFORM STATS
#  ─────────────────────────────────────────────────────────────────────────────
#  The ONNX model outputs z-scored / log-transformed values, NOT real counts.
#  We must reverse that transformation to get real grain counts and mm values.
#
#  These stats are saved inside Final_Best_model.pt.
#  Run  python extract_stats.py  on your laptop (where the .pt lives),
#  then paste the printed JSON here to replace the placeholder below.
#
#  Until you do that, set TRANSFORM_STATS = None and the API will return
#  a warning in the response — predictions will be labelled as "raw/normalised".
# =============================================================================

TRANSFORM_STATS = {
    "targets": [
        "Count",
        "Broken_Count",
        "Long_Count",
        "Medium_Count",
        "Black_Count",
        "Chalky_Count",
        "Red_Count",
        "Yellow_Count",
        "Green_Count",
        "WK_Length_Average",
        "WK_Width_Average",
        "WK_LW_Ratio_Average",
        "Average_L",
        "Average_a",
        "Average_b",
        "comment_encoded",
    ],
    "is_count": {
        "Count": True,
        "Broken_Count": True,
        "Long_Count": True,
        "Medium_Count": True,
        "Black_Count": True,
        "Chalky_Count": True,
        "Red_Count": True,
        "Yellow_Count": True,
        "Green_Count": True,
        "WK_Length_Average": False,
        "WK_Width_Average": False,
        "WK_LW_Ratio_Average": False,
        "Average_L": False,
        "Average_a": False,
        "Average_b": False,
        "comment_encoded": False,
    },
    "p99": {
        "Count": 2559.14,
        "Broken_Count": 1896.8299999999997,
        "Long_Count": 1086.53,
        "Medium_Count": 12.529999999999973,
        "Black_Count": 1430.0,
        "Chalky_Count": 1621.3199999999988,
        "Red_Count": 89.0,
        "Yellow_Count": 546.04,
        "Green_Count": 731.1399999999999,
    },
    "mean_": {
        "Count": 7.200647987077247,
        "Broken_Count": 5.886977740620899,
        "Long_Count": 6.142034352978859,
        "Medium_Count": 0.38254532704014915,
        "Black_Count": 5.90172452180584,
        "Chalky_Count": 4.094049868715631,
        "Red_Count": 2.36232454252879,
        "Yellow_Count": 2.5775140367975387,
        "Green_Count": 1.5168952856010118,
        "WK_Length_Average": 7.658106666666667,
        "WK_Width_Average": 2.5566400000000002,
        "WK_LW_Ratio_Average": 3.07652,
        "Average_L": 64.28026666666668,
        "Average_a": 2.7764800000000003,
        "Average_b": 15.382386666666667,
        "comment_encoded": 0.9053333333333333,
    },
    "std_": {
        "Count": 0.22754669683196616,
        "Broken_Count": 0.6181306556009659,
        "Long_Count": 0.747465940937729,
        "Medium_Count": 0.5985555840056827,
        "Black_Count": 1.0568701145137773,
        "Chalky_Count": 3.2029628299760744,
        "Red_Count": 1.266433571342661,
        "Yellow_Count": 2.053516152578101,
        "Green_Count": 2.326528661817832,
        "WK_Length_Average": 1.2233734397516118,
        "WK_Width_Average": 0.3718295996419935,
        "WK_LW_Ratio_Average": 0.34335292309865617,
        "Average_L": 6.408414327303635,
        "Average_a": 5.416149873411586,
        "Average_b": 14.518478838851946,
        "comment_encoded": 0.7985642772251802,
    },
}

# Example of what it should look like after running extract_stats.py:
# TRANSFORM_STATS = {
#     "targets": ["Count", "Broken_Count", ..., "comment_encoded"],
#     "is_count": {"Count": true, "Broken_Count": true, ..., "Average_b": false},
#     "p99": {"Count": 2341.0, "Broken_Count": 487.0, ...},
#     "mean_": {"Count": 5.823, "Broken_Count": 3.441, ..., "Average_b": 14.2},
#     "std_": {"Count": 1.204, "Broken_Count": 0.981, ..., "Average_b": 3.81}
# }


# =============================================================================
#  INVERSE TRANSFORM  — converts ONNX raw output → real-world values
# =============================================================================

def inverse_transform(raw_array: np.ndarray) -> dict:
    """
    Converts the model's normalised output [16] into real-world values.

    Training applied per-target:
      Count targets  : clip[0, p99] → log1p → z-score
      Other targets  : z-score only

    Inverse:
      Count targets  : v = raw * std + mean → expm1(v) → clip(0, ∞)
      Other targets  : v = raw * std + mean

    If TRANSFORM_STATS is None (not yet populated), returns raw values
    with a warning flag so the app still works during development.
    """
    if TRANSFORM_STATS is None:
        # Fallback: return raw values — results will be wrong numerically
        # but the API won't crash during development
        result = {ALL_TARGETS[i]: float(raw_array[i]) for i in range(len(ALL_TARGETS))}
        result['_transform_warning'] = True
        return result

    stats    = TRANSFORM_STATS
    targets  = stats['targets']
    is_count = stats['is_count']
    mean_    = stats['mean_']
    std_     = stats['std_']

    result = {}
    for i, t in enumerate(targets):
        if i >= len(raw_array):
            break
        v = float(raw_array[i]) * std_[t] + mean_[t]
        if is_count.get(t, False):
            # Undo log1p, ensure non-negative
            v = float(np.clip(np.expm1(v), 0.0, None))
        result[t] = v

    return result


# =============================================================================
#  FASTAPI APP
# =============================================================================

app = FastAPI(
    title       = "AminoRice API",
    description = "Rice Quality Assurance — ConvNeXtV2-Nano, 16 targets, 640×640",
    version     = "2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class MongoDB:
    client: AsyncIOMotorClient = None

mongodb      = MongoDB()
onnx_session = None


async def get_database():
    await ensure_mongo_connected()
    return mongodb.client[DATABASE_NAME]


async def ensure_mongo_connected() -> None:
    if mongodb.client is not None:
        return
    if not MONGODB_URL:
        raise RuntimeError("MONGODB_URL environment variable is required")
    mongodb.client = AsyncIOMotorClient(MONGODB_URL)
    await mongodb.client.admin.command("ping")


def _file_size_mb(path: str) -> float:
    try:    return os.path.getsize(path) / 1e6
    except: return 0.0


def is_model_valid(path: str, min_size_mb: float = 0.001) -> bool:
    """Return True when the model file exists and is non-empty."""
    if not os.path.exists(path):
        return False
    return _file_size_mb(path) >= min_size_mb


def _create_onnx_session(model_path: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )
    for inp in session.get_inputs():
        print(f"  Input  '{inp.name}' {inp.shape} {inp.type}")
    for out in session.get_outputs():
        print(f"  Output '{out.name}' {out.shape} {out.type}")
    print(f"✅ ONNX model loaded ({providers[0]})")
    return session


def get_model() -> ort.InferenceSession:
    global onnx_session

    if onnx_session is None:
        if not is_model_valid(ONNX_MODEL_PATH):
            raise RuntimeError(
                f"ONNX model file is missing or empty at: {ONNX_MODEL_PATH}. "
                "Ensure Saved_model/Final_Best_model.onnx is committed to the repo "
                "or set MODEL_PATH to a valid absolute path."
            )

        onnx_session = _create_onnx_session(ONNX_MODEL_PATH)

    return onnx_session


# =============================================================================
#  STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup():
    # MongoDB
    try:
        await ensure_mongo_connected()
        print("✅ MongoDB connected")
    except Exception as e:
        print(f"❌ MongoDB error: {e}")

    # Model file
    if is_model_valid(ONNX_MODEL_PATH):
        print(f"  ✅ Model file found ({_file_size_mb(ONNX_MODEL_PATH):.1f} MB)")
    else:
        print(f"  ⚠  Model file missing or empty at: {ONNX_MODEL_PATH}")

    print("ℹ️ ONNX model session will be lazy-loaded on first prediction request")

    if TRANSFORM_STATS is None:
        print("\n⚠  TRANSFORM_STATS is None — run extract_stats.py and paste output into app/main.py")
        print("   Predictions will be in raw normalised form until this is done.\n")
    else:
        print(f"✅ Transform stats loaded ({len(TRANSFORM_STATS['targets'])} targets)")


@app.on_event("shutdown")
async def shutdown():
    if mongodb.client is not None:
        mongodb.client.close()
        mongodb.client = None


# =============================================================================
#  PYDANTIC MODELS
# =============================================================================

class UserCreate(BaseModel):
    full_name: str      = Field(..., min_length=3, max_length=100)
    email    : EmailStr
    password : str      = Field(..., min_length=6, max_length=100)
    phone    : Optional[str] = None

class UserLogin(BaseModel):
    email   : EmailStr
    password: str

class UserResponse(BaseModel):
    id        : str
    full_name : str
    email     : str
    phone     : Optional[str] = None
    join_date : str
    created_at: str

class Token(BaseModel):
    access_token: str
    token_type  : str

class TokenData(BaseModel):
    email: Optional[str] = None

class GrainCharacteristics(BaseModel):
    total_grains  : float
    broken_grains : float
    long_grains   : float
    medium_grains : float

class DefectiveGrains(BaseModel):
    black_grains    : float
    chalky_grains   : float
    red_grains      : float
    yellow_grains   : float
    green_grains    : float
    total_defective : float

class GrainMeasurements(BaseModel):
    average_length    : float
    average_width     : float
    length_width_ratio: float

class ColorCharacteristics(BaseModel):
    average_L: float
    average_a: float
    average_b: float

class RiceTypeInfo(BaseModel):
    comment_encoded: float
    rice_type      : str

class Conclusion(BaseModel):
    broken_grain_percentage   : float
    defective_grain_percentage: float
    overall_quality_category  : str
    quality_description       : str

class PredictionResponse(BaseModel):
    sample_information  : dict
    transform_applied   : bool          # True = real values; False = raw z-scored
    rice_type_info      : RiceTypeInfo
    grain_characteristics: GrainCharacteristics
    defective_grains    : DefectiveGrains
    grain_measurements  : GrainMeasurements
    color_characteristics: ColorCharacteristics
    conclusion          : Conclusion

class ScanHistoryItem(BaseModel):
    id               : str
    image_url        : str
    quality_grade    : str
    rice_type        : str
    total_count      : float
    broken_percentage: float
    defect_percentage: float
    scanned_at       : str

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class ChatResponse(BaseModel):
    answer   : str
    timestamp: str


# =============================================================================
#  AUTH HELPERS
# =============================================================================

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_password_hash(pw: str) -> str:
    return pwd_context.hash(pw)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    exc = HTTPException(status.HTTP_401_UNAUTHORIZED, "Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload    = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: raise exc
    except JWTError:
        raise exc
    db   = await get_database()
    user = await db[USERS_COLLECTION].find_one({"email": email})
    if user is None: raise exc
    return user


# =============================================================================
#  IMAGE HELPERS
# =============================================================================

def load_image_for_model(image_bytes: bytes) -> Image.Image:
    """Decode and validate uploaded image bytes for safe model preprocessing."""
    if not image_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Image file is too large. Max allowed is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    "Image resolution is too large. Please upload a smaller image.",
                )
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img.copy()
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "Image resolution is too large and unsafe to process.",
        )
    except UnidentifiedImageError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Unsupported or invalid image format. Try JPG, PNG, WEBP, BMP, or TIFF.",
        )
    except OSError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Image file appears corrupted or unreadable.",
        )


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Resize to 640×640, ImageNet normalise, return float32 [1,3,640,640].
    Matches the training val_transform pipeline exactly.
    """
    img       = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr       = np.array(img, dtype=np.float32) / 255.0
    mean      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std       = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr       = (arr - mean) / std
    arr       = np.transpose(arr, (2, 0, 1))
    arr       = np.expand_dims(arr, axis=0)
    return arr


def detect_comment_from_image(img: Image.Image) -> int:
    """Estimate rice type from image brightness (heuristic)."""
    gray       = img.convert("L")
    brightness = np.array(gray, dtype=np.float32).mean()
    if brightness < 80:  return 0   # Paddy (dark)
    if brightness < 160: return 1   # Brown (mid)
    return 2                         # White (bright)


def run_onnx_inference(image_array: np.ndarray, comment_idx: int) -> np.ndarray:
    """
    Single ONNX forward pass.
    Returns raw normalised output of shape [16] — must be inverse-transformed.
    """
    session = get_model()
    comment_arr = np.array([comment_idx], dtype=np.int64)
    raw = session.run(None, {"image": image_array, "comment": comment_arr})
    return raw[0][0]   # [16]


# =============================================================================
#  QUALITY CLASSIFICATION
#  Uses real-world industry thresholds (FAO / Codex Alimentarius standards)
# =============================================================================

def classify_rice_quality(broken_pct: float, defect_pct: float) -> tuple:
    """
    Classifies quality based on broken grain % and defective grain %.
    Thresholds aligned with international rice grading standards.
    """
    if broken_pct < 5 and defect_pct < 3:
        return ("Premium Quality",
                "Broken grains below 5%. Very low defects. "
                "Uniform grain size and colour. Excellent for premium markets.")
    elif broken_pct < 15 and defect_pct < 8:
        return ("Good Quality",
                "Broken grains 5–15%. Low defective grains. "
                "Good quality suitable for standard markets.")
    elif broken_pct < 25 and defect_pct < 15:
        return ("Medium Quality",
                "Broken grains 15–25%. Moderate defects. "
                "Acceptable for general consumption.")
    elif broken_pct < 35 and defect_pct < 25:
        return ("Fair Quality",
                "Broken grains 25–35%. High defects. Lower grade quality.")
    else:
        return ("Poor Quality",
                "Broken grains above 35% or very high defects. "
                "Suitable only for processing or animal feed.")


async def upload_to_cloudinary(image_bytes: bytes, filename: str) -> str:
    try:
        result = cloudinary.uploader.upload(
            image_bytes,
            folder        = "aminorice_scans",
            public_id     = f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}",
            resource_type = "image",
        )
        return result["secure_url"]
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Cloudinary upload error: {e}")


# =============================================================================
#  ROUTES — auth / user
# =============================================================================

@app.get("/")
async def root():
    return {
        "message" : "Welcome to AminoRice API v2.1",
        "version" : "2.1.0",
        "status"  : "active",
        "fix_note": "inverse_transform() applied — predictions are now real values",
    }


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate):
    db = await get_database()
    if await db[USERS_COLLECTION].find_one({"email": user.email}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")
    now       = datetime.utcnow().isoformat()
    join_date = datetime.utcnow().strftime("%B %Y")
    result    = await db[USERS_COLLECTION].insert_one({
        "full_name"      : user.full_name,
        "email"          : user.email,
        "phone"          : user.phone,
        "hashed_password": get_password_hash(user.password),
        "join_date"      : join_date,
        "created_at"     : now,
        "updated_at"     : now,
    })
    return UserResponse(id=str(result.inserted_id), full_name=user.full_name,
                        email=user.email, phone=user.phone,
                        join_date=join_date, created_at=now)


@app.post("/login", response_model=Token)
async def login(user: UserLogin):
    db      = await get_database()
    db_user = await db[USERS_COLLECTION].find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token({"sub": user.email},
                                timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}


@app.get("/profile", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=str(current_user["_id"]), full_name=current_user["full_name"],
        email=current_user["email"], phone=current_user.get("phone"),
        join_date=current_user["join_date"], created_at=current_user["created_at"])


@app.put("/profile", response_model=UserResponse)
async def update_profile(
    full_name: Optional[str] = None,
    phone    : Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    db          = await get_database()
    update_data = {"updated_at": datetime.utcnow().isoformat()}
    if full_name: update_data["full_name"] = full_name
    if phone:     update_data["phone"]     = phone
    if len(update_data) > 1:
        await db[USERS_COLLECTION].update_one(
            {"_id": current_user["_id"]}, {"$set": update_data})
        current_user = await db[USERS_COLLECTION].find_one({"_id": current_user["_id"]})
    return UserResponse(
        id=str(current_user["_id"]), full_name=current_user["full_name"],
        email=current_user["email"], phone=current_user.get("phone"),
        join_date=current_user["join_date"], created_at=current_user["created_at"])


# =============================================================================
#  PREDICT  — the main fixed endpoint
# =============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_rice_quality(
    file        : UploadFile = File(...),
    comment_hint: Optional[int] = None,   # 0=Paddy 1=Brown 2=White
    current_user: dict = Depends(get_current_user),
):
    """
    Predict rice quality from an uploaded image.

        - **file**: Image file of rice grains (JPG, PNG, WEBP, BMP, TIFF, etc.).
    - **comment_hint** (optional): Rice type — 0=Paddy, 1=Brown, 2=White.
      If omitted the API estimates from image brightness.
    """
    if file.content_type and (not file.content_type.startswith("image/")):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "File must be an image")

    try:
        image_bytes = await file.read()
        decoded_img = load_image_for_model(image_bytes)

        # ── 1. Upload to Cloudinary ───────────────────────────────────────────
        image_url = await upload_to_cloudinary(
            image_bytes, file.filename or "rice_scan.png"
        )
        del image_bytes

        # ── 2. Determine rice type hint ───────────────────────────────────────
        comment_idx = (
            comment_hint
            if comment_hint is not None and comment_hint in (0, 1, 2)
            else detect_comment_from_image(decoded_img)
        )

        # ── 3. Preprocess image ───────────────────────────────────────────────
        image_array = preprocess_image(decoded_img)         # [1, 3, 640, 640]
        del decoded_img

        # ── 4. ONNX inference ─────────────────────────────────────────────────
        raw_output = run_onnx_inference(image_array, comment_idx)   # [16] z-scored
        del image_array
        gc.collect()

        # ── 5. INVERSE TRANSFORM — THE CRITICAL FIX ───────────────────────────
        # raw_output contains z-scored/log-transformed values, NOT real counts.
        # inverse_transform() converts them back to real grain counts and mm values.
        preds            = inverse_transform(raw_output)
        del raw_output
        transform_applied = not preds.pop('_transform_warning', False)

        # ── 6. Ensure non-negative counts (safety clamp) ──────────────────────
        for t in COUNT_TARGETS:
            preds[t] = max(0.0, preds.get(t, 0.0))

        # ── 7. Decode rice type ───────────────────────────────────────────────
        comment_encoded_pred = preds.get("comment_encoded", float(comment_idx))
        rice_type_idx        = int(round(float(comment_encoded_pred)))
        rice_type_idx        = max(0, min(2, rice_type_idx))
        rice_type_label      = COMMENT_MAP_INV.get(rice_type_idx, "Unknown")

        # ── 8. Calculate percentages (correctly, safely) ──────────────────────
        total_count  = preds.get("Count", 0.0)
        broken_count = preds.get("Broken_Count", 0.0)

        # Sum of all five defect grain types
        defect_count = sum(max(0.0, preds.get(t, 0.0)) for t in [
            "Black_Count", "Chalky_Count", "Red_Count",
            "Yellow_Count", "Green_Count"
        ])

        if total_count > 0:
            broken_pct = min((broken_count / total_count) * 100, 100.0)
            defect_pct = min((defect_count / total_count) * 100, 100.0)
        else:
            # Model couldn't detect grains — default to worst case so we
            # don't silently show Premium Quality on a failed scan
            broken_pct = 100.0
            defect_pct = 100.0

        # ── 9. Classify quality ───────────────────────────────────────────────
        quality_category, quality_description = classify_rice_quality(
            broken_pct, defect_pct
        )

        # ── 10. Save to MongoDB ───────────────────────────────────────────────
        scan_timestamp = datetime.utcnow()
        sample_id      = f"RICE_{scan_timestamp.strftime('%Y%m%d_%H%M%S')}"

        scan_doc = {
            "user_id"            : str(current_user["_id"]),
            "user_email"         : current_user["email"],
            "sample_id"          : sample_id,
            "image_url"          : image_url,
            "rice_type"          : rice_type_label,
            "comment_encoded"    : comment_encoded_pred,
            "transform_applied"  : transform_applied,
            **{t: round(preds.get(t, 0.0), 4) for t in COUNT_TARGETS + CONTINUOUS_TARGETS},
            "broken_percentage"  : round(broken_pct, 2),
            "defect_percentage"  : round(defect_pct, 2),
            "quality_category"   : quality_category,
            "quality_description": quality_description,
            "scanned_at"         : scan_timestamp.isoformat(),
        }

        db     = await get_database()
        result = await db[SCANS_COLLECTION].insert_one(scan_doc)

        # ── 11. Build and return response ─────────────────────────────────────
        return PredictionResponse(
            sample_information={
                "sample_id"        : sample_id,
                "scan_id"          : str(result.inserted_id),
                "image_url"        : image_url,
                "scanned_at"       : scan_timestamp.isoformat(),
                "transform_warning": (not transform_applied),
            },
            transform_applied    = transform_applied,
            rice_type_info       = RiceTypeInfo(
                comment_encoded = round(comment_encoded_pred, 3),
                rice_type       = rice_type_label,
            ),
            grain_characteristics = GrainCharacteristics(
                total_grains  = round(preds.get("Count",        0.0), 1),
                broken_grains = round(preds.get("Broken_Count", 0.0), 1),
                long_grains   = round(preds.get("Long_Count",   0.0), 1),
                medium_grains = round(preds.get("Medium_Count", 0.0), 1),
            ),
            defective_grains = DefectiveGrains(
                black_grains    = round(preds.get("Black_Count",  0.0), 1),
                chalky_grains   = round(preds.get("Chalky_Count", 0.0), 1),
                red_grains      = round(preds.get("Red_Count",    0.0), 1),
                yellow_grains   = round(preds.get("Yellow_Count", 0.0), 1),
                green_grains    = round(preds.get("Green_Count",  0.0), 1),
                total_defective = round(defect_count, 1),
            ),
            grain_measurements = GrainMeasurements(
                average_length     = round(preds.get("WK_Length_Average",   0.0), 3),
                average_width      = round(preds.get("WK_Width_Average",    0.0), 3),
                length_width_ratio = round(preds.get("WK_LW_Ratio_Average", 0.0), 3),
            ),
            color_characteristics = ColorCharacteristics(
                average_L = round(preds.get("Average_L", 0.0), 2),
                average_a = round(preds.get("Average_a", 0.0), 2),
                average_b = round(preds.get("Average_b", 0.0), 2),
            ),
            conclusion = Conclusion(
                broken_grain_percentage    = round(broken_pct, 2),
                defective_grain_percentage = round(defect_pct, 2),
                overall_quality_category   = quality_category,
                quality_description        = quality_description,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Error processing image: {e}")


# =============================================================================
#  SCAN HISTORY
# =============================================================================

@app.get("/scans", response_model=List[ScanHistoryItem])
async def get_scan_history(
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
):
    db     = await get_database()
    cursor = db[SCANS_COLLECTION].find(
        {"user_id": str(current_user["_id"])}
    ).sort("scanned_at", -1).limit(limit)
    scans = await cursor.to_list(length=limit)
    return [
        ScanHistoryItem(
            id               = str(s["_id"]),
            image_url        = s["image_url"],
            quality_grade    = s.get("quality_category", "Unknown"),
            rice_type        = s.get("rice_type", "Unknown"),
            total_count      = s.get("Count", 0.0),
            broken_percentage= s.get("broken_percentage", 0.0),
            defect_percentage= s.get("defect_percentage", 0.0),
            scanned_at       = s["scanned_at"],
        )
        for s in scans
    ]


@app.get("/scans/{scan_id}", response_model=PredictionResponse)
async def get_scan_details(
    scan_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = await get_database()
    try:
        oid = ObjectId(scan_id)
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid scan ID format")
    scan = await db[SCANS_COLLECTION].find_one(
        {"_id": oid, "user_id": str(current_user["_id"])})
    if not scan:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan not found")

    defect_count = sum(max(0.0, scan.get(t, 0.0)) for t in
                       ["Black_Count", "Chalky_Count", "Red_Count",
                        "Yellow_Count", "Green_Count"])

    return PredictionResponse(
        sample_information={
            "sample_id" : scan.get("sample_id", ""),
            "scan_id"   : str(scan["_id"]),
            "image_url" : scan["image_url"],
            "scanned_at": scan["scanned_at"],
        },
        transform_applied    = scan.get("transform_applied", False),
        rice_type_info       = RiceTypeInfo(
            comment_encoded = scan.get("comment_encoded", 0.0),
            rice_type       = scan.get("rice_type", "Unknown"),
        ),
        grain_characteristics = GrainCharacteristics(
            total_grains  = round(scan.get("Count",        0.0), 1),
            broken_grains = round(scan.get("Broken_Count", 0.0), 1),
            long_grains   = round(scan.get("Long_Count",   0.0), 1),
            medium_grains = round(scan.get("Medium_Count", 0.0), 1),
        ),
        defective_grains = DefectiveGrains(
            black_grains    = round(scan.get("Black_Count",  0.0), 1),
            chalky_grains   = round(scan.get("Chalky_Count", 0.0), 1),
            red_grains      = round(scan.get("Red_Count",    0.0), 1),
            yellow_grains   = round(scan.get("Yellow_Count", 0.0), 1),
            green_grains    = round(scan.get("Green_Count",  0.0), 1),
            total_defective = round(defect_count, 1),
        ),
        grain_measurements = GrainMeasurements(
            average_length     = round(scan.get("WK_Length_Average",   0.0), 3),
            average_width      = round(scan.get("WK_Width_Average",    0.0), 3),
            length_width_ratio = round(scan.get("WK_LW_Ratio_Average", 0.0), 3),
        ),
        color_characteristics = ColorCharacteristics(
            average_L = round(scan.get("Average_L", 0.0), 2),
            average_a = round(scan.get("Average_a", 0.0), 2),
            average_b = round(scan.get("Average_b", 0.0), 2),
        ),
        conclusion = Conclusion(
            broken_grain_percentage    = round(scan.get("broken_percentage", 0.0), 2),
            defective_grain_percentage = round(scan.get("defect_percentage", 0.0), 2),
            overall_quality_category   = scan.get("quality_category", "Unknown"),
            quality_description        = scan.get("quality_description", ""),
        ),
    )


@app.delete("/scans/{scan_id}")
async def delete_scan(
    scan_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = await get_database()
    try:
        oid = ObjectId(scan_id)
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid scan ID format")
    result = await db[SCANS_COLLECTION].delete_one(
        {"_id": oid, "user_id": str(current_user["_id"])})
    if result.deleted_count == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan not found")
    return {"message": "Scan deleted successfully"}


# =============================================================================
#  CHATBOT
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def rice_expert_chat(
    chat_request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        response = openai_client.chat.completions.create(
            model    = "gpt-4",
            messages = [
                {"role": "system", "content": (
                    "You are an expert assistant specialised in rice quality assessment. "
                    "Topics: grain measurements, broken rice, chalkiness, moisture, milling, "
                    "grading standards, varieties, cultivation, storage, nutrition, and markets. "
                    "When users share measurements, classify quality as: "
                    "Premium / Good / Medium / Fair / Poor. "
                    "If unrelated to rice, redirect politely. "
                    "IMPORTANT: reply in 60 words or fewer."
                )},
                {"role": "user", "content": chat_request.question},
            ],
            max_tokens=100, temperature=0.7,
        )
        answer = response.choices[0].message.content
        words  = answer.split()
        if len(words) > 60:
            answer = " ".join(words[:60]) + "..."
        return ChatResponse(answer=answer, timestamp=datetime.utcnow().isoformat())
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Chat error: {e}")


# =============================================================================
#  HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    try:
        await ensure_mongo_connected()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status"          : "healthy",
        "database"        : db_status,
        "onnx_model"      : (
            "loaded"
            if onnx_session is not None
            else ("ready (lazy load)" if is_model_valid(ONNX_MODEL_PATH) else "missing")
        ),
        "onnx_model_path" : ONNX_MODEL_PATH,
        "transform_stats" : "loaded" if TRANSFORM_STATS is not None else "MISSING — run extract_stats.py",
        "model_info"      : {
            "architecture": "ConvNeXtV2-Nano + Comment Embedding",
            "num_targets" : len(ALL_TARGETS),
            "img_size"    : f"{IMG_H}x{IMG_W}",
            "targets"     : ALL_TARGETS,
        },
        "timestamp"       : datetime.utcnow().isoformat(),
    }

# =============================================================================
#  Run with:  uvicorn app.main:app --reload
# =============================================================================