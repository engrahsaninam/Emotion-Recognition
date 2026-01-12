"""
Security and Authentication Module

Provides JWT authentication, password hashing, and API key validation.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel


# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============== Models ==============

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    api_key: Optional[str] = None
    usage_count: int = 0
    usage_limit: int = 1000
    created_at: datetime = datetime.utcnow()


# ============== Password Functions ==============

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# ============== Token Functions ==============

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        email = payload.get("email")
        scopes = payload.get("scopes", [])
        
        if user_id is None:
            return None
            
        return TokenData(user_id=user_id, email=email, scopes=scopes)
    except JWTError:
        return None


def create_tokens(user: User) -> Token:
    """Create both access and refresh tokens for a user."""
    token_data = {
        "sub": user.id,
        "email": user.email,
        "scopes": ["user", "admin"] if user.is_admin else ["user"]
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


# ============== API Key Functions ==============

def generate_api_key() -> str:
    """Generate a new API key."""
    return f"rs_{secrets.token_urlsafe(32)}"


def validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    return api_key.startswith("rs_") and len(api_key) > 20


# ============== In-Memory Storage (Replace with DB in production) ==============

# Simulated user database
_users_db: dict[str, dict] = {}
_api_keys_db: dict[str, str] = {}  # api_key -> user_id


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email."""
    for user_data in _users_db.values():
        if user_data.get("email") == email:
            return User(**user_data)
    return None


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID."""
    user_data = _users_db.get(user_id)
    if user_data:
        return User(**user_data)
    return None


def get_user_by_api_key(api_key: str) -> Optional[User]:
    """Get user by API key."""
    user_id = _api_keys_db.get(api_key)
    if user_id:
        return get_user_by_id(user_id)
    return None


def create_user(email: str, password: str, full_name: Optional[str] = None) -> User:
    """Create a new user."""
    import uuid
    
    user_id = str(uuid.uuid4())
    api_key = generate_api_key()
    
    user_data = {
        "id": user_id,
        "email": email,
        "full_name": full_name,
        "hashed_password": get_password_hash(password),
        "is_active": True,
        "is_admin": False,
        "api_key": api_key,
        "usage_count": 0,
        "usage_limit": 1000,
        "created_at": datetime.utcnow()
    }
    
    _users_db[user_id] = user_data
    _api_keys_db[api_key] = user_id
    
    return User(**user_data)


def increment_usage(user_id: str) -> bool:
    """Increment user's API usage count."""
    if user_id in _users_db:
        _users_db[user_id]["usage_count"] += 1
        return True
    return False


# ============== Dependencies ==============

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> User:
    """
    Get current authenticated user from JWT token or API key.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Try API key first
    if api_key:
        user = get_user_by_api_key(api_key)
        if user:
            if not user.is_active:
                raise HTTPException(status_code=403, detail="User account is disabled")
            if user.usage_count >= user.usage_limit:
                raise HTTPException(status_code=429, detail="API usage limit exceeded")
            increment_usage(user.id)
            return user
    
    # Try JWT token
    if credentials:
        token_data = decode_token(credentials.credentials)
        if token_data and token_data.user_id:
            user = get_user_by_id(token_data.user_id)
            if user:
                if not user.is_active:
                    raise HTTPException(status_code=403, detail="User account is disabled")
                return user
    
    raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current admin user."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[User]:
    """
    Optional authentication - returns None if not authenticated.
    """
    try:
        return await get_current_user(credentials, api_key)
    except HTTPException:
        return None

