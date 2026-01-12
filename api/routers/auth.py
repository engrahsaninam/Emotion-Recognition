"""
Authentication Router

Endpoints for user authentication, registration, and API key management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional

from ..auth.security import (
    User, Token, 
    create_tokens, create_user, 
    get_user_by_email, get_user_by_id,
    verify_password, get_password_hash,
    generate_api_key,
    get_current_user, get_current_active_user,
    _users_db, _api_keys_db
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============== Request/Response Models ==============

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    api_key: Optional[str]
    usage_count: int
    usage_limit: int


class APIKeyResponse(BaseModel):
    api_key: str
    message: str


class UsageResponse(BaseModel):
    usage_count: int
    usage_limit: int
    remaining: int
    percentage_used: float


# ============== Endpoints ==============

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """
    Register a new user account.
    
    Returns the user details including a generated API key.
    """
    # Check if email already exists
    existing_user = get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = create_user(
        email=user_data.email,
        password=user_data.password,
        full_name=user_data.full_name
    )
    
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        api_key=user.api_key,
        usage_count=user.usage_count,
        usage_limit=user.usage_limit
    )


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login with email and password.
    
    Returns JWT access and refresh tokens.
    """
    user = get_user_by_email(form_data.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get stored password hash
    user_data = _users_db.get(user.id, {})
    hashed_password = user_data.get("hashed_password", "")
    
    if not verify_password(form_data.password, hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return create_tokens(user)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user's information.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        api_key=current_user.api_key,
        usage_count=current_user.usage_count,
        usage_limit=current_user.usage_limit
    )


@router.get("/usage", response_model=UsageResponse)
async def get_usage(current_user: User = Depends(get_current_active_user)):
    """
    Get current API usage statistics.
    """
    return UsageResponse(
        usage_count=current_user.usage_count,
        usage_limit=current_user.usage_limit,
        remaining=current_user.usage_limit - current_user.usage_count,
        percentage_used=(current_user.usage_count / current_user.usage_limit) * 100
    )


@router.post("/api-key/regenerate", response_model=APIKeyResponse)
async def regenerate_api_key(current_user: User = Depends(get_current_active_user)):
    """
    Regenerate API key for the current user.
    
    The old API key will be invalidated immediately.
    """
    # Remove old API key
    if current_user.api_key:
        _api_keys_db.pop(current_user.api_key, None)
    
    # Generate new API key
    new_api_key = generate_api_key()
    
    # Update user
    if current_user.id in _users_db:
        _users_db[current_user.id]["api_key"] = new_api_key
        _api_keys_db[new_api_key] = current_user.id
    
    return APIKeyResponse(
        api_key=new_api_key,
        message="API key regenerated successfully. The old key is now invalid."
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_active_user)):
    """
    Refresh access token using current authentication.
    """
    return create_tokens(current_user)

