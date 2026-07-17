"""用户路由：个人信"""
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user

router = APIRouter(tags=["用户"])


@router.get("/profile")
def profile(current_user: str = Depends(get_current_user)):
    """获取当前用户信息（需登录"""
    return {"username": current_user, "status": "active"}
