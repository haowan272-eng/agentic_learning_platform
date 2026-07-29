"""用户 ORM 模型。

用户是系统的认证主体，通过 bcrypt 哈希密码存储。
ADMIN_USERNAMES 策略集中判断管理员身份，无需本表冗余 is_admin 列。
"""
from sqlalchemy import Column, Integer, String

from app.core.database import Base


class User(Base):
    """系统用户。

    属性：
        id:       自增主键
        username: 唯一用户名，用于登录和 token 签发
        password: bcrypt 哈希后的密码（不存储明文）
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
