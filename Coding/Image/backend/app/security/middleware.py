from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        mode = (settings.MODE or "local").strip().lower()
        if mode == "host":
            # Allow Hugging Face to embed backend swagger/pages in spaces
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data:; "
                "frame-ancestors 'self' https://huggingface.co https://*.hf.space;"
            )
        else:
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data:; "
                "frame-ancestors 'self';"
            )
            
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

