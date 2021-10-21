from django.urls import path
from videos import views
from django.conf.urls.static import static
from django.conf import  settings
from .views import predict, uploadvideo, showvideo


app_name = 'videos'

urlpatterns = [
    path('upload/', uploadvideo,name='upload'),
    path('display/', showvideo, name='display'),
    path('predict/', predict, name='predict'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)