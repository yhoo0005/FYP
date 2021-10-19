from django.urls import path
from videos import views
from django.conf.urls.static import static
from django.conf import  settings
from .views import uploadvideo, showvideo


app_name = 'videos'

urlpatterns = [
    path('search-video/', views.VideosView, name='searchbar'),
    path('upload/', uploadvideo,name='upload'),
    path('display', showvideo, name='display'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)