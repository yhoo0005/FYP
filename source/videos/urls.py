from django.urls import path
from videos import views
from django.conf.urls.static import static
from django.conf import  settings
from .views import showvideo


app_name = 'videos'

urlpatterns = [
    path('search-video/', views.VideosView, name='searchbar'),
    path('upload/',showvideo,name='upload'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)