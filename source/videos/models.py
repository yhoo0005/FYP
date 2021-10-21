from .validators import validate_video
from django.db import models
# from django.contrib.auth.models import User

'''
class Video(models.Model):
    name = models.CharField(max_length=500)
    description = models.TextField()
    videofile = models.FileField(
        upload_to='videos/', validators=[validate_video])

    def __str__(self):
        return self.name + ": " + str(self.videofile)
'''


class Video(models.Model):
    videofile= models.FileField(upload_to='videos/', null=True, verbose_name="")

    def __str__(self):
        return str(self.videofile)