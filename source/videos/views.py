from django.shortcuts import render
from .models import Video
from .forms import VideoForm


def VideosView(request):

    searchvalue = ''

    form = request.GET.get('search')
    #VideoSearchForm(request.POST or None)
    #if form.is_valid():
        #searchvalue = form.cleaned_data.get("search")

    searchresults = Video.objects.filter(name__icontains=form)

    context = {'form': form,
               'searchresults': searchresults,
               }

    return render(request, 'videos/search_video.html', context)


def showvideo(request):

    #lastvideo= Video.objects.last()

    #videofile= lastvideo.videofile


    form= VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    
    context= {#'videofile': videofile,
              'form': form
              }
    
      
    return render(request, 'videos/upload.html', context)
