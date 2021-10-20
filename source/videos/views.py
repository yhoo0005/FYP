from django.shortcuts import render
from .models import Video
from .forms import VideoForm

'''
import os
from os import listdir
from os.path import join
from os.path import isfile

import keras
import librosa
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.views.generic import ListView
from django.views.generic import TemplateView
from django.views.generic.edit import CreateView
from rest_framework import APIView
from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.parsers import FormParser
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.renderers import TemplateHTMLRenderer

from App.models import FileModel
from App.serialize import FileSerializer
'''

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


def uploadvideo(request):
    form= VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    context= {'form': form}
    
    return render(request, 'videos/upload.html', context)


def showvideo(request):

    allVideo= Video.objects.all()

    #videofile= allVideo.videofile

    context = {'videofile': allVideo}

    return render(request, 'videos/select_prediction.html', context)

'''
class Predict(views.APIView):
    """
    This class is used to making predictions.

    Example of input:
    {'filename': '01-01-01-01-01-01-01.wav'}

    Example of output:
    [['neutral']]
    """

    template_name = 'index.html'
    # Removing the line below shows the APIview instead of the template.
    renderer_classes = [TemplateHTMLRenderer]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = 'Emotion_Voice_Detection_Model.h5'
        self.graph = tf.get_default_graph()
        self.loaded_model = keras.models.load_model(os.path.join(settings.MODEL_ROOT, model_name))
        self.predictions = []

    def file_elaboration(self, filepath):
        """
        This function is used to elaborate the file used for the predictions with librosa.
        :param filepath:
        :return: predictions
        """
        data, sampling_rate = librosa.load(filepath)
        try:
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate,
                                                 n_mfcc=40).T, axis=0)
            training_data = np.expand_dims(mfccs, axis=2)
            training_data_expanded = np.expand_dims(training_data, axis=0)
            numpred = self.loaded_model.predict_classes(training_data_expanded)
            self.predictions.append([self.classtoemotion(numpred)])
            return self.predictions
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

    def post(self, request):
        """
        This method is used to making predictions on audio files
        loaded with FileView.post
        """
        with self.graph.as_default():
            filename = request.POST.getlist('file_name').pop()
            filepath = str(os.path.join(settings.MEDIA_ROOT, filename))
            predictions = self.file_elaboration(filepath)
            try:
                return Response({'predictions': predictions.pop()}, status=status.HTTP_200_OK)
            except ValueError as err:
                return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

    @staticmethod
    def classtoemotion(pred):
        """
        This method is used to convert the predictions (int) into human readable strings.
        ::pred:: An int from 0 to 7.
        ::output:: A string label

        Example:
        classtoemotion(0) == neutral
        """

        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label
'''





