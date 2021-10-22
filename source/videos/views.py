from django.http import request
from django.shortcuts import render
from .models import Video
from .forms import VideoForm
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.db import models

import os
import torch
import librosa
import numpy as np
from torch import nn
import moviepy.editor as mp
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

import subprocess
import math
from pydub import AudioSegment
from time import strftime, gmtime
import os, shutil

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

AUDIO_ONLY_PATH = str(os.path.join(settings.MEDIA_ROOT, "audio/audioOnly/"))
AUDIO_CHUNKS_PATH = str(os.path.join(settings.MEDIA_ROOT, "audio/audioOnly/audio_chunks/"))
class SplitWavAudio():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def single_split(self, from_sec, to_sec):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        filename = strftime("%H-%M-%S", gmtime(from_sec)) + '_' + strftime("%H-%M-%S", gmtime(to_sec)) + '.wav'
        path = AUDIO_CHUNKS_PATH + filename
        split_audio.export(path, format='wav')
        
    def multiple_split(self, sec_per_split):
        total_secs = int(self.audio.duration_seconds)
        num = deepcopy(total_secs)
        for i in range(0, num, sec_per_split):
            self.single_split(i, i+sec_per_split)
            print(str(i) + ' to ' + str(i+sec_per_split) + ' done')
            if i + sec_per_split > total_secs:
                print('All splited successfully')

SAMPLE_RATE = 48000

def get_waveforms(path):
    """
    load all audio file from path
    load full 5 seconds of the audio file; nativve sample rate = 48k
    """
    waveforms = [] # waveforms to augment later
    file_count = 0
    p = deepcopy(path)
    for root, subdr, files in os.walk(p):
        fs = deepcopy(files)
        r = deepcopy(root)
        for file in fs:
            f = deepcopy(file)
            X, sample_rate = librosa.load(os.path.join(r, f), duration=3, res_type='kaiser_fast', sr=SAMPLE_RATE)

            waveform = np.zeros((SAMPLE_RATE*3,))
            waveform[:len(X)] = X

            waveforms.append(waveform)

            file_count += 1
            # keep track of data loader's progress
            print('\r'+f' Processed {file_count} audio samples',end='')

    return waveforms


def get_features(waveforms):
    features = []
    file_count = 0
    wfs = deepcopy(waveforms)
    for waveform in wfs:
        wf = deepcopy(waveform)
        mfccs = librosa.feature.mfcc(wf,
                                     sr=SAMPLE_RATE,
                                     n_mfcc=40,
                                     n_fft=1024,
                                     win_length=512,
                                     window='hamming',
                                     n_mels=128,
                                     fmax=SAMPLE_RATE/2)
        features.append(mfccs)
        file_count += 1
        # print progress 
        print('\r'+f' Processed {file_count} waveforms',end='')

    return features


class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self,num_emotions):
        super().__init__() 
        
        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer 
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        
        # define single transformer encoder layer
        # self-attention + feedforward network from "Attention is All You Need" paper
        # 4 multi-head self-attention layers each with 40-->512--->40 feedforward network
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40, # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            nhead=4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dropout=0.4, 
            activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        
        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock2 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor 
        #    from parallel 2D convolutional and transformer blocks, output 8 logits 
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array 
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions 
        self.fc1_linear = nn.Linear(512*2+40,num_emotions) 
        
        ### Softmax layer for the 8 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding
        
    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self,x):
        
        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        # create final feature embedding from 1st convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding1 = self.conv2Dblock1(x) # x == N/batch * channel * freq * time
        
        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 
        
        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        # create final feature embedding from 2nd convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding2 = self.conv2Dblock2(x) # x == N/batch * channel * freq * time
        
        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1) 
        
         
        ########## 4-encoder-layer Transformer block w/ 40-->512-->40 feedfwd network ##############
        # maxpool input feature map: 1*40*282 w/ 1*4 kernel --> 1*40*70
        x_maxpool = self.transformer_maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        
        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2,0,1) 
        
        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)
        
        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40
        
        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2,transformer_embedding], dim=1)  

        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)  
        
        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)
        
        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax



# define loss function; CrossEntropyLoss() fairly standard for multiclass problems 
def criterion(predictions, targets): 
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

def delete(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


emotions_dict ={
    '0':'neutral',
    '1':'calm',
    '2':'happy',
    '3':'sad',
    '4':'angry',
    '5':'fearful',
    '6':'disgust',
    '7':'surprised'
    }

# need device to instantiate model
device = 'cpu'

# instantiate model for 8 emotions and move to GPU 
model = parallel_all_you_want(len(emotions_dict)).to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

# pick load folder  
load_folder = str(os.path.join(settings.MEDIA_ROOT, "checkpoint"))

# make full load path
load_path = str(os.path.join(settings.MEDIA_ROOT, "checkpoint/parallel_all_you_wantFINAL-016.pkl"))

## instantiate empty model and populate with params from binary 
model = parallel_all_you_want(len(emotions_dict))
load_checkpoint(optimizer, model, load_path)


"""
This is the main function where we will pass our videos in and return the final results back to videos/display.html.
a variable name of a to q will be used to pass the required infomation back to display.html.
"""
def predict(request):
    filename = request.POST.get('file_name')
    filepath = str(os.path.join(settings.MEDIA_ROOT, str(filename)))
    print(filepath)

    #filepath = str(os.path.join(settings.MEDIA_ROOT, "videos/test.mp4"))
    video_path = filepath
    video = mp.VideoFileClip(video_path)

    
    des = str(os.path.join(settings.MEDIA_ROOT, "audio/audioOnly/audio1.wav"))
    video.audio.write_audiofile(des)
    
    split_wav = SplitWavAudio(AUDIO_ONLY_PATH, 'audio1.wav')
    split_wav.multiple_split(3)

    waveforms = get_waveforms(AUDIO_CHUNKS_PATH)
    features = get_features(waveforms)

    emotions_dict ={
    '0':'neutral',
    '1':'calm',
    '2':'happy',
    '3':'sad',
    '4':'angry',
    '5':'fearful',
    '6':'disgust',
    '7':'surprised'
}

    def predict(X):
        emotions_dict ={
            '0':'neutral',
            '1':'calm',
            '2':'happy',
            '3':'sad',
            '4':'angry',
            '5':'fearful',
            '6':'disgust',
            '7':'surprised'
            }
        #model = parallel_all_you_want(len(emotions_dict))

        model.eval()
        with torch.no_grad():
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
        return predictions

    features = np.array(features)
    print(features.shape)

    X = np.expand_dims(features,1)
    X.shape


    scaler = StandardScaler()
    #### Scale the data ####
    # store shape so we can transform it back 
    N,C,H,W = X.shape
    # Reshape to 1D because StandardScaler operates on a 1D array
    # tell numpy to infer shape of 1D array with '-1' argument
    X = np.reshape(X, (N,-1)) 
    X = scaler.fit_transform(X)
    # Transform back to NxCxHxW 4D tensor format
    X = np.reshape(X, (N,C,H,W))

    delete(str(os.path.join(settings.MEDIA_ROOT, "audio/audioOnly/")))
    delete(str(os.path.join(settings.MEDIA_ROOT, "audio/audioOnly/audio_chunks/")))

    # check shape of X again
    #print(f'X_train scaled:{X.shape}')


    X_tensor = torch.tensor(X, device=device).float()
    predictions = predict(X_tensor)
    predictions = predictions.numpy()
    #print(predictions)

    smooth_predictions = np.copy(predictions)
    for i in range(1, len(smooth_predictions)-2):
        if smooth_predictions[i-1] == smooth_predictions[i+1]:
            if smooth_predictions[i] != smooth_predictions[i-1]:
                smooth_predictions[i] = smooth_predictions[i-1]

    #print(smooth_predictions)
    result = ""
    start_time = 0
    end_time = 0

    list = [[], [], [], [], [], [], [], []]

    previous_mood = smooth_predictions[0]
    cache = 0
    for i in range(len(smooth_predictions)):
        if smooth_predictions[i] == previous_mood:
            end_time += 3
        else:
            previous_mood = smooth_predictions[i]
            list[int(smooth_predictions[cache])].append(str(start_time//60) + ":" + str(start_time%60).zfill(2) +
             "-" + str(end_time//60) + ":" + str(end_time%60).zfill(2) + '   ')
            cache = i
            start_time = end_time
            end_time += 3
        if i == len(smooth_predictions) - 1:
            list[int(smooth_predictions[cache])].append(str(start_time//60) + ":" + str(start_time%60).zfill(2) +
             "-" + str(end_time//60) + ":" + str(end_time%60).zfill(2) + '   ')
    
    j = ""
    k = ""
    l = ""
    m = ""
    n = ""
    o = ""
    p = "" 
    q = ""

    a = "Neutral"
    for i in list[0]:
        j += str(i)

    b = "Calm"
    for i in list[1]:
         k += str(i)

    c = "Happy"
    for i in list[2]:
         l += str(i)

    d = "Sad"
    for i in list[3]:
         m += str(i)

    e = "Angry"
    for i in list[4]:
         n += str(i)

    f = "Fearful"
    for i in list[5]:
         o += str(i)

    g = "Disgust"
    for i in list[6]:
         p += str(i)

    h = "Surprised"
    for i in list[7]:
         q += str(i)

    context = {'predictions': list, 'filename': str(filename), 'smooth':smooth_predictions, 'a':a, 'b':b,
    'c':c, 'd':d, 'e':e, 'f':f, 'g':g, 'h':h, 'j':j, 'k':k, 'l':l, 'm':m, 'n':n, 'o':o, 'p':p, 'q':q} 

    return render(request, 'videos/prediction_result.html', context)










