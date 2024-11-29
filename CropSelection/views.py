from datetime import datetime, timedelta
import random

import jwt
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from databasedesignai import settings

import os
import joblib
import sklearn
import numpy as np


# Create your views here.
class CropSelection(APIView):
    def post(self, request):
        model = joblib.load(os.path.join(settings.MEDIA_ROOT, 'models/Linear_Regression_Crop_Selection.pkl'))

        N = float(request.data.get('N'))
        P = float(request.data.get('P'))
        K = float(request.data.get('K'))
        temperature = float(request.data.get('temperature'))
        humidity = float(request.data.get('humidity'))
        ph = float(request.data.get('ph'))
        rainfall = float(request.data.get('rainfall'))

        results = model.predict_proba([[N, P, K, temperature, humidity, ph, rainfall]])

        sorted_indices = np.argsort(results[0])[::-1][:3]

        probs = results[0][sorted_indices]
        crops = model.classes_[sorted_indices]

        data = {}
        for c, p in zip(crops, probs):
            data[c] = p

        # response = Response({'crops': crops, 'probs': probs, 'init_env': serializer}, status=status.HTTP_201_CREATED)
        response = Response(data, status=status.HTTP_201_CREATED)

        return response
