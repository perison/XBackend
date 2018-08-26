# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
from django.views.decorators.csrf import csrf_exempt
from interface.models import Wagers


# Create your views here.

def show(request):
    return HttpResponse('hello world');

@csrf_exempt
def addWager(request):
    if request.method == 'POST' :
        # return HttpResponse('request:'+request.POST.get('wager',''));
        print 'wager:'+request.POST.get('wager','')
        w = Wagers(wager=request.POST.get('wager',''),wid=request.POST.get('wid',''))
        w.save()
        return HttpResponse('OK');
    else :
        return HttpResponse('GET done');