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
        w = Wagers(wager=request.POST.get('wager',''),wid=request.POST.get('wid',''))
        w.save()
        return HttpResponse('addWager OK');
    else :
        return HttpResponse('GET mothod will do nothing');

def updateWager(request):
    if request.method == 'POST' :
        wid = request.POST.get('wid','')
        wager = request.POST.get('wager','')
        w = Wagers.objects.get(wid=wid)
        w.wager = wager
        w.save(update_fields=['wager'])
        return HttpResponse('updateWager OK');
    else :
        return HttpResponse('GET mothod will do nothing');

def deleteWager(request):
    if request.method == 'POST' :
        wid = request.POST.get('wid','')
        w = Wagers.objects.get(wid=wid)
        w.delete()
        return HttpResponse('deleteWager OK');
    else :
        return HttpResponse('GET mothod will do nothing');