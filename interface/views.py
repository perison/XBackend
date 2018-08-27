# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
from django.views.decorators.csrf import csrf_exempt
from interface.models import Wagers
from dss.Serializer import serializer

# Create your views here.

def response_as_json(data, foreign_penetrate=False):
    jsonString = serializer(data=data, output_type="json", foreign=foreign_penetrate)
    response = HttpResponse(
            # json.dumps(dataa, cls=MyEncoder),
            jsonString,
            content_type="application/json",
    )
    response["Access-Control-Allow-Origin"] = "*"
    return response


def json_response(data, code=200, foreign_penetrate=False, **kwargs):
    data = {
        "code": code,
        "msg": "成功",
        "data": data,
    }
    return response_as_json(data, foreign_penetrate=foreign_penetrate)


def json_error(error_string="", code=500, **kwargs):
    data = {
        "code": code,
        "msg": error_string,
        "data": {}
    }
    data.update(kwargs)
    return response_as_json(data)

JsonResponse = json_response
JsonError = json_error

@csrf_exempt
def getWagerList(request):
    result = serializer(Wagers.objects.all())
    print result
    # return HttpResponse('getWagers OK');
    return JsonResponse(result)

@csrf_exempt
def addWager(request):
    if request.method == 'POST' :
        w = Wagers(wager=request.POST.get('wager',''),wid=request.POST.get('wid',''))
        w.save()
        return HttpResponse('addWager OK');
    else :
        return HttpResponse('GET mothod will do nothing');

@csrf_exempt
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

@csrf_exempt
def deleteWager(request):
    if request.method == 'POST' :
        wid = request.POST.get('wid','')
        w = Wagers.objects.get(wid=wid)
        w.delete()
        return HttpResponse('deleteWager OK');
    else :
        return HttpResponse('GET mothod will do nothing');