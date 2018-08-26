# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Wagers(models.Model):
    wid = models.IntegerField(primary_key=True)
    wager = models.CharField(max_length=5)

# Create your models here.
