# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Wagers(models.Model):
    id = models.IntegerField
    wager = models.CharField(max_length=5)

# Create your models here.
