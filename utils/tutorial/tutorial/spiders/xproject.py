# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import scrapy


class XprojectSpider(scrapy.Spider):
    name = 'xproject'
    # allowed_domains = ['acf1.qr68.us']
    # start_urls = ['http://acf1.qr68.us/']
    allowed_domains = ['baidu.com']
    start_urls = ['baidu.com']

    def parse(self, response):
        filename = 'xproject.html'
        open(filename,'w').write(response.body)

