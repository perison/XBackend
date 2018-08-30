#-*- coding:utf-8 -*-
import urllib
import urllib2
from urlparse import urlparse
import requests
import requests.utils
import re
import sys
sys.path.append("libs")

KEY = '8pj39'
USER = 'poiu0099'
PASSWORD = 'As14258@'
URL = 'http://acf1.qr68.us/'
# URL = 'http://httpbin.org/get'
SUBMIT_BT = u'搜索'

def compare():
    first_url = URL  # 设置请求的URL
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Content-Type': 'application/x-www-form-urlencoded'}  # 设置请求的信息头
    s = requests.session()  # 获取会话对象
    first_html = s.get(first_url,headers=headers)  # 加载目标URL
    print ('first_html.text', first_html.text)

def action():
    first_url = URL  # 设置请求的URL
    host = urlparse(first_url).hostname
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
             'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
             'Accept-Encoding': 'gzip, deflate',
             'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
             'Connection': 'keep-alive',
             'Upgrade-Insecure-Requests': '1',
             'Content-Type':'application/x-www-form-urlencoded'}#设置请求的信息头
    headers['host'] = host

    s = requests.session()  # 获取会话对象

    first_html = s.get(first_url, headers=headers) #加载目标URL
    print ('first_html.text', first_html.text)
    print ('headers after get first_html', s.headers)

    first_form_data = {'code': KEY, 'submit_bt': SUBMIT_BT}
    headers['Origin'] = URL
    headers['Referer'] = URL+'/'
    # test_url = 'http://httpbin.org/post'
    real_page = s.post(first_url, data=first_form_data, headers=headers) #发起表单请求输入key进入真正有内容的页面
    first_cookies = s.cookies.get_dict() #获取第一个cookies
    print ('real_page.status_code', real_page.status_code) #请求的状态码
    print ('real_page.url', real_page.url) #请求成功后跳转页面的URL
    print ('real_page.text', real_page.text) #请求成功后跳转页面的URL
    print ('first_cookies', first_cookies) #请求成功后跳转页面的URL

    #response=s.get(afterUrl,cookies=s.cookies,headers=headers)#获取登录后页面的内容，其中afterUrl为登陆后可见的URL

if __name__ == "__main__":
    # compare()
    action()