# -*- coding:utf-8 -*-
from urlparse import urlparse
import requests
import requests.utils
import re
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append("libs")

KEY = '8pj39'
USER = 'poiu0099'
PASSWORD = 'As14258@'
URL = 'http://acf1.qr68.us/'
# URL = 'http://httpbin.org/get'
SUBMIT_BT = u'搜索'


def compare():
	file = open("real_page.html", "r")
	all_line_txt = file.readlines()  # 读所有行
	file.close()
	# link_list = re.findall(r"(?<=href=\").+?(?=\")|(?<=href=\').+?(?=\')", all_line_txt[0])
	text =''.join(all_line_txt)
	link_list = re.findall(r"<a.*?href=\"http.*?<\/a>",text,re.I)
	for link in link_list:
		print 'url',link.split('"')[1]

def action():
	first_url = URL  # 设置请求的URL
	host = urlparse(first_url).hostname
	headers = {
		'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
		'Accept-Encoding': 'gzip, deflate',
		'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
		'Connection': 'keep-alive',
		'Upgrade-Insecure-Requests': '1',
		'Content-Type': 'application/x-www-form-urlencoded'}  # 设置请求的信息头
	headers['host'] = host
	
	s = requests.session()  # 获取会话对象
	s.headers.update(headers)
	print '(1)s.headers',s.headers
	
	# first_page = s.get(first_url, headers=headers)  # 加载目标URL
	first_page = s.get(first_url)
	first_page.encoding = 'utf-8'
	print ('first_page.text', first_page.text)
	print ('headers after get first_page', s.headers)
	
	first_form_data = {'code': KEY, 'submit_bt': SUBMIT_BT}
	headers['Origin'] = URL
	headers['Referer'] = URL + '/'
	# test_url = 'http://httpbin.org/post'
	real_page = s.post(first_url, data=first_form_data, headers=headers)  # 发起表单请求输入key进入真正有内容的页面
	first_cookies = real_page.cookies.get_dict()  # 获取第一个cookies
	if first_cookies :
		s.cookies.update(real_page.cookies)
		temp_cookies = []
		for key,value in first_cookies.iteritems():
			temp_cookies.append(str(key) + '=' + str(value))
		headers['Cookie'] = ','.join(temp_cookies)
		s.headers.update(headers)
	# print ('real_page.status_code', real_page.status_code)  # 请求的状态码
	# print ('real_page.url', real_page.url)  # 请求成功后跳转页面的URL
	# print ('real_page.text', real_page.text)  # 请求成功后跳转页面的URL
	# print ('first_cookies', first_cookies)  # 请求成功后跳转页面的URL

	# # 打开文件
	# fo = open("real_page.html", "w")
	# print "文件名为: ", fo.name
	# fo.write(real_page.text)
	# # 关闭文件
	# fo.close()
	
	link_list = re.findall(r"<a.*?href=\"http.*?<\/a>", real_page.text, re.I)
	login_link = link_list[0].split('"')[1]
	# print 'target_link',login_link
	
	print '(2)s.headers',s.headers
	# login_page = s.get(login_link, headers=headers)
	login_page = s.get(login_link)
	print ('login_page.text', login_page.text)
	second_cookies = s.cookies.get_dict()
	print ('second_cookies', second_cookies)
	
	
	# response=s.get(afterUrl,cookies=s.cookies,headers=headers)#获取登录后页面的内容，其中afterUrl为登陆后可见的URL


if __name__ == "__main__":
	# compare()
	action()
