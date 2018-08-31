# -*- coding:utf-8 -*-
from urlparse import urlparse
import requests
import requests.utils
import re
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append("libs")


def compare():
	# # 打开文件
	# fo = open("real_page.html", "w")
	# print "文件名为: ", fo.name
	# fo.write(real_page.text)
	# # 关闭文件
	# fo.close()
	
	file = open("real_page.html", "r")
	all_line_txt = file.readlines()  # 读所有行
	file.close()
	# link_list = re.findall(r"(?<=href=\").+?(?=\")|(?<=href=\').+?(?=\')", all_line_txt[0])
	text =''.join(all_line_txt)
	# print 'text',text
	p = r'http\S+?set\.php'
	pattern = re.compile(p)
	link_list = pattern.findall(text)
	# link_list = re.findall(r"^http.*?php$",text,re.I)
	for link in link_list:
		# print 'url',link.split('"')[1]
		print link

@staticmethod
def choose_link(dict,links):
	for link in links:
		for key in dict[link]:
			if key.indexof('visid') == -1 :
				if key.indexof('SafeCode') != -1:
					return link
	return links[1]


class Action():
	KEY = '8pj39'
	USER = 'poiu0099'
	PASSWORD = 'As14258@'
	URL = 'http://acf1.qr68.us/'
	# URL = 'http://httpbin.org/get'
	SUBMIT_BT = u'搜索'
	headers = {
		'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
		'Accept-Encoding': 'gzip, deflate',
		'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
		'Connection': 'keep-alive',
		'Upgrade-Insecure-Requests': '1',
		'Content-Type': 'application/x-www-form-urlencoded'}  # 设置请求的信息头
	session = requests.session()  # 获取会话对象
	
	def goto_first_page(self):
		first_url = self.URL  # 设置请求的URL
		host = urlparse(first_url).hostname
		self.headers['host'] = host
		
		self.session.headers.update(self.headers)
		print 'goto_first_url headers', self.session.headers
		
		# first_page = s.get(first_url, headers=headers)  # 加载目标URL
		first_page = self.session.get(first_url)
		self.headers['Origin'] = self.URL
		self.headers['Referer'] = self.URL + '/'
		first_page.encoding = 'utf-8'
		print ('goto_first_url first_page.text', first_page.text)
		print ('goto_first_url headers after get first_page', self.session.headers)
		return 0
	
	def post_data_to_first_page(self):
		first_form_data = {'code': self.KEY, 'submit_bt': self.SUBMIT_BT}
		real_first_page = self.session.post(self.session.url, data=first_form_data, headers=self.headers)
		first_cookies = real_first_page.cookies.get_dict()  # 获取第一个cookies
		if first_cookies:
			self.session.cookies.update(real_first_page.cookies)
			temp_cookies = []
			for key, value in first_cookies.iteritems():
				temp_cookies.append(str(key) + '=' + str(value))
			self.headers['Cookie'] = ','.join(temp_cookies)
			self.session.headers.update(self.headers)
			return 0
		else :
			return 1
	
	def set_cookies_in_first_page(self):
		text = ''.join(self.session.text)
		print 'set_cookies_in_first_page text',text
		p = r'http\S+?set\.php'
		pattern = re.compile(p)
		link_list = pattern.findall(text)
		cookies_box = dict()
		for link in link_list:
			page = requests.get(link,self.headers)
			cookies = page.cookies.get_dict()
			cookies_box[link] = cookies
			print 'cookies_box',cookies_box
	
	def goto_target_url(self):
		link_list = re.findall(r"<a.*?href=\"http.*?<\/a>", self.session.text, re.I)
		
		login_link = link_list[0].split('"')[1]
		print '(2)s.headers', self.session.headers
		# login_page = s.get(login_link, headers=headers)
		login_page = self.session.get(login_link)
		print ('login_page.text', login_page.text)
		second_cookies = self.session.cookies.get_dict()
		print ('second_cookies', second_cookies)

if __name__ == "__main__":
	compare()
	# action()
