# -*- coding:utf-8 -*-
from urlparse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import json,time
from collections import OrderedDict
import requests
import re,random
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append("libs")

def write_html(text):
    # 打开文件
    fo = open("test.html", "w")
    print ("文件名为: ", fo.name)
    fo.write(text)
    # 关闭文件
    fo.close()

def compare():
    # # 打开文件
    # fo = open("real_page.html", "w")
    # print "文件名为: ", fo.name
    # fo.write(real_page.text)
    # # 关闭文件
    # fo.close()
    
    headers = {'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
               'Upgrade-Insecure-Requests': '1',
               'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
               # 'javascriptEnabled': True,
               'Accept-Encoding': 'gzip, deflate',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
               # 'platform': 'ANY',
               # 'browserName': 'phantomjs',
               # 'version': '',
               # 'Host': u'pc4.1ll11.com',
               'Cookie': 'SafeCode=activated;AC=86467c19f776574db849b8cea55c5ca8|1535889010;PHPSESSID=747a5d52ef_1040020_6706',
               'Referer': u'http://pc4.1ll11.com/sscgp53745f/user/login.html.auth'
               }
    
    pt = PhantomJs(headers)
    pt.browse('https://www.bejson.com/')
    print ('page_source1',pt.instance().page_source)
    print (type(pt.instance().page_source))
    submit_button = pt.instance().find_element_by_id('validate')
    submit_button.send_keys('abcd')
    print ('page_source2',pt.instance().page_source)

    pt.quite()

def choose_link(cookie_box, links) :
    print 'choose_link'
    for link in links :
        print 'choose_link_link',link
        if cookie_box.has_key(urlparse(link).hostname):
            if cookie_box[urlparse(link).hostname].find('visid') == -1 and cookie_box[urlparse(link).hostname].find('SafeCode') != -1:
                return link
    return '' #找不到合适的就返回空字符串

class PhantomJs(object):
    def __init__(self,headers):
        for key, value in headers.iteritems() :
            capability_key = 'phantomjs.page.customHeaders.{}'.format(key)
            webdriver.DesiredCapabilities.PHANTOMJS[capability_key] = value
        dcap = dict(DesiredCapabilities.PHANTOMJS)
        print 'dcap', dcap
        self.__browser = webdriver.PhantomJS(desired_capabilities=dcap)
        
    def instance(self):
        return self.__browser
    
    def getResponseHeaders(self) :
        har = json.loads(self.__browser.get_log('har')[0]['message'])
        return OrderedDict(
            sorted([(header["name"], header["value"]) for header in har['log']['entries'][0]['response']["headers"]],
                   key=lambda x : x[0]))

    def getResponseStatus(self) :
        har = json.loads(self.__browser.get_log('har')[0]['message'])
        return (har['log']['entries'][0]['response']["status"],str(har['log']['entries'][0]['response']["statusText"]))

    def browse(self,url):
        print ('browse url',url)
        self.__browser.get(url)
        # print ('page_source', self.__browser.page_source)
    
    def quite(self):
        self.__browser.quit()


class Action(object) :
    def __init__(self, user, password) :
        self.user = user
        self.password = password
        self.key = '8pj39'
        self.url = 'http://acf1.qr68.us/'
        self.text = ''
        self.verify_value = ''
        self.verify_code = ''
        self.systemversion = ''
        self.submit_bt = u'搜索'
        self.headers = {
            'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
            'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding' : 'gzip, deflate',
            'Accept-Language' : 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
            'Connection' : 'keep-alive',
            'Upgrade-Insecure-Requests' : '1',
            'Content-Type' : 'application/x-www-form-urlencoded'}  # 设置请求的信息头
        self.cookies_box = dict()
        self.session = requests.session()  # 获取会话对象
    
    def goto_first_page(self) :  # 第一步，请求URL
        
        host = urlparse(self.url).hostname
        self.headers['host'] = host  # 每一次get或者post之前，都需要更新目标host
        self.session.headers.update(self.headers)
        print 'goto_first_url headers', self.session.headers
        
        first_page = self.session.get(self.url)
        first_page.encoding = 'utf-8'
        print ('goto_first_url first_page.text', first_page.text)
        
        self.url = first_page.url
        self.text = first_page.text
        return 0
    
    def post_data_to_first_page(self) :  # 第二步，输入KEY,刷新页面，获取线路列表
        first_form_data = {'code' : self.key, 'submit_bt' : self.submit_bt}
        
        self.headers['Origin'] = self.url  # 只需要设置一次
        
        self.headers['Referer'] = self.url  # 每次请求之前都需要更新
        host = urlparse(self.url).hostname
        self.headers['host'] = host  # 每一次get或者post之前，都需要更新目标host
        self.session.headers.update(self.headers)
        print 'post_data_to_first_page headers 1', self.session.headers
        
        real_first_page = self.session.post(self.url, data=first_form_data, headers=self.headers)
        self.url = real_first_page.url
        self.text = real_first_page.text
        first_cookies = real_first_page.cookies.get_dict()  # 获取第一个cookies
        if first_cookies :
            self.session.cookies.update(real_first_page.cookies)
            temp_cookies = []
            for key, value in first_cookies.iteritems() :
                temp_cookies.append(str(key) + '=' + str(value))
            self.cookies_box[urlparse(self.url).hostname] = ';'.join(temp_cookies)
            # self.headers['Cookie'] = ','.join(temp_cookies)
            # self.session.headers.update(self.headers)
            print 'post_data_to_first_page headers 2', self.session.headers
            return 0
        else :
            return 1
    
    def set_cookies_in_first_page(self) :  # 第三步，获取页面的set.php列表，逐个访问获取cookies
        print 'set_cookies_in_first_page'
        text = ''.join(self.text)
        # print 'set_cookies_in_first_page_text',text
        p = r'http\S+?set\.php'
        pattern = re.compile(p)
        link_list = pattern.findall(text)
        
        temp_headers = {
            'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
            'Accept' : 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer' : 'http://acf1.qr68.us/',
            'Accept-Encoding' : 'gzip, deflate',
            'Accept-Language' : 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7'}
        
        for link in link_list :
            print 'link', link
            host = urlparse(link).hostname
            temp_headers['host'] = host  # 每一次get或者post之前，都需要更新目标host
            print 'temp_headers', temp_headers
            try :
                page = requests.get(link, headers=temp_headers)
            except requests.exceptions.RequestException as e :
                print e
                continue
            print 'page.status_code', page.status_code
            cookies = page.cookies.get_dict()
            print 'page.cookies', cookies
            temp_cookies = []
            for key, value in cookies.iteritems() :
                temp_cookies.append(str(key) + '=' + str(value))
            self.cookies_box[urlparse(link).hostname] = ','.join(temp_cookies)
            if cookies.has_key('SafeCode') :
                break
        print 'cookies_box', self.cookies_box
        return 0
    
    def goto_login_page(self) :  # 第四步，筛选cookies为SafeCode的链接，访问登录界面，需要访问两次
        link_list = re.findall(r"<a.*?href=\"http.*?<\/a>", self.text, re.I)
        index = 0
        for link in link_list:
            link_list[index] = link.split('"')[1]
            index += 1
        login_link = choose_link(self.cookies_box,link_list)
        
        if login_link == '':
            return -1
        print 'login_link',login_link
        
        host = urlparse(login_link).hostname
        self.headers['host'] = host
        self.headers['Referer'] = self.url
        self.headers['Cookie'] = self.cookies_box[host]
        self.session.headers.update(self.headers)
        
        login_page = self.session.get(login_link,headers=self.headers)
        self.url = login_page.url
        self.text = login_page.text
        print ('login_page.text', login_page.text)
        write_html(login_page.text)
        login_page_cookies = login_page.cookies.get_dict()
        print ('login_page_cookies', login_page_cookies)
        
        return 0
    
    def get_verifycode(self):
        temp_headers = {
            'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
            'Accept' : '*/*',
            'Accept-Encoding' : 'gzip, deflate',
            'Accept-Language' : 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7'}
        host = urlparse(self.url).hostname
        temp_headers['Host'] = host
        temp_headers['Referer'] = self.url

        p = r'systemversion\s*=\s*".+"\s*;'
        pattern = re.compile(p)
        self.systemversion = pattern.findall(self.text)[0].split('"')[1]
        code_info = requests.get('http://' + host + '/getCodeInfo/.auth?u=' + str(random.random()) + '&systemversion=' + self.systemversion + '&.auth',headers=temp_headers)
        print 'code_info.text',code_info.text
        t = code_info.text.split('_')
        getVcode_t = t[0]
        print 'getVcode_t',getVcode_t
        self.verify_value = t[1]
        print 'verify_value',self.verify_value
        
        temp_headers['Accept'] = 'image/webp,image/apng,image/*,*/*;q=0.8'
        img = requests.get('http://' + host + '/getVcode/.auth?t=' + getVcode_t + '&systemversion=' + self.systemversion + '&.auth',headers=temp_headers)
        if img.status_code == 200 :
            file = open('./trainImage/verifycode_'+str(random.randint(0,10000))+'.jpeg', 'wb')
            file.write(img.content)
            file.close()
        else :
            return 1
        return 0
    
    def login(self):
        self.headers = {
            'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
            'Accept' : '*/*',
            'Accept-Encoding' : 'gzip, deflate',
            'Accept-Language' : 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7'
            # 'Content-Type' : 'application/x-www-form-urlencoded'
        }
        host = urlparse(self.url).hostname
        self.headers['Host'] = host
        self.headers['Referer'] = self.url
        self.headers['Cookie'] = self.cookies_box[host]

        soup = BeautifulSoup(self.text, features="html5lib")
        s = soup.find("input", {"name" : "cname"})
        cname = s.attrs['value']
        s = soup.find("input", {"name" : "cid"})
        cid = s.attrs['value']
        
        self.verify_code = raw_input('input verify_code:')
        
        self.go_with_phantomjs(self.headers,self.url)
        return 1
        
        form_data = {'isSec' : '0', 'cid' : cid, 'cname' : cname, 'systemversion' : self.systemversion,
                     'VerifyCode' : self.verify_code, '__VerifyValue' : self.verify_value, 'password' : self.password,
                     '__name' : self.user}
        
        redirect_info = self.session.post('http://' + host + '/loginVerify/.auth',form_data,headers=self.headers) #返回值是可以跳转的url的一部分
        login_cookies = redirect_info.cookies.get_dict() #获取到最终需要的cookies!!!
        if login_cookies :
            self.session.cookies.update(redirect_info.cookies)
            temp_cookies = []
            for key, value in login_cookies.iteritems() :
                temp_cookies.append(str(key) + '=' + str(value))
            temp_cookies_format = ';'.join(temp_cookies)
            self.cookies_box[host] += ';' + temp_cookies_format
            print 'the most important cookies',self.cookies_box[host]
        else :
            return -1
        
        self.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
        self.headers['Upgrade-Insecure-Requests'] = '1'
        self.headers['Cookie'] = self.cookies_box[host]
        self.session.headers.update(self.headers)
        
        redirect_url = redirect_info.text.split('\n')[1].replace('host',host) #替换类似http://host/sscgp53745f/login/632bea9f9a_rdsess/k中的host
        print 'redirect_url',redirect_url
        
        #http://pc4.1ll11.com/sscgp53745f_6706/index.htm?20902_20903_4.6.trunk_20150316
        target_url = 'http://' + host + redirect_info.text.split('\n')[0] + 'index.htm?20902_20903_4.6.trunk_20150316'
        print 'target_url',target_url
        
        print 'last headers',self.headers

        pt = PhantomJs(self.headers)
        pt.browse(redirect_url)
        # pt.browse('http://httpbin.org/get')
        print pt.getResponseStatus()
        print pt.getResponseHeaders()

        pt.browse(target_url)
        # pt.browse('http://httpbin.org/get')
        print pt.getResponseStatus()
        print pt.getResponseHeaders()

        pt.quite()
        
        
        # redirect_page = self.session.get(redirect_url, headers = self.headers,allow_redirects=False)
        # print 'redirect_page.status_code',redirect_page.status_code  # 302
        # print 'redirect_page.url',redirect_page.url
        # print 'redirect_page.text',redirect_page.text
        # print redirect_page.headers

        # if last_page.status_code == 200 and last_page.text.find('赛车') != -1 :
        #     return 0
        
        return 1
    
    def get_with_requests(self,headers):
        pass
    
    def post_with_requests(self,headers,data):
        pass
    
    def go_with_phantomjs(self,headers,url):
        pt = PhantomJs(headers)
        pt.browse(url)
        print ('page_source',(pt.instance().page_source))
        # print pt.getResponseStatus()
        print ('response headers',pt.getResponseHeaders())
        
        time.sleep(10)
        username_input = pt.instance().find_element_by_name('__name')
        password_input = pt.instance().find_element_by_name('password')
        verifycode_input = pt.instance().find_element_by_name('VerifyCode')
        checked_input = pt.instance().find_element_by_id('sec')
        submit_button = pt.instance().find_element_by_name('submit')
        
        username_input.send_keys(self.user)
        password_input.send_keys(self.password)
        verifycode_input.send_keys(self.verify_code)
        checked_input.click()
        submit_button.click()
        
        time.sleep(3)
        
        print ('page after submit',pt.instance().page_source)
        print ('cookies after submit',pt.instance().get_cookies())
    
        pt.quite()
        

def combine():
    user = 'ee222222'
    password = 'Po2......'
    a = Action(user, password)
    if a.goto_first_page() == 0 :
        if a.post_data_to_first_page() == 0 :
            if a.set_cookies_in_first_page() == 0 :
                if a.goto_login_page() == 0 :
                    if a.get_verifycode() == 0 :
                        if a.login() == 0 :
                            print 'ok'
                        else :
                            print 'shit6'
                    else :
                        print 'shit5'
                else :
                    print 'shit4'
            else :
                print 'shit3'
        else :
            print 'shit2'
    else :
        print 'shit1'

if __name__ == "__main__" :
    compare()
    # combine()
