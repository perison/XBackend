# -*- coding: utf-8 -*-
import mechanize

key = '8pj39'
USERNAME = 'poiu0099'
PASSWORD = 'As14258@'
URL = 'http://acf1.qr68.us'

# Browser
br = mechanize.Browser()

# options
br.set_handle_equiv(True)
br.set_handle_gzip(True)
br.set_handle_redirect(True)
br.set_handle_referer(True)
br.set_handle_robots(False)

# Follows refresh 0 but not hangs on refresh > 0
br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

# debugging?
br.set_debug_http(True)
br.set_debug_redirects(True)
br.set_debug_responses(True)

# User-Agent (this is cheating, ok?)
br.addheaders = [
    ('Origin',URL),
    ('Referer',URL),
    ('User-Agent','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36')]


def gotoRealUrl():
    br.open(URL)
    br.select_form(nr=0)
    br.form['code'] = key
    br.submit()
    print ('title',br.title())
    new_link = br.click_link(text=u'\u7ebf\u8def1')
    br.open(new_link)
    print ('title', br.title())


if __name__ == "__main__":
    gotoRealUrl()
