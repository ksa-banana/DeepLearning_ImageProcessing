from bs4 import BeautifulSoup
from urllib.request import urlopen, Request


# 웹페이지의 소스 가져오기
url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=image&query=tnr+%EA%B3%A0%EC%96%91%EC%9D%B4&oquery=tnr+cat&tqi=UySqKsprvmZss6H1RhVssssssAs-123656#imgId=blog113772171%7C16%7C220840188487_1858063148&vType=rollout'
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
fp = urlopen(req)
source = fp.read()
fp.close()


# 소스에서 rg_i Q4LuWd 클래스의 하위 소스 가져오기
soup = BeautifulSoup(source, "html.parser")
soup = soup.find('a', class_='thumb_thumb')


# 이미지 경로를 받아온다.
imgUrl = soup.find["src"]
