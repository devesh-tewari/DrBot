import requests
from bs4 import BeautifulSoup 
import json

def generateQuery(location, search_tag, query_type, page=None):
    tag_arr = search_tag.split(' ');
    url = ''
    if query_type=='Hospital':
        modified_search_tag = '%20'.join(tag_arr);
        url = 'https://www.practo.com/search?results_type=hospital&q=%5B%7B%22word%22%3A%22'+modified_search_tag+'%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22hospital_name%22%7D%5D&city='+location
    elif query_type=='Clinic':
        modified_search_tag = '%20'.join(tag_arr);
        url = 'https://www.practo.com/search?results_type=clinic&q=%5B%7B%22word%22%3A%22'+modified_search_tag+'%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22practice_name%22%7D%5D&city='+location
    elif query_type=='Doctor':
        modified_search_tag = '%20and%20'.join(tag_arr);
        url = 'https://www.practo.com/search?results_type=doctor&q=%5B%7B%22word%22%3A%22'+modified_search_tag+'%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city='+location
    elif query_type=='Doctor Name':
        modified_search_tag = '%20'.join(tag_arr);
        url = 'https://www.practo.com/search?results_type=doctor&q=%5B%7B"word"%3A"'+modified_search_tag+'"%2C"autocompleted"%3Atrue%2C"category"%3A"doctor_name"%7D%5D&city='+location
    
    if page != None:
        url = url+'&page='+str(page)
    return url

def scrap_data(location, search_tag, query_type, page_num=0):
    scrapped_data = []
    URL = ""
    if page_num != 0:
        URL = generateQuery(location, search_tag, query_type, page_num)
    else:
        URL = generateQuery(location, search_tag, query_type)
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html5lib')
    for link in soup.find_all("script"):
        if link.get('type') == 'application/ld+json':
            query_data = json.loads(link.text)
            if 'name' in query_data.keys() and '@type' in query_data.keys() and query_data['@type'] != 'LocalBusiness':
                scrapped_data.append(query_data)
                print(query_data['name'])
    return scrapped_data