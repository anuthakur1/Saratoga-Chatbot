# Crawl a set of domains, chunk up the URLs that were crawled, and index
# the content in a vector db.

# Two variables control the crawl - max_urls_per_domain can be modified in Crawler::__init__
# and controls the number of urls to index per domain. The urls list in __main__ lists the
# set of domains to crawl.

# To run:
# python createindex.py

import langchain
from langchain.vectorstores import FAISS

import os
import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import re
import langchain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class Crawler:
    
    def get_domain(self, url):
        """Gets the domain of a URL.

        Args:
            url (str): The URL.

        Returns:
            str: The domain of the URL.
        """

        match = re.match(r"https?://(.[^/:]+)(:\d+)?/?.*", url)
        if match:
            return match.group(1)
        else:
            return None
    
    def is_same_domain(self, url1, url2):
        """Checks if two URLs have the same domain.

        Args:
          url1 (str): The first URL.
         url2 (str): The second URL.

        Returns:
          bool: True if the two URLs have the same domain, False otherwise.
        """

        domain1 = self.get_domain(url1)
        domain2 = self.get_domain(url2)

        return domain1 == domain2

    def __init__(self):
        #self.embeddings = OpenAIEmbeddings()
        self.embeddings = HuggingFaceEmbeddings()
        self.db = FAISS.from_texts(["test"], self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.max_urls_per_domain = 10000

    def download_url(self, url):
        print("attempting to download " + url)
        response = requests.get(url)
        print(response.headers['Content-Type'])
        if ('text/html' in response.headers['Content-Type']):
          return response.text
        else:
          return ''

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def crawl(self, url):
        print("crawling url " + url)
        html = self.download_url(url)
        if (not len(html)): return

        soup = BeautifulSoup(html, 'html.parser')
        data = [Document(page_content=soup.get_text(), metadata={"source":url})]
        #print(data.page_content)
        texts = self.text_splitter.split_documents(data)
        #print (f'Now you have {len(texts)} documents')
        self.db.add_documents(texts)

        for url1 in self.get_linked_urls(url, html):
            if (url1 and self.is_same_domain(url, url1)):
                print("queuing up " + url1)
                self.add_url_to_visit(url1)

    def run(self, urls=[]):
        self.visited_urls = []
        self.urls_to_visit = urls
        self.count = 0
        while self.urls_to_visit and self.count <= self.max_urls_per_domain:
            self.count = self.count+1
            if (self.count % 100 == 0): print("Processed " + str(self.count) + " docs")
            url = self.urls_to_visit.pop(0)
            logging.info(f'Crawling: {url}')
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f'Failed to crawl: {url}')
            finally:
                self.visited_urls.append(url)
    
    def finish(self):
        print("creating index")
        self.db.save_local("faiss_index")

if __name__ == '__main__':
    crawler = Crawler();
    urls=['https://www.saratogahigh.org/', 'https://www.saratoga.ca.us/',
          'https://www.saratogachamber.org/', 'https://www.lgsuhsd.org/']
    for url in urls:
        crawler.run([url])
    crawler.finish()

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)
