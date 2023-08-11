
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

    def __init__(self, urls=[]):
        self.visited_urls = []
        self.urls_to_visit = urls
        self.count = 0
        self.embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_texts(["test"], self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    def download_url(self, url):
        print("attempting to download " + url)
        return requests.get(url).text

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

    def run(self):
        while self.urls_to_visit and self.count <= 500:
            self.count = self.count+1
            url = self.urls_to_visit.pop(0)
            logging.info(f'Crawling: {url}')
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f'Failed to crawl: {url}')
            finally:
                self.visited_urls.append(url)
        self.db.save_local("faiss_index")

if __name__ == '__main__':
    Crawler(urls=['https://www.saratogahigh.org/']).run()

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)
