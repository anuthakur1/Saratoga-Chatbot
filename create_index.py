# Crawl a set of domains, chunk up the URLs that were crawled, and index
# the content in a vector db.

# Two variables control the crawl - max_urls_per_domain can be modified in Crawler::__init__
# and controls the number of urls to index per domain. The urls list in __main__ lists the
# set of domains to crawl.

# To run:
# python createindex.py

import logging
import re
import requests

from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)

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
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        self.max_urls_per_domain = 10000
        self.seen_chunks = set()

    def download_url(self, url):
        print("attempting to download " + url)

        response = requests.get(url, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        print(content_type)

        if 'text/html' in content_type:
          return response.text
        
        return ""

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path

    def add_url_to_visit(self, url):
        url = urldefrag(url)[0]

        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def extract_clean_text(self, soup):
        for element in soup.select(
            "script, style, nav, header, footer, "
            "aside, form, noscript, [role='navigation']"
        ):
            element.decompose()

        main_content = (
            soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find("article")
            or soup.body
            or soup
        )

        text = main_content.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def crawl(self, url):
        print("crawling url " + url)

        html = self.download_url(url)

        if not html:
            return

        soup = BeautifulSoup(html, 'html.parser')
        clean_text = self.extract_clean_text(soup)

        if clean_text:
            data = [
                Document(
                    page_content=clean_text,
                    metadata={"source": url}
                )
            ]

            texts = self.text_splitter.split_documents(data)

            unique_texts = []

            for doc in texts:
                normalized_text = " ".join(
                    doc.page_content.split()
                ).lower()

                if normalized_text and normalized_text not in self.seen_chunks:
                    self.seen_chunks.add(normalized_text)
                    unique_texts.append(doc)

            if unique_texts:
                if self.db is None:
                    self.db = FAISS.from_documents(
                        unique_texts, self.embeddings
                    )
                else:
                    self.db.add_documents(unique_texts)

            print(
                f"Indexed {len(unique_texts)} of "
                f"{len(texts)} chunks"
            )

        for url1 in self.get_linked_urls(url, html):
            if url1 and self.is_same_domain(url, url1):
                print("queuing up " + url1)
                self.add_url_to_visit(url1)

    def run(self, urls):
        self.visited_urls = []
        self.urls_to_visit = list(urls)
        self.count = 0

        while self.urls_to_visit and self.count < self.max_urls_per_domain:
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
        if self.db is None:
            raise RuntimeError("No documents were indexed.")
        
        print("creating index")
        print("Total indexed chunks:", self.db.index.ntotal)

        self.db.save_local("faiss_index")

if __name__ == '__main__':
    crawler = Crawler();
    urls=['https://www.saratogahigh.org/', 'https://www.saratoga.ca.us/',
          'https://www.saratogachamber.org/', 'https://www.lgsuhsd.org/',
          'https://www.saratogausd.org/']
    for url in urls:
        crawler.run([url])
    crawler.finish()
