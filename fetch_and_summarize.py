import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
import os
from mistralai.client import MistralClient  # Use Mistral's official client

# 1. Scrape
def scrape_ai_news():
    urls = ["https://therundown.ai/", "https://venturebeat.com/category/ai/", "https://techcrunch.com/"]
    articles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract headlines/links (adjust selectors per site)
        for headline in soup.select('h2.entry-title a'):
            articles.append({"title": headline.text, "url": headline['href']})
    return articles[:5]  # Limit to 5 articles

# 2. Summarize with Mistral
def summarize_article(url):
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    response = client.chat(model="mistral-tiny", messages=[
        {"role": "user", "content": f"Summarize this AI news article in 2 sentences: {url}"}
    ])
    return response.choices[0].message.content

# 3. Format and send email
def send_digest(articles):
    body = "\n".join([f"- {article['title']}: {summarize_article(article['url'])}\n{article['url']}"
                     for article in articles])
    msg = MIMEText(body)
    msg['Subject'] = "Your Daily AI News Digest"
    msg['From'] = os.getenv("GMAIL_USER")
    msg['To'] = os.getenv("GMAIL_TO")

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(os.getenv("GMAIL_USER"), os.getenv("GMAIL_PASSWORD"))
        server.send_message(msg)

# Main
if __name__ == "__main__":
    articles = scrape_ai_news()
    send_digest(articles)
