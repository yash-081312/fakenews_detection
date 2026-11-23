"""
link_extractor.py
Module for extracting textual content and metadata from:
1. Wikipedia URLs
2. YouTube URLs (video title, description, and captions/transcripts)
3. General News / Webpage URLs
"""

import re
import urllib.parse
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def extract_youtube_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from various YouTube URL formats.
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def extract_from_youtube(url: str) -> dict:
    """
    Extracts title, channel, description, and transcript from a YouTube URL.
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL: Could not find video ID.")

    title = "YouTube Video"
    author_name = "Unknown Channel"
    
    # 1. Fetch title and author from YouTube oEmbed API
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        resp = requests.get(oembed_url, headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("title", title)
            author_name = data.get("author_name", author_name)
    except Exception as e:
        print(f"[YouTube] oEmbed fetch warning: {e}")

    # 2. Attempt to fetch captions/transcript
    transcript_text = ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        # Try fetching transcript
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        # Find English or first available transcript
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en'])
        except Exception:
            # Fallback to any generated transcript
            for t in transcript_list:
                transcript = t
                break
        
        if transcript:
            data = transcript.fetch()
            # youtube_transcript_api returns dicts with 'text'
            transcript_text = " ".join([item.get('text', '') for item in data])
    except Exception as e:
        print(f"[YouTube] Transcript API note: {e}")

    # Combine text
    combined_content = f"{title}. "
    if transcript_text:
        combined_content += transcript_text
    else:
        combined_content += f"Video published by {author_name}. Analysis based on title and video metadata."

    return {
        "source_type": "YouTube Video",
        "url": url,
        "title": title,
        "author": author_name,
        "video_id": video_id,
        "has_transcript": bool(transcript_text),
        "content": combined_content,
        "preview": combined_content[:500] + ("..." if len(combined_content) > 500 else "")
    }


def extract_from_wikipedia(url: str) -> dict:
    """
    Extracts article title, summary, and main body text from a Wikipedia URL.
    """
    parsed = urllib.parse.urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    if "wiki" in path_parts:
        title_slug = path_parts[path_parts.index("wiki") + 1]
    else:
        title_slug = path_parts[-1]

    title_clean = urllib.parse.unquote(title_slug).replace("_", " ")

    # 1. Try official Wikipedia REST API for clean summary
    summary_text = ""
    try:
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title_slug)}"
        resp = requests.get(api_url, headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            title_clean = data.get("title", title_clean)
            summary_text = data.get("extract", "")
    except Exception as e:
        print(f"[Wikipedia] REST API note: {e}")

    # 2. Scrape full paragraphs from article HTML
    body_text = ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove references, infoboxes, navboxes, edit links
            for tag in soup(["script", "style", "table", "sup", "nav"]):
                tag.decompose()
            
            content_div = soup.find("div", {"id": "mw-content-text"})
            if content_div:
                paragraphs = content_div.find_all("p")
                body_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
                body_text = " ".join(body_parts[:8])  # Top 8 paragraphs
    except Exception as e:
        print(f"[Wikipedia] HTML fetch note: {e}")

    final_content = f"{title_clean}. "
    if summary_text:
        final_content += f"{summary_text} "
    if body_text and body_text != summary_text:
        final_content += body_text

    return {
        "source_type": "Wikipedia Article",
        "url": url,
        "title": title_clean,
        "author": "Wikipedia Contributors",
        "content": final_content.strip(),
        "preview": final_content[:500] + ("..." if len(final_content) > 500 else "")
    }


def extract_from_webpage(url: str) -> dict:
    """
    Extracts article title, lead text, and body paragraphs from any generic web or news URL.
    """
    domain = urllib.parse.urlparse(url).netloc

    resp = requests.get(url, headers=HEADERS, timeout=12)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content elements
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
        element.decompose()

    # Extract Title
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Extract Description
    description = ""
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        description = og_desc["content"].strip()
    elif soup.find("meta", attrs={"name": "description"}):
        description = soup.find("meta", attrs={"name": "description"}).get("content", "").strip()

    # Extract Body Paragraphs
    paragraphs = soup.find_all("p")
    text_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40]
    body_text = " ".join(text_parts[:15])  # Top 15 paragraphs

    content = f"{title}. "
    if description:
        content += f"{description} "
    content += body_text

    return {
        "source_type": f"Web Article ({domain})",
        "url": url,
        "domain": domain,
        "title": title or f"Article from {domain}",
        "author": domain,
        "content": content.strip(),
        "preview": content[:500] + ("..." if len(content) > 500 else "")
    }


def extract_content_from_url(url: str) -> dict:
    """
    Auto-detects the URL type (YouTube, Wikipedia, or generic Webpage) and extracts text.
    """
    url = url.strip()
    lower_url = url.lower()

    if "youtube.com" in lower_url or "youtu.be" in lower_url:
        return extract_from_youtube(url)
    elif "wikipedia.org" in lower_url:
        return extract_from_wikipedia(url)
    elif lower_url.startswith("http://") or lower_url.startswith("https://"):
        return extract_from_webpage(url)
    else:
        raise ValueError("Please provide a valid URL starting with http:// or https://")


if __name__ == "__main__":
    # Test with Wikipedia
    wiki_test = "https://en.wikipedia.org/wiki/Fake_news"
    print("Testing Wikipedia extraction...")
    res = extract_from_wikipedia(wiki_test)
    print("Title:", res["title"])
    print("Preview:", res["preview"][:200])
