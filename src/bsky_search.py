"""Historic search of posts on Bluesky.
"""
import requests
import json
import pandas as pd
from typing import Optional, List, Dict, Tuple

# Base URL for the API calls
BASE_URL: str = 'https://api.bsky.app/xrpc/app.bsky.feed.searchPosts'


def parse_post(post: Dict) -> Dict:
    """Parses a post dictionary to extract specific fields, including embedded images and URLs.

    Args:
        post (Dict): The original post dictionary containing various details.

    Returns:
        Dict: A dictionary containing the parsed post details, including:
            - cid: The content ID of the post.
            - uri: The AT URI of the post.
            - author_displayName: The display name of the author.
            - author_handle: The handle of the author.
            - author_did: The decentralized identifier (DID) of the author.
            - createdAt: The creation timestamp of the post.
            - langs: The languages used in the post.
            - text: The text content of the post.
            - replyCount: The number of replies to the post.
            - repostCount: The number of reposts of the post.
            - likeCount: The number of likes on the post.
            - quoteCount: The number of quotes of the post.
            - reply_parent_cid: The content ID of the parent post in a reply.
            - reply_root_cid: The content ID of the root post in a reply.
            - image_thumbnails: List of thumbnail URLs for embedded images.
            - image_fullsizes: List of full-size URLs for embedded images.
            - urls: List of URLs from external embeds and facets.
    """
    parsed_post: Dict = {
        "cid": post.get("cid"),
        "uri": post.get("uri"),
        "author_displayName": post.get("author", {}).get("displayName"),
        "author_handle": post.get("author", {}).get("handle"),
        "author_did": post.get("author", {}).get("did"),
        "createdAt": post.get("record", {}).get("createdAt"),
        "langs": post.get("record", {}).get("langs"),
        "text": post.get("record", {}).get("text"),
        "replyCount": post.get("replyCount"),
        "repostCount": post.get("repostCount"),
        "likeCount": post.get("likeCount"),
        "quoteCount": post.get("quoteCount"),
        "reply_parent_cid": post.get("record", {}).get("reply", {}).get("parent", {}).get("cid"),
        "reply_root_cid": post.get("record", {}).get("reply", {}).get("root", {}).get("cid"),
        # new fields for media and links
        "image_thumbnails": [],
        "image_fullsizes": [],
        "urls": []
    }

    # Extract embedded images (thumbnails and full-size) from the top-level embed
    for embed_img in post.get("embed", {}).get("images", []):
        thumb = embed_img.get("thumb")
        full = embed_img.get("fullsize")
        if thumb:
            parsed_post["image_thumbnails"].append(thumb)
        if full:
            parsed_post["image_fullsizes"].append(full)

    # Extract external URLs and images from record embed
    record_embed = post.get("record", {}).get("embed", {})
    if record_embed.get("$type") == "app.bsky.embed.external":
        external = record_embed.get("external", {})
        uri = external.get("uri")
        if uri:
            parsed_post["urls"].append(uri)
    elif record_embed.get("$type") == "app.bsky.embed.images":
        for img in record_embed.get("images", []):
            # top-level images are blobs; URLs may need to be constructed by client
            # but include alt and raw ref for now
            ref = img.get("image", {}).get("ref")
            if ref:
                parsed_post["urls"].append(ref.get("$link"))

    # Extract URLs from facets in record
    for facet in post.get("record", {}).get("facets", []):
        for feature in facet.get("features", []):
            uri = feature.get("uri")
            if uri:
                parsed_post["urls"].append(uri)
    return parsed_post


def fetch_all_posts(q: str, limit: int = 100, since: Optional[str] = None,
                    until: Optional[str] = None) -> Tuple[List[dict], Tuple[List[dict]]]:
    """Fetches all posts matching the query parameters.

    Args:
        q (str): The search query string.
        limit (int, optional): The maximum number of posts to fetch per request. Defaults to 100.
        since (Optional[str], optional): The start date for fetching posts (ISO 8601 format). Defaults to None.
        until (Optional[str], optional): The end date for fetching posts (ISO 8601 format). Defaults to None.

    Returns:
        Tuple[List[dict], Tuple[List[dict]]]: A tuple containing two lists:
            - The first list contains the raw post data as dictionaries.
            - The second list contains the parsed post data as dictionaries.
    """
    next_cursor: Optional[int] = None
    raw_results: List[Dict] = []
    parsed_results: List[Dict] = []

    # Perform the search in a loop until we've fetched all posts
    while True:
        # build parameter dict
        params: Dict[str, str] = {'q': q, 'limit': limit}
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until

        # in case the cursor is set, set the parameter as well
        if next_cursor is not None:
            # params["cursor"] = next_cursor  # ! cursor is deactivated because of a bug
            params["cursor"] = None  # TODO: this will need to be fixed once the API bug is fixed 

        # main data gathering
        try:
            # perform the GET request
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data: Dict = response.json()

            # print and store the response data
            raw_posts: List[Dict] = data.get("posts", [])
            raw_results.extend(raw_posts)
            parsed_posts: List[Dict] = [parse_post(post) for post in raw_posts]
            # for post in parsed_posts:
            #     print(f'{post["author_displayName"]} @{post["createdAt"]}: {post["text"].strip()}')
            parsed_results.extend(parsed_posts)

            # Check for the next cursor
            next_cursor = data.get("cursor")

            # If there's no next cursor, we've reached the end
            # ! deactivated due to API bug
            # if not next_cursor:
            #     break
            break  # TODO: fix this after API bug is fixed

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f'An error occurred: {e}')

    return raw_results, parsed_results


# Main entry point for running the "crawling" process
if __name__ == '__main__':
    # Prompt the user for a search term
    search_term: str = input('Enter the query: ')
    raw_posts, parsed_posts = fetch_all_posts(search_term)

    # Save the raw results to a JSON file
    with open('bsky_posts.json', 'w') as f:
        json.dump(raw_posts, f, indent=4)

    # Save the parsed results to a csv file
    with open('bsky_posts.csv', 'w') as f:
        df = pd.DataFrame(parsed_posts)
        df.to_csv(f, index=False)

    print(f'Fetched {len(parsed_posts)} posts. Results saved to bsky_posts.json.')
