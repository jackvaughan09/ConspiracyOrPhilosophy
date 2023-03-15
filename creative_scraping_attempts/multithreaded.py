import requests
import pandas as pd
import threading


def fetch_data(subreddit: str, params: dict, last_utc: int) -> pd.DataFrame:
    """Fetches a batch of posts from a subreddit using the Pushshift API

    Args:
        subreddit (str): Subreddit to fetch posts from
        params (dict): Dictionary of parameters to pass to the API excluding subreddit and size
        last_utc (int): The latest utc time in the previous batch

    Returns:
        pd.DataFrame: DataFrame containing the fetched posts
    """
    base_url = "https://api.pushshift.io/reddit/search/submission"
    p_string = "&".join([f"{k}={v}" for k, v in params.items()])

    if last_utc:
        # update the p-string to get the next batch
        params["before"] = last_utc
        p_string = "&".join([f"{k}={v}" for k, v in params.items()])

    response = requests.get(
        f"{base_url}?subreddit={subreddit}&size=500&{p_string}"
    ).json()
    batch = pd.DataFrame(response["data"])

    return batch


def get_posts(subreddit: str, n: int, params: dict) -> pd.DataFrame:
    """Download posts from a subreddit using the Pushshift API

    Args:
        subreddit (str): subreddit to scrape
        n (int): number requests to make (500 posts per request)
        params (dict): dictionary of parameters to pass to the API excluding subreddit and size

    Returns:
        pd.DataFrame: DataFrame containing the scraped posts
    """
    lock = threading.Lock()
    dfs = []
    last_utc = None

    def worker():
        nonlocal last_utc
        while len(dfs) < n:
            batch = fetch_data(subreddit, params, last_utc)
            if batch.empty:
                continue

            last_utc = batch["created_utc"].min()

            with lock:
                dfs.append(batch)

    threads = [threading.Thread(target=worker) for _ in range(4)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return pd.concat(dfs).drop_duplicates(subset="title").reset_index(drop=True)


def cleanup(df):
    blank = df.selftext == ""
    removed = df.selftext == "[removed]"
    return df.loc[~(blank | removed)]


if __name__ == "__main__":
    subreddit = "conspiracy"
    n = 50
    params = {"sort_type": "score", "metadata": False, "after": "1y"}

    consp = get_posts(subreddit, n, params)
    consp = consp[["selftext"]]
    consp["news"] = 0
    consp["conspiracy"] = 1
    consp = cleanup(consp)
    print(consp.shape)

    # news = get_posts("conservative", n, params)
    # news = news[["selftext"]]
    # news["news"] = 1
    # news["conspiracy"] = 0
    # news = cleanup(news)
    # print(news.shape)
